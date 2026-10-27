"""Loops: `Loop` (0..extent) and `StridedLoop` (start..end:step), and how each is laid out.

One decision procedure picks the form of every loop (see `LoopLowering.loop`):

1. Known trip count, small enough: unrolled completely. The loop variable is a constant in
   every copy, so its index arithmetic folds into address offsets. A loop whose variable
   picks a tensor-core register fragment (`_b[_r0]`) is unrolled at any size and level.
2. Trip count only known to lie in a small range (a cooperative loop `for i in tid..N:step`,
   where `tid` is a thread index): unrolled completely, with an exit check before each copy
   that might not run (`short_loop`).
3. Otherwise a runtime loop, tested at the bottom. A plain-computation body is unrolled so one
   trip issues about LOADS_IN_FLIGHT loads, and a remainder loop finishes the last iterations.
   In front of it, loads that never change across iterations are issued once (`hoist`).

Optimization level 1 restricts (1) to loops the IR marks `unroll`; level 0 turns off every
optional unrolling and the hoisting, for the smallest PTX.
"""

from __future__ import annotations

import math

from .ptx import Val, const_f32, const_int

# Loops the IR marks `unroll` are unrolled completely up to this trip count and this many
# statements of unrolled code; other loops over plain computation up to FULL_UNROLL_STMTS.
UNROLL_MAX = 128
FLAGGED_UNROLL_STMTS = 4096
FULL_UNROLL_STMTS = 512
# Runtime loops over bodies of at most UNROLL_BODY_MAX statements are unrolled so one trip
# issues about LOADS_IN_FLIGHT independent loads: with few threads per SM, memory-level
# parallelism inside each thread is what hides DRAM latency.
LOADS_IN_FLIGHT = 16
UNROLL_BODY_MAX = 40
# A loop whose trip count is only known to lie in a range is unrolled up to this many copies.
SHORT_LOOP_MAX = 16
# Loads hoisted in front of one runtime loop, in registers.
HOIST_MAX = 128

# Statements a body must not contain to be unrolled at will.
_NOT_UNROLLABLE = {"Sync", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "WarpShuffle", "TreeHalve", "Smem"}


def children(stmt) -> list:
    """The nested statement lists of a block statement (Tile, Loop, StridedLoop, Cond)."""
    out = []
    for attr in ("body", "else_body"):
        sub = getattr(stmt, attr, None)
        if sub is not None and not isinstance(sub, (str, bytes)):
            out.append(sub)
    return out


def walk(body):
    """Every statement in `body`, nested ones included."""
    for s in body:
        yield s
        for sub in children(s):
            yield from walk(sub)


def defines(stmt) -> tuple[str, ...]:
    """The names a leaf statement binds."""
    kind = type(stmt).__name__
    if kind == "Load":
        return tuple(stmt.names)
    if kind in ("Assign", "Let", "Accum", "Init"):
        return (stmt.name,)
    return ()


class LoopLowering:
    """Mixin: the loop statements. Part of `KernelCompiler`."""

    def visit_Loop(self, s) -> None:
        """for axis in 0..extent."""
        self._seed_accumulators(s)
        extent = int(s.axis.extent.as_static())
        self.loop(s.axis.name, const_int(0), const_int(extent), const_int(1), s.body, s.unroll)

    def visit_StridedLoop(self, s) -> None:
        """for axis in start..end:step (each cooperating thread strides the axis)."""
        self._seed_accumulators(s)
        start = self.to_s32(self.expr(s.start))
        step = const_int(s.step) if isinstance(s.step, int) else self.to_s32(self.expr(s.step))
        if s.end is not None:
            end = self.to_s32(self.expr(s.end))
        else:
            end = const_int(int(s.axis.extent.as_static()))
        self.loop(s.axis.name, start, end, step, s.body, s.unroll)

    def _seed_accumulators(self, loop) -> None:
        """Each Accum directly inside a loop starts at its op's identity, just before the loop."""
        if not getattr(loop, "seed", True):
            return
        seen = set()
        for s in loop.body:
            if type(s).__name__ == "Accum" and s.name not in seen:
                seen.add(s.name)
                identity = s.op.identity
                self.copy_into(self.define(s.name, "f32"), const_f32(0.0 if identity is None else identity))

    # ---- choosing the form ----------------------------------------------------------------
    def loop(self, var: str, start: Val, end: Val, step: Val, body, unroll: bool) -> None:
        """Emit `for (var = start; var < end; var += step) body` in the best fitting form."""
        saved = self.env.get(var)
        trips = None
        if start.is_const and end.is_const and step.is_const and step.const > 0:
            trips = max(0, -(-(end.const - start.const) // step.const))
        if trips is not None and (var in self.fragment_index_vars(body) or self.fully_unrolled(trips, body, unroll)):
            for k in range(trips):
                self.env[var] = const_int(start.const + k * step.const)
                self.body(body)
        elif not self._short_loop(var, start, end, step, body):
            self._runtime_loop(var, start, end, step, body, trips)
        if saved is None:
            self.env.pop(var, None)
        else:
            self.env[var] = saved

    def fully_unrolled(self, trips: int, body, unroll: bool) -> bool:
        """Whether a loop with a known trip count is unrolled completely."""
        if trips <= 2:
            return True
        if self.level == 0:
            return False
        size = trips * self._unrolled_size(body)
        if self.level == 1:
            return unroll and trips <= UNROLL_MAX and size <= FULL_UNROLL_STMTS
        if unroll and trips <= UNROLL_MAX and size <= FLAGGED_UNROLL_STMTS:
            return True
        return self.unroll_factor(body) > 1 and size <= FULL_UNROLL_STMTS

    def _unrolled_size(self, body) -> int:
        """Statements `body` expands to, counting the copies of inner loops that unroll."""
        size = 0
        for t in body:
            size += 1
            if type(t).__name__ == "Loop":
                trips = int(t.axis.extent.as_static())
                inner = self._unrolled_size(t.body)
                size += inner * (trips if self.fully_unrolled(trips, t.body, t.unroll) else 1)
            else:
                size += sum(self._unrolled_size(sub) for sub in children(t))
        return size

    def unroll_factor(self, body) -> int:
        """Copies of a runtime loop's body per trip: a power of two, 1 for no unrolling."""
        stmts = list(walk(body))
        if self.level == 0 or len(stmts) > UNROLL_BODY_MAX or any(type(s).__name__ in _NOT_UNROLLABLE for s in stmts):
            return 1
        loads = sum(type(s).__name__ == "Load" for s in stmts)
        factor = 2
        while factor * 2 * max(loads, 1) <= LOADS_IN_FLIGHT:
            factor *= 2
        return factor

    # ---- the forms --------------------------------------------------------------------------
    def _short_loop(self, var: str, start: Val, end: Val, step: Val, body) -> bool:
        """Unroll a loop whose trip count depends on the thread but is known to be small.

        A cooperative loop `for i in tid..N:step` runs between tmin and tmax times, from the
        range of `tid`. The first tmin copies run unconditionally; each later copy first
        checks `i < end` and jumps past the rest when it fails. Returns False when the loop
        is not of this kind.
        """
        if not (step.is_const and step.const > 0 and end.is_const) or self.unroll_factor(body) == 1:
            return False
        r = self.range_of(start)
        if r is None:
            return False
        tmax = max(0, -(-(end.const - r[0]) // step.const))
        tmin = max(0, -(-(end.const - r[1]) // step.const))
        if tmax > SHORT_LOOP_MAX:
            return False
        em = self.em
        done = em.label("short_end")
        with self.block():
            for k in range(tmax):
                v = self.iop("+", start, const_int(k * step.const))
                if k >= tmin:
                    p = em.reg("pred")
                    em.emit(f"setp.ge.s32 {p}, {v}, {end}")
                    em.emit(f"@{p} bra {done}")
                self.env[var] = v
                self.body(body)
        em.place(done)
        return True

    def _runtime_loop(self, var: str, start: Val, end: Val, step: Val, body, trips: int | None) -> None:
        hoist = trips is not None and trips >= 1 and self.level >= 1
        if hoist:
            self._hoist_invariant_loads(var, body)
        i = self.em.reg("s32")
        self.copy_into(i, start)
        r = self.range_of(start)
        if r is not None and end.is_const:                # inside the body: start <= i < end
            self.ranges[i.text] = (r[0], end.const - 1)
        if step.is_const:                                 # i = start + n * step, for every n
            m = math.gcd(self.mult_of(start), int(step.const))
            if m > 1:
                self.mults[i.text] = m
        factor = self.unroll_factor(body) if step.is_const else 1
        if trips is not None and trips < 2 * factor:
            factor = 1
        if factor > 1:
            self._rotated_loop(var, i, end, step, body, factor)
        if factor == 1 or trips is None or trips % factor:
            self._rotated_loop(var, i, end, step, body, 1)
        if hoist:
            self.hoisted.pop()

    def _rotated_loop(self, var: str, i: Val, end: Val, step: Val, body, factor: int) -> None:
        """Run `body` `factor` times per trip while i + (factor-1)*step < end, testing at the bottom.

            @(i + (factor-1)*step >= end) bra done      // enter only if a full trip fits
          top:
            body(i), body(i + step), ...                // `factor` copies
            i += factor * step
            @(i + (factor-1)*step < end) bra top        // one branch per trip
          done:
        """
        em = self.em
        reach = (factor - 1) * step.const if factor > 1 else 0
        name = f"loop_x{factor}" if factor > 1 else "loop"
        top, done = em.label(name), em.label(name + "_end")

        def test(cmp: str, target: str) -> None:
            with self.block():
                last = self.iop("+", i, const_int(reach))
                p = em.reg("pred")
                em.emit(f"setp.{cmp}.s32 {p}, {last}, {end}")
                em.emit(f"@{p} bra {target}")

        test("ge", done)
        em.place(top)
        if factor == 1:
            em.emit('.pragma "nounroll"')                 # remainder loops stay small
        with self.block():
            for k in range(factor):
                self.env[var] = self.iop("+", i, const_int(k * step.const)) if factor > 1 else i
                self.body(body)
        self.forget(i)
        em.emit(f"add.s32 {i}, {i}, {factor * step.const if factor > 1 else step}")
        test("lt", top)
        em.place(done)

    # ---- loop-invariant loads ---------------------------------------------------------------
    def _hoist_invariant_loads(self, var: str, body) -> None:
        """Load, in front of a runtime loop, the values its body reads that never change.

        A load is invariant when it reads a global buffer the kernel never writes, at an index
        built only from names that exist before the loop and from the variables of inner loops
        that are unrolled completely. Each such load is issued once here; `hoisted_load` later
        finds its address in `self.hoisted` instead of loading again every iteration. Only for
        loops known to run at least once, so no load runs that would not have.
        """
        inner_defs = {var}.union(*(defines(t) for t in walk(body)))
        found: list[tuple] = []

        def scan(stmts, bound: dict) -> None:
            for t in stmts:
                kind = type(t).__name__
                if kind == "Load":
                    buf = self.buffers.get(t.input)
                    free = set().union(*(e.free_vars() for e in t.index)) - set(bound)
                    if (buf is not None and buf.space == "global" and buf.dtype == "f32"
                            and t.input not in self.written and not free & inner_defs and free <= set(self.env)):
                        found.append((t, dict(bound)))
                elif kind == "Loop":
                    trips = int(t.axis.extent.as_static())
                    if self.fully_unrolled(trips, t.body, t.unroll):
                        for k in range(trips):
                            scan(t.body, {**bound, t.axis.name: k})

        scan(body, {})
        table: dict = {}
        budget = HOIST_MAX
        for load, bound in found:
            n = len(load.names)
            if n not in (1, 2, 4) or budget < n:
                continue
            saved = {name: self.env.get(name) for name in bound}
            self.env.update({name: const_int(k) for name, k in bound.items()})
            buf = self.buffers[load.input]
            flat = self.flat_index(buf, load.index)
            addr = self.address(buf, flat)
            key = (buf.name, addr, n)
            # A vector load not proven aligned stays in the loop, where visit_Load reports it.
            if key not in table and (n == 1 or self.known_alignment(buf, flat) >= n * buf.esize):
                regs = [self.em.reg("f32") for _ in load.names]
                width = f".v{n}" if n > 1 else ""
                self.em.emit(f"ld.global{width}.f32 {{{', '.join(map(str, regs))}}}, {addr}"
                             if n > 1 else f"ld.global.f32 {regs[0]}, {addr}")
                table[key] = regs
                budget -= n
            for name, old in saved.items():
                if old is None:
                    self.env.pop(name, None)
                else:
                    self.env[name] = old
        self.hoisted.append(table)

    def hoisted_load(self, s) -> bool:
        """If this load was moved in front of an enclosing loop, bind its names and say so."""
        if not self.hoisted or s.input not in self.buffers:
            return False
        buf = self.buffers[s.input]
        if buf.space != "global":
            return False
        addr = self.address(buf, self.flat_index(buf, s.index))
        for table in reversed(self.hoisted):
            regs = table.get((buf.name, addr, len(s.names)))
            if regs is not None:
                self.env.update(zip(s.names, regs))
                return True
        return False
