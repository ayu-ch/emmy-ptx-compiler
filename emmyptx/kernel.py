"""Kernel IR -> PTX: the compiler proper.

`KernelCompiler` walks one Emmy `KernelOp` and emits a PTX entry point. It is organized in
three layers:

1. Buffers and addressing - every array the kernel touches (the pointer parameters in
   global memory, and the `Smem` arrays in shared memory), and `address()`, which turns
   "element `i` of buffer B" into a PTX address operand.
2. Expressions - `expr()` compiles Emmy's index/predicate trees (`Var`, `Literal`,
   `BinaryExpr`, `TernaryExpr`, ...) into `Val`s, folding constants as it goes.
3. Statements - one `visit_<Kind>` method per Kernel IR statement.

Naming model: every IR name (SSA value, accumulator, axis variable) maps to one `Val` in
`self.env`. Redefining a name overwrites its register, which is exactly the semantics the IR
needs for accumulators (`acc0 <- add(acc0, x)`), loop-carried state and the in-place state
updates of `WarpShuffle` / `TreeHalve` combines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ops import lower_int_op, lower_op
from .ptx import Emitter, Val, const_f32, const_int, const_pred

# Element types of buffers: bytes per element and the PTX type used to load/store them.
ELEM = {"f32": (4, "f32"), "f16": (2, "b16"), "i32": (4, "s32")}
SMEM_DTYPE = {"float": "f32", "__half": "f16", "half": "f16", "int": "i32"}
INT_DTYPES = {"int", "i32", "i64", "int32", "int64", "u32", "u8", "i8"}
# `bool` values (comparison results) are kept as 1.0 / 0.0 floats, Emmy's mask convention.

# Loops the IR marks `unroll` are unrolled at compile time up to this trip count, which turns
# their index arithmetic into constants (e.g. matmul's shared-memory reads become [reg+imm]).
UNROLL_MAX = 128
# Runtime loops over plain computation (bodies up to UNROLL_BODY_MAX statements) are unrolled
# so that one trip issues about LOADS_IN_FLIGHT independent loads: with few threads per SM,
# memory-level parallelism inside each thread is what hides DRAM latency.
LOADS_IN_FLIGHT = 16
UNROLL_BODY_MAX = 40
# A loop whose trip count is only known to lie in a range is fully unrolled up to this many copies.
SHORT_LOOP_MAX = 16
# Constant-trip loops are unrolled completely while the unrolled code stays under this many
# statements: FLAGGED_UNROLL_STMTS when the IR marks the loop `unroll`, FULL_UNROLL_STMTS for
# other loops over plain computation.
FLAGGED_UNROLL_STMTS = 4096
FULL_UNROLL_STMTS = 512
# The runner accepts PTX up to 2,000,000 characters; past this the kernel is recompiled lean.
PTX_SIZE_LIMIT = 1_800_000
# Loads that do not change across a runtime loop are moved in front of it, up to this many registers.
HOIST_MAX = 128

# Shared-memory XOR swizzles: mode -> (element shift, row mask), as in Emmy's slab layouts.
SWIZZLE_XOR = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}


class CompileError(NotImplementedError):
    pass


@dataclass
class Buffer:
    """An array the kernel reads or writes."""

    name: str
    space: str               # "global" or "shared"
    base: Val                # global: a u64 register; shared: an s32 register with its address
    shape: tuple[int, ...]
    dtype: str               # "f32", "f16" or "i32"
    align: int = 16          # known byte alignment of element 0

    @property
    def esize(self) -> int:
        return ELEM[self.dtype][0]


def _c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero), for constant folding."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


_INT_INSTR = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32", "/": "div.s32", "//": "div.s32",
              "%": "rem.s32", "<<": "shl.b32", ">>": "shr.s32", "&": "and.b32", "|": "or.b32",
              "^": "xor.b32", "max": "max.s32", "min": "min.s32"}
_INT_FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
             "/": _c_div, "//": _c_div, "%": lambda a, b: a - b * _c_div(a, b),
             "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b, "&": lambda a, b: a & b,
             "|": lambda a, b: a | b, "^": lambda a, b: a ^ b, "max": max, "min": min}
_FLOAT_INSTR = {"+": "add.f32", "-": "sub.f32", "*": "mul.f32", "/": "div.rn.f32",
                "max": "max.f32", "min": "min.f32"}
_COMMUTATIVE = {"+", "*", "&", "|", "^", "max", "min", "==", "!="}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_CMP_FOLD = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
             ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_CMP_SWAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
# Elementwise names used by FuncCallExpr -> the op names `ops.lower_op` knows.
_FUNC_ALIASES = {"fmax": "maximum", "fmin": "minimum", "fabs": "abs"}


def _children(stmt) -> list:
    """The nested statement lists of a block statement (Tile, Loop, StridedLoop, Cond)."""
    out = []
    for attr in ("body", "else_body"):
        sub = getattr(stmt, attr, None)
        if sub is not None and not isinstance(sub, (str, bytes)):
            out.append(sub)
    return out


def _walk(body):
    for s in body:
        yield s
        for sub in _children(s):
            yield from _walk(sub)


def _defines(stmt) -> tuple[str, ...]:
    """The names a leaf statement binds."""
    kind = type(stmt).__name__
    if kind == "Load":
        return tuple(stmt.names)
    if kind in ("Assign", "Let", "Accum", "Init"):
        return (stmt.name,)
    return ()


def _dtype_name(dt) -> str | None:
    if dt is None:
        return None
    return getattr(dt, "name", dt)


class KernelCompiler:
    def __init__(self, graph, kernel, launch: dict, target: str = "sm_89", lean: bool = False) -> None:
        self.kernel = kernel
        self.lean = lean                           # no optional unrolling or hoisting
        self.launch = launch
        self.target = target
        self.em = Emitter()
        self.env: dict[str, Val] = {}
        self.buffers: dict[str, Buffer] = {}
        self.smem_decls: list[str] = []
        self.smem_for: dict[int, Buffer] = {}      # id(Smem stmt) -> its buffer
        self.guarded = False                       # some launched threads exit early
        self.exit_label = "$L_exit"
        # Integer common-subexpression cache, one dict per open block: a value computed in a
        # loop body or a branch is forgotten when that block closes, so it is never reused
        # where it was not computed.
        self.cse: list[dict] = [{}]
        # Registers known to equal `base + constant` (register text -> (base, constant)), so
        # addresses can fold the constant into the `[reg+imm]` operand.
        self.affine: dict[str, tuple[Val, int]] = {}
        # Known value ranges of integer registers (register text -> (lo, hi)), from the tile
        # decode onward. They let a loop whose start depends on the thread still know its trip count.
        self.ranges: dict[str, tuple[int, int]] = {}
        # Known divisors of integer registers (register text -> d with value % d == 0), which
        # prove the alignment that merging two scalar loads into one vector load needs.
        self.mults: dict[str, int] = {}
        # Buffers the kernel writes (they are never read ahead of time), and the stack of loads
        # moved in front of the runtime loops being compiled: (buffer, address, width) -> registers.
        self.written = {getattr(t, "output", None) or getattr(t, "smem", None)
                        for t in _walk(kernel.body) if type(t).__name__ in ("Write", "CpAsyncCopy")}
        self.hoisted: list[dict] = []
        # Names whose value is carried across iterations or combine steps: they keep one register.
        self.carried: set[str] = set()
        for s in _walk(kernel.body):
            kind = type(s).__name__
            if kind in ("Accum", "Init"):
                self.carried.add(s.name)
            elif kind in ("WarpShuffle", "TreeHalve"):
                self.carried.update(s.state)
        # Every tensor the graph knows: name -> (shape, dtype). Scalars count as one element.
        self.tensors = {}
        for node in graph.nodes.values():
            for t in node.outputs:
                shape = tuple(d.as_static() for d in t.shape) or (1,)
                self.tensors[t.name] = (shape, t.dtype.name)

    # =====================================================================================
    # Entry point
    # =====================================================================================
    def compile(self) -> str:
        params = self._prologue()
        for s in self.kernel.body:
            self.stmt(s)
        body = self.em.lines
        return "\n".join([
            "//",
            f"// {self.kernel.name}: generated by emmyptx from Emmy Kernel IR",
            "//",
            ".version 8.0",
            f".target {self.target}",
            ".address_size 64",
            "",
            *self.smem_decls,
            "",
            f".visible .entry {self.kernel.name}(",
            ",\n".join(params),
            ")",
            "{",
            *self.em.declarations(),
            "",
            *body,
            f"{self.exit_label}:",
            "    ret;",
            "}",
            "",
        ])

    def _prologue(self) -> list[str]:
        """Parameters, buffer base addresses, shared arrays and the thread's lane / warp."""
        em = self.em
        args = {a["name"]: a for a in self.launch.get("args", [])}
        params = []
        for name in self.launch["arg_order"]:
            arg = args.get(name, {})
            if arg.get("scalar"):                         # passed by value, not as a pointer
                ty = "f32" if arg.get("dtype", "f32") == "f32" else "s32"
                params.append(f"    .param .{'f32' if ty == 'f32' else 's32'} {name}")
                r = em.reg(ty)
                em.emit(f"ld.param.{'f32' if ty == 'f32' else 's32'} {r}, [{name}]")
                self.env[name] = r
                continue
            params.append(f"    .param .u64 {name}")
            r = em.reg("u64")
            em.emit(f"ld.param.u64 {r}, [{name}]")
            em.emit(f"cvta.to.global.u64 {r}, {r}")
            if name in self.tensors:
                shape, dtype = self.tensors[name]
            else:
                shape, dtype = tuple(arg.get("shape") or (1,)), arg.get("dtype", "f32")
            if dtype not in ELEM:
                raise CompileError(f"buffer {name!r} has unsupported dtype {dtype!r}")
            self.buffers[name] = Buffer(name, "global", r, tuple(shape), dtype, align=256)  # cudaMalloc
        self._declare_shared()
        tid = em.reg("s32")
        em.emit(f"mov.u32 {tid}, %tid.x")
        lane, warp = em.reg("s32"), em.reg("s32")
        em.emit(f"and.b32 {lane}, {tid}, 31")
        em.emit(f"shr.u32 {warp}, {tid}, 5")
        block = math.prod(self.launch.get("block", [1024]))
        self.ranges[tid.text] = (0, block - 1)
        self.ranges[lane.text] = (0, min(block, 32) - 1)
        self.ranges[warp.text] = (0, (block - 1) // 32)
        self.tid = tid
        self.env["lane"], self.env["warp"] = lane, warp
        return params

    def _declare_shared(self) -> None:
        """Declare every `Smem` array up front (PTX wants them at function scope).

        Up to 48 KiB the arrays are static `.shared` variables. Above that, `meta.json`
        launches with a dynamic allocation (`smem_bytes`), so the arrays are carved out of one
        `.extern .shared` pool instead, exactly as Emmy's own renderer does.
        """
        em = self.em
        smems = [s for s in _walk(self.kernel.body) if type(s).__name__ == "Smem"]
        dynamic = self.launch.get("smem_bytes", 0) > 0
        pool = None
        if dynamic:
            self.smem_decls.append(".extern .shared .align 16 .b8 _smem_pool[];")
            pool = em.reg("s32")
            em.emit(f"mov.u32 {pool}, _smem_pool")
        used: dict[str, int] = {}
        offset = 0
        for s in smems:
            dtype = SMEM_DTYPE.get(s.dtype)
            if dtype is None:
                raise CompileError(f"shared array {s.name!r} has unsupported dtype {s.dtype!r}")
            esize = ELEM[dtype][0]
            nbytes = math.prod(int(e) for e in s.extents) * esize
            align = max(int(s.align or 0), esize, 4)
            sym = s.name if s.name not in used else f"{s.name}_{used[s.name]}"
            used[s.name] = used.get(s.name, 0) + 1
            base = em.reg("s32")
            if dynamic:
                offset = (offset + 15) // 16 * 16
                em.emit(f"add.s32 {base}, {pool}, {offset}")
                offset += nbytes
            else:
                self.smem_decls.append(f".shared .align {align} .b8 {sym}[{nbytes}];")
                em.emit(f"mov.u32 {base}, {sym}")
            self.smem_for[id(s)] = Buffer(s.name, "shared", base, tuple(int(e) for e in s.extents), dtype,
                                          align=16 if dynamic else align)
        if dynamic and offset > self.launch["smem_bytes"]:
            raise CompileError(f"shared arrays need {offset} bytes, launch provides {self.launch['smem_bytes']}")

    # =====================================================================================
    # Names
    # =====================================================================================
    def lookup(self, name) -> Val:
        if isinstance(name, (int, float)):                # a numeric operand in an Assign
            return const_f32(name)
        if name not in self.env:
            raise CompileError(f"use of undefined name {name!r}")
        return self.env[name]

    def define(self, name: str, ty: str) -> Val:
        """The register a new value of `name` is written to.

        Carried names (accumulators, `Init` state, shuffle / tree states) keep one register
        that is updated in place, since their value flows across loop iterations and combine
        steps. Every other definition gets a fresh register: plain SSA form, so independent
        values (e.g. the loads of an unrolled loop) never share a register, and an integer
        register never changes after it is written, which the CSE cache relies on.
        """
        old = self.env.get(name)
        if name in self.carried and old is not None and not old.is_const and old.ty == ty:
            return old
        r = self.em.reg(ty)
        self.env[name] = r
        return r

    def to_f32(self, v: Val) -> Val:
        if v.ty == "f32":
            return v
        if v.is_const:
            return const_f32(float(v.const))
        if v.ty == "pred":
            r = self.em.reg("f32")
            self.em.emit(f"selp.f32 {r}, {const_f32(1.0)}, {const_f32(0.0)}, {v}")
            return r
        r = self.em.reg("f32")
        self.em.emit(f"cvt.rn.f32.s32 {r}, {v}")
        return r

    def to_s32(self, v: Val) -> Val:
        if v.ty == "s32":
            return v
        if v.is_const:
            return const_int(int(v.const))
        r = self.em.reg("s32")
        if v.ty == "pred":
            self.em.emit(f"selp.s32 {r}, 1, 0, {v}")
        else:
            self.em.emit(f"cvt.rzi.s32.f32 {r}, {v}")
        return r

    def to_pred(self, v: Val) -> Val:
        if v.ty == "pred":
            return v
        if v.is_const:
            return const_pred(bool(v.const))
        p = self.em.reg("pred")
        if v.ty == "f32":
            self.em.emit(f"setp.neu.f32 {p}, {v}, {const_f32(0.0)}")
        else:
            self.em.emit(f"setp.ne.s32 {p}, {v}, 0")
        return p

    def copy_into(self, dst: Val, v: Val) -> None:
        if dst.ty == "s32":
            self._forget(dst)
        if dst.text != v.text:
            self.em.emit(f"mov.{'b32' if dst.ty == 's32' else dst.ty} {dst}, {v}")

    # =====================================================================================
    # Expressions
    # =====================================================================================
    def expr(self, e) -> Val:
        kind = type(e).__name__
        if kind == "Literal":
            if isinstance(e.value, bool):
                return const_pred(e.value)
            if _dtype_name(e.dtype) in INT_DTYPES or (isinstance(e.value, int) and _dtype_name(e.dtype) is None):
                return const_int(int(e.value))
            return const_f32(float(e.value))
        if kind == "Var":
            return self.lookup(e.name)
        if kind == "BinaryExpr":
            return self.binary(e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            return self.select(self.to_pred(self.expr(e.cond)), self.expr(e.if_true), self.expr(e.if_false))
        if kind == "CastExpr":
            v = self.expr(e.expr)
            return self.to_s32(v) if e.dtype in INT_DTYPES else self.to_f32(v)
        if kind == "FlatIndex":
            return self.flat_index(self.buffer(e.buffer), e.index)
        if kind == "Builtin":
            return self.builtin(e.name)
        if kind == "FuncCallExpr":
            args = [self.to_f32(self.expr(a)) for a in e.args]
            dst = self.em.reg("f32")
            lower_op(self.em, _FUNC_ALIASES.get(e.name, e.name), args, dst)
            return dst
        raise CompileError(f"unsupported expression {kind}: {e!r}")

    def builtin(self, name: str) -> Val:
        sreg = {"threadIdx.x": "%tid.x", "threadIdx.y": "%tid.y", "threadIdx.z": "%tid.z",
                "blockIdx.x": "%ctaid.x", "blockIdx.y": "%ctaid.y", "blockIdx.z": "%ctaid.z",
                "blockDim.x": "%ntid.x", "blockDim.y": "%ntid.y", "blockDim.z": "%ntid.z",
                "gridDim.x": "%nctaid.x", "gridDim.y": "%nctaid.y", "gridDim.z": "%nctaid.z",
                "warpSize": "WARP_SZ"}.get(name)
        if sreg is None:
            raise CompileError(f"unsupported builtin {name!r}")
        r = self.em.reg("s32")
        self.em.emit(f"mov.u32 {r}, {sreg}")
        return r

    def binary(self, op: str, a: Val, b: Val) -> Val:
        em = self.em
        if op in ("&&", "||"):
            a, b = self.to_pred(a), self.to_pred(b)
            if a.is_const and b.is_const:
                return const_pred(a.const and b.const if op == "&&" else a.const or b.const)
            if a.is_const:
                a, b = b, a
            if b.is_const:                                # x && true = x, x || false = x
                if b.const == (op == "&&"):
                    return a
                return const_pred(not (op == "&&"))
            p = em.reg("pred")
            em.emit(f"{'and' if op == '&&' else 'or'}.pred {p}, {a}, {b}")
            return p
        if op in _CMP:
            if a.ty == "f32" or b.ty == "f32":
                a, b, ty = self.to_f32(a), self.to_f32(b), "f32"
            else:
                a, b, ty = self.to_s32(a), self.to_s32(b), "s32"
            if a.is_const and b.is_const:
                return const_pred(_CMP_FOLD[op](a.const, b.const))
            if a.is_const:
                a, b, op = b, a, _CMP_SWAP[op]
            p = em.reg("pred")
            cmp = _CMP[op] + ("u" if ty == "f32" and op == "!=" else "")
            em.emit(f"setp.{cmp}.{ty} {p}, {a}, {b}")
            return p
        if a.ty == "f32" or b.ty == "f32":               # float arithmetic
            if op not in _FLOAT_INSTR:
                raise CompileError(f"float operator {op!r} is not supported")
            a, b = self.to_f32(a), self.to_f32(b)
            if a.is_const and op in _COMMUTATIVE:
                a, b = b, a
            r = em.reg("f32")
            em.emit(f"{_FLOAT_INSTR[op]} {r}, {em.materialize(a)}, {b}")
            return r
        if a.ty == "pred" and b.ty == "pred" and op == "^":
            p = em.reg("pred")
            em.emit(f"xor.pred {p}, {em.materialize(a)}, {em.materialize(b)}")
            return p
        a, b = self.to_s32(a), self.to_s32(b)
        return self.iop(op, a, b)

    def iop(self, op: str, a: Val, b: Val) -> Val:
        """Integer arithmetic: constant folding, identities, affine splitting and CSE.

        `+`, `-` and `* constant` keep track of each result as `base + constant` (see
        `self.affine`), so `(i + 3) * 64 + c` is recorded as `(i * 64 + c) + 192`. The
        register holding the full value is still emitted (ptxas drops it when unused), but
        `address()` can address `base` with an immediate offset of 192 elements.
        """
        if op not in _INT_INSTR:
            raise CompileError(f"integer operator {op!r} is not supported")
        if a.is_const and b.is_const:
            return const_int(_INT_FOLD[op](a.const, b.const))
        if a.is_const and op in _COMMUTATIVE:
            a, b = b, a
        if b.is_const:
            k = b.const
            if (op in ("+", "-", "|", "^", "<<", ">>") and k == 0) or (op in ("*", "/", "//") and k == 1):
                return a
            if op == "*" and k == 0:
                return const_int(0)
            if op == "%" and k == 1:
                return const_int(0)
        result = self._iop(op, a, b)
        if not result.is_const:
            if result.text not in self.ranges:
                r = self._range_after(op, self.range_of(a), self.range_of(b))
                if r is not None:
                    self.ranges[result.text] = r
            if result.text not in self.mults:
                m = self._mult_after(op, a, b)
                if m > 1:
                    self.mults[result.text] = m
        return result

    def mult_of(self, v: Val) -> int:
        """A known divisor of an integer value (1 when nothing is known)."""
        if v.is_const:
            return abs(int(v.const)) or 1 << 30
        if v.text in self.affine:
            base, k = self.affine[v.text]
            return math.gcd(self.mult_of(base), abs(k) or 1 << 30)
        return self.mults.get(v.text, 1)

    def _mult_after(self, op: str, a: Val, b: Val) -> int:
        ma, mb = self.mult_of(a), self.mult_of(b)
        if op in ("+", "-"):
            return math.gcd(ma, mb)
        if op == "*":
            return min(ma * mb, 1 << 30)
        if op == "<<" and b.is_const:
            return min(ma << int(b.const), 1 << 30)
        return 1

    def _iop(self, op: str, a: Val, b: Val) -> Val:
        if op in ("+", "-"):
            (ra, ka), (rb, kb) = self._split(a), self._split(b)
            sign = 1 if op == "+" else -1
            if rb is None:                                # x +/- constant
                return self._offset(ra, ka + sign * kb)
            if ra is None and op == "+":
                return self._offset(rb, ka + kb)
            if ra is None:                                # constant - x
                return self._offset(self._raw("sub.s32", const_int(0), rb), ka - kb)
            return self._offset(self._raw(_INT_INSTR[op], ra, rb), ka + sign * kb)
        if op == "*" and b.is_const:                      # (base + c) * k = base * k + c * k
            ra, ka = self._split(a)
            return self._offset(self._raw("mul.lo.s32", ra, b), ka * b.const)
        if op in ("/", "//", "%") and b.is_const and b.const > 0:
            r = self.range_of(a)
            if r is not None and r[0] >= 0:               # non-negative: unsigned is cheaper
                k = int(b.const)
                if k & (k - 1) == 0:                      # a power of two: a shift or a mask
                    if op == "%":
                        return self._raw("and.b32", a, const_int(k - 1))
                    return self._raw("shr.u32", a, const_int(k.bit_length() - 1))
                return self._raw("rem.u32" if op == "%" else "div.u32", a, b)
        return self._raw(_INT_INSTR[op], a, b)

    def range_of(self, v: Val) -> tuple[int, int] | None:
        """The known (lo, hi) of an integer value, or None."""
        if v.is_const:
            return int(v.const), int(v.const)
        if v.text in self.ranges:
            return self.ranges[v.text]
        if v.text in self.affine:
            base, k = self.affine[v.text]
            r = self.range_of(base)
            return (r[0] + k, r[1] + k) if r else None
        return None

    @staticmethod
    def _range_after(op: str, ra, rb) -> tuple[int, int] | None:
        """Interval arithmetic for the integer operators whose result range is easy to bound."""
        if ra is None or rb is None:
            return None
        if op == "+":
            return ra[0] + rb[0], ra[1] + rb[1]
        if op == "-":
            return ra[0] - rb[1], ra[1] - rb[0]
        if op == "*":
            corners = [x * y for x in ra for y in rb]
            return min(corners), max(corners)
        if op in ("/", "//") and rb[0] == rb[1] and rb[0] > 0 and ra[0] >= 0:
            return ra[0] // rb[0], ra[1] // rb[0]
        if op == "%" and rb[0] == rb[1] and rb[0] > 0 and ra[0] >= 0:
            return (ra[0], ra[1]) if ra[1] < rb[0] else (0, rb[0] - 1)
        if op in ("max", "min"):
            f = max if op == "max" else min
            return f(ra[0], rb[0]), f(ra[1], rb[1])
        return None

    def _split(self, v: Val) -> tuple[Val | None, int]:
        """`v` as (register base, constant offset); a constant is (None, value)."""
        if v.is_const:
            return None, int(v.const)
        return self.affine.get(v.text, (v, 0))

    def _offset(self, base: Val, k: int) -> Val:
        """The register `base + k`, remembered as affine so addresses can split it again."""
        if k == 0:
            return base
        r = self._raw("add.s32", base, const_int(k))
        self.affine[r.text] = (base, k)
        return r

    def _raw(self, instr: str, a: Val, b: Val) -> Val:
        """Emit one integer instruction, or reuse the register of an identical earlier one."""
        a = self.em.materialize(a)
        key = (instr, a.text, b.text)
        hit = self._cse_get(key)
        if hit is not None:
            return hit
        r = self.em.reg("s32")
        self.em.emit(f"{instr} {r}, {a}, {b}")
        self.cse[-1][key] = r
        return r

    def _cse_get(self, key):
        for table in reversed(self.cse):
            if key in table:
                return table[key]
        return None

    def _forget(self, reg: Val) -> None:
        """`reg` is about to change: drop every cached result and affine fact that read it."""
        for table in self.cse:
            for key in [k for k, v in table.items() if reg.text in k[1:] or v.text == reg.text]:
                del table[key]
        for key in [k for k, (base, _) in self.affine.items() if k == reg.text or base.text == reg.text]:
            del self.affine[key]

    def block(self):
        """Context manager for a block of code that may not run (a loop body or a branch)."""
        compiler = self

        class _Block:
            def __enter__(self):
                compiler.cse.append({})

            def __exit__(self, *exc):
                compiler.cse.pop()

        return _Block()

    def select(self, p: Val, t: Val, f: Val) -> Val:
        """`p ? t : f` with both sides brought to one type."""
        if t.ty == "f32" or f.ty == "f32":
            t, f, ty = self.to_f32(t), self.to_f32(f), "f32"
        else:
            t, f, ty = self.to_s32(t), self.to_s32(f), "s32"
        if p.is_const:
            return t if p.const else f
        if t.is_const and f.is_const and t.const == f.const:
            return t
        r = self.em.reg(ty)
        self.em.emit(f"selp.{'b32' if ty == 's32' else ty} {r}, {t}, {f}, {p}")
        return r

    # =====================================================================================
    # Buffers and addresses
    # =====================================================================================
    def buffer(self, name: str) -> Buffer:
        if name not in self.buffers:
            raise CompileError(f"unknown buffer {name!r} (not a kernel argument or a declared Smem)")
        return self.buffers[name]

    def flat_index(self, buf: Buffer, index) -> Val:
        """Row-major element offset of buf[i0, i1, ...]: ((i0 * d1) + i1) * d2 + i2 ..."""
        index = tuple(index)
        if not index:
            return const_int(0)
        shape = buf.shape if len(buf.shape) == len(index) else (1,) * len(index)
        flat = None
        for dim, e in enumerate(index):
            i = self.to_s32(self.expr(e))
            flat = i if flat is None else self.iop("+", self.iop("*", flat, const_int(shape[dim])), i)
        return flat

    def swizzle(self, flat: Val, mode: str) -> Val:
        """Emmy's shared-memory XOR swizzle: e ^ (((e >> shift) & mask) << 3)."""
        base, _, shift = (mode or "NONE").partition("@")
        if base not in SWIZZLE_XOR:
            return flat
        sh, mask = SWIZZLE_XOR[base]
        sh = int(shift) if shift else sh
        bits = self.iop("<<", self.iop("&", self.iop(">>", flat, const_int(sh)), const_int(mask)), const_int(3))
        return self.iop("^", flat, bits)

    def address(self, buf: Buffer, flat: Val, extra: int = 0) -> str:
        """The PTX address operand of element `flat + extra` of `buf`, e.g. `[%rd7+8]`."""
        em = self.em
        root, k = self._split(flat)                       # flat = root + k elements
        off = (k + extra) * buf.esize
        if root is None:
            reg = buf.base
        else:
            key = ("addr", buf.name, root.text)
            reg = self._cse_get(key)
            if reg is None:
                if buf.space == "global":
                    wide, reg = em.reg("u64"), em.reg("u64")
                    em.emit(f"mul.wide.s32 {wide}, {root}, {buf.esize}")
                    em.emit(f"add.s64 {reg}, {buf.base}, {wide}")
                else:
                    reg = em.reg("s32")
                    em.emit(f"mad.lo.s32 {reg}, {root}, {buf.esize}, {buf.base}")
                self.cse[-1][key] = reg
        if off < 0:                                       # keep the immediate offset non-negative
            r = em.reg(reg.ty)
            em.emit(f"add.{'s64' if reg.ty == 'u64' else 's32'} {r}, {reg}, {off}")
            reg, off = r, 0
        return f"[{reg}+{off}]" if off else f"[{reg}]"

    # =====================================================================================
    # Statements
    # =====================================================================================
    def stmt(self, s) -> None:
        handler = getattr(self, "visit_" + type(s).__name__, None)
        if handler is None:
            raise CompileError(f"statement {type(s).__name__} is not supported")
        handler(s)

    def body(self, stmts) -> None:
        stmts = list(stmts)
        merged: set[int] = set()
        for i, s in enumerate(stmts):
            if i in merged:
                continue
            if type(s).__name__ == "Load" and self._hoisted_load(s):
                continue
            if type(s).__name__ == "Load" and len(s.names) == 1:
                j = self._paired_load(stmts, i)
                if j is not None:
                    merged.add(j)
                    continue
            self.stmt(s)

    def _paired_load(self, stmts: list, i: int) -> int | None:
        """Merge stmts[i], a scalar load of element e, with a later load of element e + 1.

        Two f32 loads of neighbouring elements become one 8-byte `.v2` load when the address
        is provably 8-byte aligned and nothing in between could change the buffer or the
        second load's index. Returns the index of the absorbed load, or None.
        """
        first = stmts[i]
        buf = self.buffers.get(first.input)
        if buf is None or buf.dtype != "f32" or buf.align < 8:
            return None
        if any(type(t).__name__ == "Write" and t.output == first.input for t in stmts):
            return None                                   # the block writes this buffer
        flat = self.flat_index(buf, first.index)
        root, k = self._split(flat)
        if k % 2 or (root is not None and self.mult_of(root) % 2):
            return None
        defined: set[str] = set(first.names)
        stoppers = {"Sync", "CpAsyncCopy", "CpAsyncWait", "Loop", "StridedLoop", "Cond", "TreeHalve", "Smem"}
        for j in range(i + 1, min(len(stmts), i + 32)):
            t = stmts[j]
            kind = type(t).__name__
            if kind in stoppers:
                return None
            if (kind == "Load" and len(t.names) == 1 and t.input == first.input
                    and not set().union(*(e.free_vars() for e in t.index)) & defined):
                r2, k2 = self._split(self.flat_index(buf, t.index))
                if k2 == k + 1 and (r2 is None) == (root is None) and (root is None or r2.text == root.text):
                    a, b = self.define(first.names[0], "f32"), self.define(t.names[0], "f32")
                    self.em.emit(f"ld.{buf.space}.v2.f32 {{{a}, {b}}}, {self.address(buf, flat)}")
                    return j
            defined.update(_defines(t))
        return None

    # ---- the launch ---------------------------------------------------------------------
    def visit_Tile(self, t) -> None:
        """Map the global thread id onto the tile's axes (last axis fastest)."""
        em = self.em
        extents = [int(a.extent.as_static()) for a in t.axes]
        n = math.prod(extents) if extents else 1
        if n > 2**31 - 1:
            raise CompileError(f"tile of {n} cells needs 64-bit indexing")
        block = math.prod(self.launch.get("block", [256]))
        grid = math.prod(self.launch.get("grid", [1]))
        ctaid = em.reg("s32")
        em.emit(f"mov.u32 {ctaid}, %ctaid.x")
        gid = em.reg("s32")
        if t.aux_threads:
            # A warp-specialized producer band runs past the cooperative cells and wraps onto them.
            em.emit(f"rem.u32 {gid}, {self.tid}, {t.block_threads}")
            em.emit(f"mad.lo.s32 {gid}, {ctaid}, {t.block_threads}, {gid}")
            cells = grid * t.block_threads
        else:
            ntid = em.reg("s32")
            em.emit(f"mov.u32 {ntid}, %ntid.x")
            em.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {self.tid}")
            cells = grid * block
        # The launch can carry more threads than the tile has cells; the extra ones exit.
        if t.aux_threads or cells != n:
            self.guarded = True
            p = em.reg("pred")
            em.emit(f"setp.ge.s32 {p}, {gid}, {n}")
            em.emit(f"@{p} bra {self.exit_label}")
        self.ranges[gid.text] = (0, n - 1)                # past the guard, or launched exactly
        self._decode_axes(t, extents, gid)
        for a, e in zip(t.axes, extents):
            v = self.env[a.name]
            if not v.is_const:
                self.ranges[v.text] = (0, e - 1)
        self.body(t.body)

    def _decode_axes(self, t, extents: list[int], gid: Val) -> None:
        """axis_i = (gid / stride_i) % extent_i, with stride_i the product of the inner extents.

        With `raster_group` set (2-D matmul grids), the (m, n) block pair is remapped so that
        groups of consecutive CTAs share an operand slab, as Emmy's renderer does.
        """
        strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
        names = [a.name for a in t.axes]
        raster = None
        if t.raster_group and t.raster_axes and all(x in names for x in t.raster_axes):
            mi, ni = names.index(t.raster_axes[0]), names.index(t.raster_axes[1])
            if ni == mi + 1:
                gi, oi = (mi, ni) if t.raster_orient == "m" else (ni, mi)
                g = min(t.raster_group, extents[gi])
                if extents[gi] > 1 and g > 1:
                    raster = (mi, ni, gi, oi, g)
        if raster is not None:
            mi, ni, gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            sub = self.iop("/", gid, const_int(strides[ni]))
            if mi != 0:
                sub = self.iop("%", sub, const_int(ea * eb))
            within = self.iop("%", sub, const_int(g * eb))
            if ea % g == 0:
                a_val = self.iop("+", self.iop("*", self.iop("/", sub, const_int(g * eb)), const_int(g)),
                                 self.iop("%", sub, const_int(g)))
                b_val = self.iop("/", within, const_int(g))
            else:
                grp = self.iop("/", sub, const_int(g * eb))
                size = self.iop("min", const_int(g), self.iop("-", const_int(ea), self.iop("*", grp, const_int(g))))
                a_val = self.iop("+", self.iop("*", grp, const_int(g)), self.iop("%", within, size))
                b_val = self.iop("/", within, size)
            self.env[names[gi]], self.env[names[oi]] = a_val, b_val
        for i, name in enumerate(names):
            if raster is not None and i in raster[:2]:
                continue
            v = self.iop("/", gid, const_int(strides[i]))
            if i > 0:                                     # the outermost axis needs no modulo
                v = self.iop("%", v, const_int(extents[i]))
            self.env[name] = v

    # ---- memory -------------------------------------------------------------------------
    def visit_Smem(self, s) -> None:
        self.buffers[s.name] = self.smem_for[id(s)]       # declared up front; now in scope

    def visit_Load(self, s) -> None:
        """`a[, b, c, d] = load buf[index]`: one value, or 2/4 consecutive ones in one vector load."""
        buf = self.buffer(s.input)
        flat = self.flat_index(buf, s.index)
        n = len(s.names)
        reg_ty = "s32" if buf.dtype == "i32" else "f32"
        raw_ty = ELEM[buf.dtype][1]
        if buf.dtype == "f16":                            # load the 16 bits, then widen to f32
            raws = [self.em.reg("b16") for _ in range(n)]
        else:
            raws = [self.define(name, reg_ty) for name in s.names]
        if n in (2, 4):
            self.em.emit(f"ld.{buf.space}.v{n}.{raw_ty} {{{', '.join(map(str, raws))}}}, {self.address(buf, flat)}")
        else:
            for k, r in enumerate(raws):
                self.em.emit(f"ld.{buf.space}.{raw_ty} {r}, {self.address(buf, flat, k)}")
        if buf.dtype == "f16":
            for name, r in zip(s.names, raws):
                self.em.emit(f"cvt.f32.f16 {self.define(name, 'f32')}, {r}")

    def visit_Write(self, s) -> None:
        """`buf[index] = (v0[, v1, ...])`: a store, a vector store, or an atomic add."""
        em = self.em
        buf = self.buffer(s.output)
        flat = self.flat_index(buf, s.index)
        if buf.space == "shared":
            flat = self.swizzle(flat, getattr(s, "swizzle", "NONE"))
        vals = []
        for name in s.values:
            v = self.lookup(name)
            v = self.to_s32(v) if buf.dtype == "i32" else self.to_f32(v)
            if buf.dtype == "f16":
                h = em.reg("b16")
                em.emit(f"cvt.rn.f16.f32 {h}, {em.materialize(v)}")
                v = h
            vals.append(em.materialize(v))
        raw_ty = ELEM[buf.dtype][1]
        if s.atomic:                                      # split-K partial sums: each lane adds its own
            if buf.dtype != "f32":
                raise CompileError("atomic writes are only supported for f32 buffers")
            for k, v in enumerate(vals):
                em.emit(f"red.{buf.space}.add.f32 {self.address(buf, flat, k)}, {v}")
        elif len(vals) in (2, 4):
            em.emit(f"st.{buf.space}.v{len(vals)}.{raw_ty} {self.address(buf, flat)}, {{{', '.join(map(str, vals))}}}")
        else:
            for k, v in enumerate(vals):
                em.emit(f"st.{buf.space}.{raw_ty} {self.address(buf, flat, k)}, {v}")

    def visit_CpAsyncCopy(self, s) -> None:
        """Copy 4/8/16 bytes from global straight into shared memory (sm_80+)."""
        dst, src = self.buffer(s.smem), self.buffer(s.src)
        flat = self.flat_index(dst, s.smem_index)
        if s.lane_index is not None:                      # stripe base + this lane's (row, col)
            cols = dst.shape[-1]
            lane = self.iop("+", self.iop("*", self.to_s32(self.expr(s.lane_index[0])), const_int(cols)),
                            self.to_s32(self.expr(s.lane_index[1])))
            flat = self.iop("+", flat, lane)
        flat = self.swizzle(flat, s.swizzle)
        smem_addr = self.address(dst, flat)
        gmem_addr = self.address(src, self.flat_index(src, s.src_index))
        cache = "cg" if s.nbytes == 16 else "ca"          # .cg (bypass L1) only exists for 16 bytes
        self.em.emit(f"cp.async.{cache}.shared.global {smem_addr}, {gmem_addr}, {s.nbytes}")

    def visit_CpAsyncCommit(self, s) -> None:
        self.em.emit("cp.async.commit_group")

    def visit_CpAsyncWait(self, s) -> None:
        self.em.emit(f"cp.async.wait_group {int(s.group)}")

    def visit_Sync(self, s) -> None:
        if getattr(s, "warp", False):
            self.em.emit("bar.warp.sync 0xffffffff")
        elif s.barrier_id:
            self.em.emit(f"bar.sync {s.barrier_id}, {s.count}")
        else:
            self.em.emit("bar.sync 0")

    # ---- scalar compute -----------------------------------------------------------------
    def visit_Assign(self, s) -> None:
        dtype = _dtype_name(s.dtype)
        ty = "s32" if dtype in INT_DTYPES else "f32"
        args = [self.lookup(a) for a in s.args]
        args = [self.to_s32(a) if ty == "s32" else self.to_f32(a) for a in args]
        dst = self.define(s.name, ty)
        # Ops that take several instructions must not overwrite an input midway: compute into
        # a temporary when the destination is also an argument (e.g. `acc0 = op(acc0, x)`).
        out = self.em.reg(ty) if any(a.text == dst.text for a in args) else dst
        (lower_int_op if ty == "s32" else lower_op)(self.em, s.op.name, args, out)
        self.copy_into(dst, out)

    def visit_Let(self, s) -> None:
        v = self.expr(s.value)
        dtype = _dtype_name(s.dtype)
        if dtype is not None:
            v = self.to_s32(v) if dtype in INT_DTYPES else self.to_f32(v)
        if v.is_const:                                    # constants stay constants (folding)
            self.env[s.name] = v
            return
        self.copy_into(self.define(s.name, v.ty), v)

    def visit_Init(self, s) -> None:
        self.copy_into(self.define(s.name, "f32"), const_f32(s.identity))

    def visit_Accum(self, s) -> None:
        """`name <- op(base, value)`: fold into the accumulator (seeded by the enclosing loop)."""
        acc = self.lookup(s.name)
        base = self.lookup(s.base) if s.base is not None else acc
        value = self.to_f32(self.lookup(s.value))
        out = self.em.reg("f32") if value.text == acc.text else acc
        lower_op(self.em, s.op.name, [base, value], out)
        self.copy_into(acc, out)

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

    # ---- control flow -------------------------------------------------------------------
    def visit_Loop(self, s) -> None:
        """for axis in 0..extent."""
        self._seed_accumulators(s)
        extent = int(s.axis.extent.as_static())
        self._loop(s.axis.name, const_int(0), const_int(extent), const_int(1), s.body, s.unroll)

    def visit_StridedLoop(self, s) -> None:
        """for axis in start..end:step (each cooperating thread strides the axis)."""
        self._seed_accumulators(s)
        start = self.to_s32(self.expr(s.start))
        step = self.to_s32(self.expr(s.step)) if not isinstance(s.step, int) else const_int(s.step)
        if s.end is not None:
            end = self.to_s32(self.expr(s.end))
        else:
            end = const_int(int(s.axis.extent.as_static()))
        self._loop(s.axis.name, start, end, step, s.body, s.unroll)

    def _loop(self, var: str, start: Val, end: Val, step: Val, body, unroll: bool) -> None:
        """Emit `for (var = start; var < end; var += step) body`.

        - A loop with a known trip count is unrolled completely when the IR asks for it (or
          when it runs at most twice): the variable becomes a constant in every copy.
        - Otherwise it is a runtime loop. When the body is plain computation, it is unrolled
          so that each trip of the main loop issues about LOADS_IN_FLIGHT loads at once, and
          a remainder loop finishes the last few iterations.
        """
        em = self.em
        saved = self.env.get(var)
        trips = None
        if start.is_const and end.is_const and step.is_const and step.const > 0:
            trips = max(0, -(-(end.const - start.const) // step.const))
        if trips is not None and self._fully_unrolled(trips, body, unroll):
            for k in range(trips):                        # unrolled: the variable is a constant
                self.env[var] = const_int(start.const + k * step.const)
                self.body(body)
        elif self._short_loop(var, start, end, step, body):
            pass
        else:
            hoist = trips is not None and trips >= 1 and not self.lean
            if hoist:
                self._hoist_invariant_loads(var, body)
            i = em.reg("s32")
            self.copy_into(i, start)
            factor = self._unroll_factor(body) if step.is_const else 1
            if trips is not None and trips < 2 * factor:
                factor = 1
            if factor > 1:
                self._rotated_loop(var, i, end, step, body, factor, f"loop_x{factor}")
            if factor == 1 or trips is None or trips % factor:
                self._rotated_loop(var, i, end, step, body, 1, "loop")
            if hoist:
                self.hoisted.pop()
        if saved is None:
            self.env.pop(var, None)
        else:
            self.env[var] = saved

    def _fully_unrolled(self, trips: int, body, unroll: bool) -> bool:
        """Whether a loop with a known trip count is unrolled completely."""
        if trips <= 2:
            return True
        if self.lean:
            return False
        size = trips * self._unrolled_size(body)
        if unroll and trips <= UNROLL_MAX and size <= FLAGGED_UNROLL_STMTS:
            return True
        return self._unroll_factor(body) > 1 and size <= FULL_UNROLL_STMTS

    def _unrolled_size(self, body) -> int:
        """Statements `body` expands to, counting the copies of inner loops that unroll."""
        size = 0
        for t in body:
            size += 1
            if type(t).__name__ == "Loop":
                trips = int(t.axis.extent.as_static())
                inner = self._unrolled_size(t.body)
                size += inner * (trips if self._fully_unrolled(trips, t.body, t.unroll) else 1)
            else:
                size += sum(self._unrolled_size(sub) for sub in _children(t))
        return size

    def _hoist_invariant_loads(self, var: str, body) -> None:
        """Load, in front of a runtime loop, the values its body reads that never change.

        A load is invariant when it reads a global buffer the kernel never writes, at an index
        built only from names that exist before the loop and from the variables of inner loops
        that are unrolled completely. Each such load is issued once here, and `visit_Load`
        then finds its address in `self.hoisted` instead of loading again every iteration.
        Only for loops known to run at least once, so no load runs that would not have.
        """
        inner_defs = {var}.union(*(_defines(t) for t in _walk(body)))
        found: list[tuple] = []

        def scan(stmts, bound: dict) -> None:
            for t in stmts:
                kind = type(t).__name__
                if kind == "Load":
                    buf = self.buffers.get(t.input)
                    free = set().union(*(e.free_vars() for e in t.index)) if t.index else set()
                    if (buf is not None and buf.space == "global" and buf.dtype == "f32"
                            and t.input not in self.written and not (free - set(bound)) & inner_defs
                            and (free - set(bound)) <= set(self.env)):
                        found.append((t, dict(bound)))
                elif kind == "Loop":
                    trips = int(t.axis.extent.as_static())
                    if self._fully_unrolled(trips, t.body, t.unroll):
                        for k in range(trips):
                            scan(t.body, {**bound, t.axis.name: k})

        scan(body, {})
        table: dict = {}
        budget = HOIST_MAX
        for load, bound in found:
            if budget < len(load.names):
                break
            saved = {name: self.env.get(name) for name in bound}
            self.env.update({name: const_int(k) for name, k in bound.items()})
            buf = self.buffers[load.input]
            addr = self.address(buf, self.flat_index(buf, load.index))
            key = (buf.name, addr, len(load.names))
            if key not in table:
                regs = [self.em.reg("f32") for _ in load.names]
                width = f".v{len(regs)}" if len(regs) in (2, 4) else ""
                if width or len(regs) == 1:
                    self.em.emit(f"ld.global{width}.f32 {{{', '.join(map(str, regs))}}}, {addr}"
                                 if width else f"ld.global.f32 {regs[0]}, {addr}")
                    table[key] = regs
                    budget -= len(regs)
            for name, old in saved.items():
                if old is None:
                    self.env.pop(name, None)
                else:
                    self.env[name] = old
        self.hoisted.append(table)

    def _hoisted_load(self, s) -> bool:
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

    def _short_loop(self, var: str, start: Val, end: Val, step: Val, body) -> bool:
        """Unroll a loop whose trip count depends on the thread but is known to be small.

        A cooperative loop `for i in tid..N:step` runs between tmin and tmax times, from the
        range of `tid`. The first tmin copies run unconditionally; each later copy first
        checks `i < end` and jumps past the rest when it fails. Returns False when the loop
        is not of this kind.
        """
        if not (step.is_const and step.const > 0 and end.is_const) or self._unroll_factor(body) == 1:
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

    def _rotated_loop(self, var: str, i: Val, end: Val, step: Val, body, factor: int, name: str) -> None:
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
        top, done = em.label(name), em.label(name + "_end")

        def test(cmp: str, target: str) -> None:
            with self.block():
                last = self.iop("+", i, const_int(reach))
                p = em.reg("pred")
                em.emit(f"setp.{cmp}.s32 {p}, {last}, {end}")
                em.emit(f"@{p} bra {target}")

        test("ge", done)
        lo = self.range_of(i)
        if lo is not None and end.is_const:               # inside the body, start <= i < end
            self.ranges[i.text] = (lo[0], end.const - 1)
        em.place(top)
        if factor == 1:
            em.emit('.pragma "nounroll"')                 # remainder loops stay small
        with self.block():
            for k in range(factor):
                self.env[var] = self.iop("+", i, const_int(k * step.const)) if factor > 1 else i
                self.body(body)
        self._forget(i)
        em.emit(f"add.s32 {i}, {i}, {factor * step.const if factor > 1 else step}")
        test("lt", top)
        em.place(done)

    def _unroll_factor(self, body) -> int:
        """How many copies of a runtime loop's body one trip runs (1 = no unrolling).

        Only plain computation of modest size is unrolled: no barriers, async copies or
        cross-thread combines. The factor is the power of two that brings the loads per trip
        to about LOADS_IN_FLIGHT.
        """
        stmts = list(_walk(body))
        blocked = {"Sync", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait", "WarpShuffle", "TreeHalve", "Smem"}
        if self.lean or len(stmts) > UNROLL_BODY_MAX or any(type(s).__name__ in blocked for s in stmts):
            return 1
        loads = sum(type(s).__name__ == "Load" for s in stmts)
        factor = 2
        while factor * 2 * max(loads, 1) <= LOADS_IN_FLIGHT:
            factor *= 2
        return factor

    def visit_Cond(self, s) -> None:
        em = self.em
        p = self.to_pred(self.expr(s.cond))
        if p.is_const:                                    # decided at compile time
            self.body(s.body if p.const else s.else_body)
            return
        else_label, end_label = em.label("else"), em.label("endif")
        em.emit(f"@!{p} bra {else_label if s.else_body else end_label}")
        with self.block():
            self.body(s.body)
        if s.else_body:
            em.emit(f"bra {end_label}")
            em.place(else_label)
            with self.block():
                self.body(s.else_body)
        em.place(end_label)

    # ---- cross-thread combines ----------------------------------------------------------
    def _shuffle_mask(self) -> Val:
        """The lanes taking part in a warp shuffle.

        With no early-exiting threads and whole warps, that is all 32 (the `.sync` then also
        waits for any lane still finishing a loop). Otherwise it is the lanes still running,
        as Emmy's `__activemask()`.
        """
        if not self.guarded and math.prod(self.launch.get("block", [256])) % 32 == 0:
            return const_int(0xFFFFFFFF)
        m = self.em.reg("s32")
        self.em.emit(f"activemask.b32 {m}")
        return m

    def visit_WarpShuffle(self, s) -> None:
        """Butterfly over `length` lanes: shuffle every state component, then run the combine."""
        mask = self._shuffle_mask()
        offset = int(s.length) // 2
        while offset > 0:
            for st, sb in zip(s.state, s.state_b):
                src = self.em.materialize(self.lookup(st))
                self.em.emit(f"shfl.sync.bfly.b32 {self.define(sb, 'f32')}, {src}, {offset}, 31, {mask}")
            self.body(s.combine_states)
            offset //= 2

    def visit_TreeHalve(self, s) -> None:
        """Halving tree over shared slabs, then every thread reads the result (broadcast)."""
        em = self.em
        bufs = [self.buffer(b) for b in s.bufs]
        t = self.to_s32(self.lookup(s.tid_var))
        if s.inner is not None:                           # segment-indexed slabs
            iv, scale = s.inner
            inner = self.to_s32(self.lookup(iv))
            slot = self.iop("+", self.iop("*", t, const_int(scale)), inner)
            slot_of = lambda h: self.iop("+", self.iop("*", self.iop("+", t, const_int(h)), const_int(scale)), inner)
            root = inner
        else:
            slot, slot_of, root = t, (lambda h: self.iop("+", t, const_int(h))), const_int(0)
        if s.barrier_id:
            barrier = f"bar.sync {s.barrier_id}, {s.barrier_count}"
        else:
            barrier = "bar.sync 0"
        half = int(s.length) // 2
        while half > 0:
            skip = em.label("tree_skip")
            p = em.reg("pred")
            em.emit(f"setp.ge.s32 {p}, {em.materialize(t)}, {half}")
            em.emit(f"@{p} bra {skip}")
            with self.block():                            # only threads t < half run this
                other = slot_of(half)
                for buf, st, sb in zip(bufs, s.state, s.state_b):
                    em.emit(f"ld.shared.f32 {self.define(st, 'f32')}, {self.address(buf, slot)}")
                    em.emit(f"ld.shared.f32 {self.define(sb, 'f32')}, {self.address(buf, other)}")
                self.body(s.combine_states)
                for buf, st in zip(bufs, s.state):
                    em.emit(f"st.shared.f32 {self.address(buf, slot)}, {em.materialize(self.lookup(st))}")
            em.place(skip)
            em.emit(barrier)                              # reached by every thread, worker or not
            half //= 2
        for buf, st in zip(bufs, s.state):
            em.emit(f"ld.shared.f32 {self.define(st, 'f32')}, {self.address(buf, root)}")


def compile_kernel(graph, kernel, launch: dict, target: str = "sm_89") -> str:
    """Compile with all optimizations, falling back to the lean mode if the PTX grows too big."""
    ptx = KernelCompiler(graph, kernel, launch, target).compile()
    if len(ptx) > PTX_SIZE_LIMIT:
        ptx = KernelCompiler(graph, kernel, launch, target, lean=True).compile()
    return ptx
