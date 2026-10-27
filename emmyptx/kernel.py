"""Kernel IR -> PTX: the compiler proper.

`KernelCompiler` walks one Emmy `KernelOp` and emits one PTX entry point:

- the prologue: parameters, buffer base addresses, shared arrays, the thread's lane / warp;
- names: every IR name (an SSA value, an accumulator, an axis variable) maps to a `Val` in
  `self.env`, either a register or a compile-time constant;
- expressions: `expr()` compiles Emmy's index / predicate trees, folding constants;
- buffers: `address()` turns "element i of buffer B" into a PTX address operand;
- statements: one `visit_<Kind>` method per Kernel IR statement. The loop statements live
  in `loops.LoopLowering` and the integer arithmetic in `intmath.IntMath`.

Registers follow SSA, with one exception. Every definition gets a fresh register, except the
*carried* names (accumulators, `Init` state, the states of `WarpShuffle` / `TreeHalve`),
whose value flows across loop iterations and combine steps: they keep one register that is
updated in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .intmath import COMMUTATIVE, IntMath
from .loops import LoopLowering, defines, walk
from .ops import lower_int_op, lower_op
from .ptx import Emitter, Val, const_f32, const_int, const_pred

# Element types of buffers: bytes per element, the PTX type used to load / store them, and the
# register class of the raw element when it is converted on load and store (None: no conversion).
# Values are computed as f32 (f16, bf16, and bool as Emmy's 1.0 / 0.0 masks) or s32 (i64).
ELEM = {"f32": (4, "f32", None), "i32": (4, "s32", None), "f16": (2, "b16", "b16"), "bf16": (2, "b16", "b16"),
        "bool": (1, "u8", "b16"), "i64": (8, "s64", "u64")}
INT_BUFFERS = {"i32", "i64"}
_WIDEN = {"f16": "cvt.f32.f16", "bf16": "cvt.f32.bf16", "bool": "cvt.rn.f32.u16"}
_NARROW = {"f16": "cvt.rn.f16.f32", "bf16": "cvt.rn.bf16.f32"}
SMEM_DTYPE = {"float": "f32", "__half": "f16", "half": "f16", "int": "i32"}
# Integer dtypes. `bool` values (comparison results) are 1.0 / 0.0 floats, Emmy's mask convention.
INT_DTYPES = {"int", "i32", "i64", "int32", "int64", "u32", "u8", "i8"}
# The runner accepts PTX up to 2,000,000 characters; past this the kernel is recompiled at a
# lower optimization level (see `compile_kernel`).
PTX_SIZE_LIMIT = 1_800_000
# Shared-memory XOR swizzles: mode -> (element shift, row mask), as in Emmy's slab layouts.
SWIZZLE_XOR = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}

_FLOAT_INSTR = {"+": "add.f32", "-": "sub.f32", "*": "mul.f32", "/": "div.rn.f32",
                "max": "max.f32", "min": "min.f32"}
_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_CMP_FOLD = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
             ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
_CMP_SWAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_FUNC_ALIASES = {"fmax": "maximum", "fmin": "minimum", "fabs": "abs"}   # FuncCallExpr names
_SPECIAL_REGS = {f"{v}.{d}": f"%{r}.{d}" for v, r in (("threadIdx", "tid"), ("blockIdx", "ctaid"),
                                                      ("blockDim", "ntid"), ("gridDim", "nctaid"))
                 for d in "xyz"} | {"warpSize": "WARP_SZ"}
EXIT_LABEL = "$L_exit"


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


def _dtype_name(dt) -> str | None:
    return None if dt is None else getattr(dt, "name", dt)


class KernelCompiler(IntMath, LoopLowering):
    error = CompileError

    def __init__(self, graph, kernel, launch: dict, target: str = "sm_89", level: int = 2) -> None:
        self.kernel = kernel
        self.launch = launch
        self.target = target
        # Optimization level: 2 = everything; 1 = no complete unrolling beyond what the IR asks
        # for; 0 = the plain lowering: no optional unrolling, load hoisting or load pairing.
        self.level = level
        self.em = Emitter()
        self.init_intmath()
        self.env: dict[str, Val] = {}
        self.buffers: dict[str, Buffer] = {}
        self.smem_decls: list[str] = []
        self.smem_for: dict[int, Buffer] = {}     # id(Smem statement) -> its buffer
        self.guarded = False                      # some launched threads exit early
        self.hoisted: list[dict] = []             # loads issued ahead of open runtime loops
        self.products: dict[str, tuple[Val, Val]] = {}   # lazy f32 product register -> (a, b)
        self.carried: set[str] = set()            # names whose register is updated in place
        self.carried_regs: set[str] = set()       # ... and those registers
        for s in walk(kernel.body):
            kind = type(s).__name__
            if kind in ("Accum", "Init"):
                self.carried.add(s.name)
            elif kind in ("WarpShuffle", "TreeHalve"):
                self.carried.update(s.state)
        # Buffers the kernel writes: their contents are never assumed to stay the same.
        self.written = {getattr(t, "output", None) or getattr(t, "smem", None)
                        for t in walk(kernel.body) if type(t).__name__ in ("Write", "CpAsyncCopy")}
        # Every tensor the graph knows: name -> (shape, dtype). Scalars count as one element.
        self.tensors = {}
        for node in graph.nodes.values():
            for t in node.outputs:
                self.tensors[t.name] = (tuple(d.as_static() for d in t.shape) or (1,), t.dtype.name)

    # =====================================================================================
    # The kernel
    # =====================================================================================
    def compile(self) -> str:
        params = self._prologue()
        self.body(self.kernel.body)
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
            *self.em.render(),
            f"{EXIT_LABEL}:",
            "    ret;",
            "}",
            "",
        ])

    def _prologue(self) -> list[str]:
        """Parameters in meta.json's `arg_order`, shared arrays, and the thread's lane / warp."""
        em = self.em
        args = {a["name"]: a for a in self.launch.get("args", [])}
        params = []
        for name in self.launch["arg_order"]:
            arg = args.get(name, {})
            if arg.get("scalar"):                         # passed by value, not as a pointer
                ty = "f32" if arg.get("dtype", "f32") == "f32" else "s32"
                params.append(f"    .param .{ty} {name}")
                r = em.reg(ty)
                em.emit(f"ld.param.{ty} {r}, [{name}]")
                self.env[name] = r
                continue
            params.append(f"    .param .u64 {name}")
            r = em.reg("u64")
            em.emit(f"ld.param.u64 {r}, [{name}]")
            em.emit(f"cvta.to.global.u64 {r}, {r}")
            shape, dtype = self.tensors.get(name, (tuple(arg.get("shape") or (1,)), arg.get("dtype", "f32")))
            if dtype == "f64":
                raise CompileError(f"buffer {name!r} is f64: emmyptx computes in f32, which would silently "
                                   "lose the f64 precision")
            if dtype not in ELEM:
                raise CompileError(f"buffer {name!r} has unsupported dtype {dtype!r}")
            if dtype == "bf16" and int("".join(filter(str.isdigit, self.target)) or 0) < 80:
                raise CompileError(f"buffer {name!r} is bf16, whose conversions need sm_80 or newer")
            self.buffers[name] = Buffer(name, "global", r, tuple(shape), dtype, align=256)  # cudaMalloc
        self._declare_shared()
        tid, lane, warp = em.reg("s32"), em.reg("s32"), em.reg("s32")
        em.emit(f"mov.u32 {tid}, %tid.x")
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
        """Declare every `Smem` array up front, at module scope.

        Up to 48 KiB the arrays are static `.shared` variables. Above that, `meta.json`
        launches with a dynamic allocation (`smem_bytes`), and the arrays are carved out of one
        `.extern .shared` pool instead, as Emmy's own renderer does.
        """
        em = self.em
        dynamic = self.launch.get("smem_bytes", 0) > 0
        if dynamic:
            self.smem_decls.append(".extern .shared .align 16 .b8 _smem_pool[];")
            pool = em.reg("s32")
            em.emit(f"mov.u32 {pool}, _smem_pool")
        uses: dict[str, int] = {}
        offset = 0
        for s in (t for t in walk(self.kernel.body) if type(t).__name__ == "Smem"):
            dtype = SMEM_DTYPE.get(s.dtype)
            if dtype is None:
                raise CompileError(f"shared array {s.name!r} has unsupported dtype {s.dtype!r}")
            esize = ELEM[dtype][0]
            nbytes = math.prod(int(e) for e in s.extents) * esize
            align = max(int(s.align or 0), esize, 4)
            symbol = s.name if s.name not in uses else f"{s.name}_{uses[s.name]}"
            uses[s.name] = uses.get(s.name, 0) + 1
            if dynamic:                                   # carve the array out of the pool
                offset = (offset + 15) // 16 * 16
                base = pool
                if offset:
                    base = em.reg("s32")
                    em.emit(f"add.s32 {base}, {pool}, {offset}")
                offset += nbytes
                align = 16
            else:                                         # a static array of its own
                self.smem_decls.append(f".shared .align {align} .b8 {symbol}[{nbytes}];")
                base = em.reg("s32")
                em.emit(f"mov.u32 {base}, {symbol}")
            self.smem_for[id(s)] = Buffer(s.name, "shared", base, tuple(int(e) for e in s.extents), dtype, align)
        if dynamic and offset > self.launch["smem_bytes"]:
            raise CompileError(f"shared arrays need {offset} bytes, the launch provides {self.launch['smem_bytes']}")

    # =====================================================================================
    # Names and conversions
    # =====================================================================================
    def lookup(self, name) -> Val:
        if isinstance(name, (int, float)):                # a numeric operand of an Assign
            return const_f32(name)
        if name not in self.env:
            raise CompileError(f"use of undefined name {name!r}")
        return self.env[name]

    def define(self, name: str, ty: str) -> Val:
        """The register a new value of `name` is written to: a fresh one, except for carried names."""
        old = self.env.get(name)
        if name in self.carried and old is not None and not old.is_const and old.ty == ty:
            return old
        r = self.em.reg(ty)
        self.env[name] = r
        if name in self.carried:
            self.carried_regs.add(r.text)
        return r

    def copy_into(self, dst: Val, v: Val) -> None:
        if dst.ty == "s32":
            self.forget(dst)
        if dst.text != v.text:
            self.em.emit(f"mov.{'b32' if dst.ty == 's32' else dst.ty} {dst}, {v}")

    def to_f32(self, v: Val) -> Val:
        if v.ty == "f32":
            return v
        if v.is_const:
            return const_f32(float(v.const))
        r = self.em.reg("f32")
        if v.ty == "pred":
            self.em.emit(f"selp.f32 {r}, {const_f32(1.0)}, {const_f32(0.0)}, {v}")
        else:
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
            self.em.emit(f"setp.neu.f32 {p}, {v}, {const_f32(0.0)}")  # C truthiness: NaN is true
        else:
            self.em.emit(f"setp.ne.s32 {p}, {v}, 0")
        return p

    # =====================================================================================
    # Expressions
    # =====================================================================================
    def expr(self, e) -> Val:
        kind = type(e).__name__
        if kind == "Literal":
            if isinstance(e.value, bool):
                return const_pred(e.value)
            dtype = _dtype_name(e.dtype)
            if dtype in INT_DTYPES or (dtype is None and isinstance(e.value, int)):
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
            if e.name not in _SPECIAL_REGS:
                raise CompileError(f"unsupported builtin {e.name!r}")
            r = self.em.reg("s32")
            self.em.emit(f"mov.u32 {r}, {_SPECIAL_REGS[e.name]}")
            return r
        if kind == "FuncCallExpr":
            dst = self.em.reg("f32")
            lower_op(self.em, _FUNC_ALIASES.get(e.name, e.name), [self.to_f32(self.expr(a)) for a in e.args], dst)
            return dst
        raise CompileError(f"unsupported expression {kind}: {e!r}")

    def binary(self, op: str, a: Val, b: Val) -> Val:
        em = self.em
        if op in ("&&", "||"):
            a, b = self.to_pred(a), self.to_pred(b)
            if a.is_const and b.is_const:
                return const_pred(a.const and b.const if op == "&&" else a.const or b.const)
            if a.is_const:
                a, b = b, a
            if b.is_const:                                # x && true = x, x || false = x
                return a if b.const == (op == "&&") else const_pred(op == "||")
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
        if a.ty == "f32" or b.ty == "f32":
            if op not in _FLOAT_INSTR:
                raise CompileError(f"float operator {op!r} is not supported")
            a, b = self.to_f32(a), self.to_f32(b)
            if a.is_const and op in COMMUTATIVE:
                a, b = b, a
            r = em.reg("f32")
            em.emit(f"{_FLOAT_INSTR[op]} {r}, {em.materialize(a)}, {b}")
            return r
        if a.ty == "pred" and b.ty == "pred" and op == "^":
            p = em.reg("pred")
            em.emit(f"xor.pred {p}, {em.materialize(a)}, {em.materialize(b)}")
            return p
        return self.iop(op, self.to_s32(a), self.to_s32(b))

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
        if ty == "s32":                                   # either value: a multiple of what both are
            m = math.gcd(self.mult_of(t), self.mult_of(f))
            if m > 1:
                self.mults[r.text] = m
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
        """The PTX address operand of element `flat + extra` of `buf`, e.g. `[%rd7+8]`.

        The constant part of `flat` becomes the immediate offset, so neighbouring elements
        share one address register.
        """
        em = self.em
        root, k = self.split(flat)
        off = (k + extra) * buf.esize
        if root is None:
            reg = buf.base
        else:
            key = ("addr", buf.name, root.text)
            reg = self._cse_get(key)
            if reg is None:
                if buf.space == "global":
                    wide_key = ("wide", root.text, buf.esize)     # the byte offset, shared by
                    wide = self._cse_get(wide_key)                # every buffer indexed by root
                    if wide is None:
                        wide = em.reg("u64")
                        em.emit(f"mul.wide.s32 {wide}, {root}, {buf.esize}")
                        self.remember(wide_key, wide)
                    reg = em.reg("u64")
                    em.emit(f"add.s64 {reg}, {buf.base}, {wide}")
                else:
                    reg = em.reg("s32")
                    em.emit(f"mad.lo.s32 {reg}, {root}, {buf.esize}, {buf.base}")
                self.remember(key, reg)
        if off < 0:                                       # keep the immediate offset non-negative
            r = em.reg(reg.ty)
            em.emit(f"add.{'s64' if reg.ty == 'u64' else 's32'} {r}, {reg}, {off}")
            reg, off = r, 0
        return f"[{reg}+{off}]" if off else f"[{reg}]"

    def known_alignment(self, buf: Buffer, flat: Val) -> int:
        """The largest power of two the address of element `flat` of `buf` is known to be a
        multiple of: the base's alignment and a known divisor of the byte offset."""
        m = math.gcd(buf.align, self.mult_of(flat) * buf.esize)
        return m & -m

    def require_aligned(self, s, buf: Buffer, flat: Val, nbytes: int) -> None:
        """Stop unless element `flat` of `buf` is provably `nbytes`-aligned. A PTX memory access
        must be aligned to its whole size (a `.v4.f32` load or a 16-byte `cp.async` to 16
        bytes); a misaligned one faults or silently reads other bytes."""
        known = self.known_alignment(buf, flat)
        if known < nbytes:
            what = s.pretty()[0].strip() if hasattr(s, "pretty") else type(s).__name__
            raise CompileError(f"{what}: this {nbytes}-byte access to {buf.name!r} needs a {nbytes}-byte aligned "
                               f"address, and the index only proves {known}-byte alignment")

    # =====================================================================================
    # Statements
    # =====================================================================================
    def stmt(self, s) -> None:
        handler = getattr(self, "visit_" + type(s).__name__, None)
        if handler is None:
            raise CompileError(f"statement {type(s).__name__} is not supported")
        handler(s)

    def body(self, stmts) -> None:
        """Compile a statement list, in order."""
        stmts = list(stmts)
        absorbed: set[int] = set()
        for i in range(len(stmts)):
            if i not in absorbed:
                absorbed |= self.statement_in(stmts, i)

    def statement_in(self, stmts: list, i: int) -> set[int]:
        """Compile stmts[i]; return the indices of later statements it took care of too.

        Loads get two chances before the plain lowering: to reuse a value hoisted out of an
        enclosing loop, and to merge with a later load of the next element.
        """
        s = stmts[i]
        if type(s).__name__ == "Load":
            if self.hoisted_load(s):
                return set()
            if len(s.names) == 1 and self.level > 0:
                j = self._paired_load(stmts, i)
                if j is not None:
                    return {j}
        self.stmt(s)
        return set()

    def _paired_load(self, stmts: list, i: int) -> int | None:
        """Merge stmts[i], a scalar load of element e, with a later load of element e + 1.

        Two f32 loads of neighbouring elements become one 8-byte `.v2` load when the address
        is provably 8-byte aligned and nothing in between could change the buffer or the
        second load's index. Returns the index of the absorbed load, or None.
        """
        first = stmts[i]
        buf = self.buffers.get(first.input)
        if buf is None or buf.dtype != "f32":
            return None
        flat = self.flat_index(buf, first.index)
        if self.known_alignment(buf, flat) < 8:
            return None
        root, k = self.split(flat)
        defined: set[str] = set(first.names)
        stoppers = {"Sync", "CpAsyncCopy", "CpAsyncWait", "Loop", "StridedLoop", "Cond", "TreeHalve", "Smem"}
        for j in range(i + 1, min(len(stmts), i + 32)):
            t = stmts[j]
            kind = type(t).__name__
            if kind in stoppers or (kind == "Write" and t.output == first.input):
                return None                               # the buffer may change in between
            if (kind == "Load" and len(t.names) == 1 and t.input == first.input
                    and not set().union(*(e.free_vars() for e in t.index)) & defined):
                r2, k2 = self.split(self.flat_index(buf, t.index))
                if k2 == k + 1 and (r2 is None) == (root is None) and (root is None or r2.text == root.text):
                    a, b = self.define(first.names[0], "f32"), self.define(t.names[0], "f32")
                    self.em.emit(f"ld.{buf.space}.v2.f32 {{{a}, {b}}}, {self.address(buf, flat)}")
                    return j
            defined.update(defines(t))
        return None

    # ---- the launch ---------------------------------------------------------------------
    def visit_Tile(self, t) -> None:
        """Map the global thread id onto the tile's axes (last axis fastest)."""
        em = self.em
        extents = [int(a.extent.as_static()) for a in t.axes]
        n = math.prod(extents)
        if n > 2**31 - 1:
            raise CompileError(f"a tile of {n} cells needs 64-bit indexing")
        block = math.prod(self.launch.get("block", [256]))
        grid = math.prod(self.launch.get("grid", [1]))
        ctaid, gid = em.reg("s32"), em.reg("s32")
        em.emit(f"mov.u32 {ctaid}, %ctaid.x")
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
        if t.aux_threads or cells != n:
            # The launch carries more threads than the tile has cells: the extra ones exit.
            # (This guard is not in the IR; it follows from N and the launch in meta.json.)
            self.guarded = True
            p = em.reg("pred")
            em.emit(f"setp.ge.s32 {p}, {gid}, {n}")
            em.emit(f"@{p} bra {EXIT_LABEL}")
        self.ranges[gid.text] = (0, n - 1)
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
        # A buffer the kernel never writes holds the same values at the same address, so a load
        # already done in this block (or an enclosing one) is reused instead of repeated.
        reusable = buf.name not in self.written
        if reusable:
            key = ("load", self.address(buf, flat), len(s.names))
            regs = self._cse_get(key)
            if regs is not None:
                self.env.update(zip(s.names, regs))
                return
        regs = self._emit_load(s, buf, flat)
        if reusable:
            self.remember(key, regs)

    @staticmethod
    def _vector(buf: Buffer, n: int) -> bool:
        """Whether n neighbouring elements move as one `.v2` / `.v4` access (of at most 16 bytes)."""
        return n in (2, 4) and n * buf.esize <= 16

    def _emit_load(self, s, buf: Buffer, flat: Val) -> list[Val]:
        names = s.names
        n = len(names)
        _, raw_ty, raw_reg = ELEM[buf.dtype]
        value_ty = "s32" if buf.dtype in INT_BUFFERS else "f32"
        vector = self._vector(buf, n)
        if vector:
            self.require_aligned(s, buf, flat, n * buf.esize)
        if raw_reg is None:                               # straight into the names' registers
            raws = [self.define(name, value_ty) for name in names]
        else:                                             # load the raw elements, then widen them
            raws = [self.em.reg(raw_reg) for _ in range(n)]
        if vector:
            self.em.emit(f"ld.{buf.space}.v{n}.{raw_ty} {{{', '.join(map(str, raws))}}}, {self.address(buf, flat)}")
        else:
            for k, r in enumerate(raws):
                self.em.emit(f"ld.{buf.space}.{raw_ty} {r}, {self.address(buf, flat, k)}")
        if raw_reg is None:
            return raws
        out = []
        for name, r in zip(names, raws):
            out.append(self.define(name, value_ty))
            self._widen(buf.dtype, r, out[-1])
        return out

    def _widen(self, dtype: str, raw: Val, dst: Val) -> None:
        """dst = a loaded raw element of a `dtype` buffer, as an f32 or s32 value."""
        em = self.em
        if dtype != "i64":
            em.emit(f"{_WIDEN[dtype]} {dst}, {raw}")
            return
        # Integers are computed in 32 bits. A loaded value outside that range stops the kernel
        # (trap) instead of wrapping around silently; arithmetic that leaves the range still wraps.
        em.emit(f"cvt.s32.s64 {dst}, {raw}")
        back, p = em.reg("u64"), em.reg("pred")
        em.emit(f"cvt.s64.s32 {back}, {dst}")
        em.emit(f"setp.ne.s64 {p}, {back}, {raw}")
        em.emit(f"@{p} trap")

    def _narrow(self, dtype: str, v: Val) -> Val:
        """The register holding value `v` as stored in a `dtype` buffer."""
        em = self.em
        if dtype in INT_BUFFERS:
            v = em.materialize(self.to_s32(v))
            if dtype == "i32":
                return v
            raw = em.reg("u64")
            em.emit(f"cvt.s64.s32 {raw}, {v}")
            return raw
        if dtype == "bool":                               # nonzero (or NaN) is True, as in torch
            p, raw = self.to_pred(v), em.reg("b16")
            em.emit(f"mov.b16 {raw}, {int(p.const)}" if p.is_const else f"selp.b16 {raw}, 1, 0, {p}")
            return raw
        v = em.materialize(self.to_f32(v))
        if dtype == "f32":
            return v
        raw = em.reg("b16")
        em.emit(f"{_NARROW[dtype]} {raw}, {v}")
        return raw

    def visit_Write(self, s) -> None:
        """`buf[index] = (v0[, v1, ...])`: a store, a vector store, or an atomic add."""
        em = self.em
        buf = self.buffer(s.output)
        flat = self.flat_index(buf, s.index)
        if buf.space == "shared":
            flat = self.swizzle(flat, getattr(s, "swizzle", "NONE"))
        vals = [self._narrow(buf.dtype, self.lookup(name)) for name in s.values]
        raw_ty = ELEM[buf.dtype][1]
        if s.atomic:                                      # split-K partial sums: each lane adds its own
            if buf.dtype != "f32":
                raise CompileError("atomic writes are only supported for f32 buffers")
            for k, v in enumerate(vals):
                em.emit(f"red.{buf.space}.add.f32 {self.address(buf, flat, k)}, {v}")
        elif self._vector(buf, len(vals)):
            self.require_aligned(s, buf, flat, len(vals) * buf.esize)
            em.emit(f"st.{buf.space}.v{len(vals)}.{raw_ty} {self.address(buf, flat)}, {{{', '.join(map(str, vals))}}}")
        else:
            for k, v in enumerate(vals):
                em.emit(f"st.{buf.space}.{raw_ty} {self.address(buf, flat, k)}, {v}")

    def visit_CpAsyncCopy(self, s) -> None:
        """Copy 4/8/16 bytes from global straight into shared memory (sm_80+)."""
        dst, src = self.buffer(s.smem), self.buffer(s.src)
        flat = self.flat_index(dst, s.smem_index)
        if s.lane_index is not None:                      # stripe base + this lane's (row, col)
            row, col = (self.to_s32(self.expr(e)) for e in s.lane_index)
            flat = self.iop("+", flat, self.iop("+", self.iop("*", row, const_int(dst.shape[-1])), col))
        flat = self.swizzle(flat, s.swizzle)
        self.require_aligned(s, dst, flat, s.nbytes)
        smem_addr = self.address(dst, flat)
        src_flat = self.flat_index(src, s.src_index)
        self.require_aligned(s, src, src_flat, s.nbytes)
        gmem_addr = self.address(src, src_flat)
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
        """`name = op(args)`, through the elementwise op table in `ops`."""
        ty = "s32" if _dtype_name(s.dtype) in INT_DTYPES else "f32"
        args = [self.to_s32(a) if ty == "s32" else self.to_f32(a) for a in map(self.lookup, s.args)]
        dst = self.define(s.name, ty)
        if (ty == "f32" and s.op.name == "multiply" and dst.text not in self.carried_regs
                and not any(a.text in self.carried_regs for a in args)):
            # A product is emitted lazily: kept only if read directly, while an accumulation
            # of it (`visit_Accum`) becomes one FMA. Its operands never change (SSA).
            a, b = (args[1], args[0]) if args[0].is_const else (args[0], args[1])
            a = self.em.materialize(a)
            self.em.emit_lazy(dst, f"mul.f32 {dst}, {a}, {b}")
            self.products[dst.text] = (a, b)
            return
        # Ops that take several instructions must not overwrite an input midway: compute into
        # a temporary when the destination is also an argument (e.g. `acc0 = op(acc0, x)`).
        out = self.em.reg(ty) if any(a.text == dst.text for a in args) else dst
        (lower_int_op if ty == "s32" else lower_op)(self.em, s.op.name, args, out)
        self.copy_into(dst, out)

    def visit_Accum(self, s) -> None:
        """`name <- op(base, value)`: fold into the accumulator (seeded by the enclosing loop)."""
        acc = self.lookup(s.name)
        base = self.lookup(s.base) if s.base is not None else acc
        value = self.to_f32(self.lookup(s.value))
        if s.op.name == "add" and value.text in self.products:     # acc = base + a*b: one FMA
            a, b = self.products[value.text]
            self.em.emit(f"fma.rn.f32 {acc}, {a}, {b}, {self.em.materialize(base)}")
            return
        out = self.em.reg("f32") if value.text == acc.text else acc
        lower_op(self.em, s.op.name, [base, value], out)
        self.copy_into(acc, out)

    def visit_Let(self, s) -> None:
        v = self.expr(s.value)
        dtype = _dtype_name(s.dtype)
        if dtype is not None:
            v = self.to_s32(v) if dtype in INT_DTYPES else self.to_f32(v)
        if v.is_const:                                    # constants stay constants (folding)
            self.env[s.name] = v
        else:
            self.copy_into(self.define(s.name, v.ty), v)

    def visit_Init(self, s) -> None:
        self.copy_into(self.define(s.name, "f32"), const_f32(s.identity))

    def visit_Select(self, s) -> None:
        """`name = v0 when p0, v1 when p1, ...`: the first branch whose predicate holds (the last is
        the catch-all), i.e. the chained ternary p0 ? v0 : (p1 ? v1 : ...)."""
        branches = list(s.branches)
        result = self.to_f32(self.lookup(branches[-1].value))
        for b in reversed(branches[:-1]):
            result = self.select(self.to_pred(self.expr(b.select)), self.to_f32(self.lookup(b.value)), result)
        if result.is_const:
            self.env[s.name] = result
        else:                                             # a copy: a branch value may be a
            self.copy_into(self.define(s.name, "f32"), result)  # carried register that changes later

    # ---- control flow -------------------------------------------------------------------
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
        like Emmy's `__activemask()`.
        """
        if not self.guarded and math.prod(self.launch.get("block", [256])) % 32 == 0:
            return const_int(0xFFFFFFFF)
        m = self.em.reg("s32")
        self.em.emit(f"activemask.b32 {m}")
        return m

    def visit_WarpShuffle(self, s) -> None:
        """XOR butterfly over `length` lanes: shuffle every state component, then run the
        combine program (`combine_states`, ordinary Assigns) to merge the neighbour in."""
        mask = self._shuffle_mask()
        offset = int(s.length) // 2
        while offset > 0:
            for st, sb in zip(s.state, s.state_b):
                src = self.em.materialize(self.lookup(st))
                self.em.emit(f"shfl.sync.bfly.b32 {self.define(sb, 'f32')}, {src}, {offset}, 31, {mask}")
            self.body(s.combine_states)
            offset //= 2

    def visit_TreeHalve(self, s) -> None:
        """Halving tree over shared slabs, then every thread reads the result (the broadcast)."""
        em = self.em
        bufs = [self.buffer(b) for b in s.bufs]
        t = self.to_s32(self.lookup(s.tid_var))
        if s.inner is not None:                           # segment-indexed slabs
            iv, scale = s.inner
            inner = self.to_s32(self.lookup(iv))
            slot = self.iop("+", self.iop("*", t, const_int(scale)), inner)
            partner = lambda h: self.iop("+", self.iop("*", self.iop("+", t, const_int(h)), const_int(scale)), inner)
            root = inner
        else:
            slot, partner, root = t, (lambda h: self.iop("+", t, const_int(h))), const_int(0)
        barrier = f"bar.sync {s.barrier_id}, {s.barrier_count}" if s.barrier_id else "bar.sync 0"
        half = int(s.length) // 2
        while half > 0:
            skip = em.label("tree_skip")
            p = em.reg("pred")
            em.emit(f"setp.ge.s32 {p}, {em.materialize(t)}, {half}")
            em.emit(f"@{p} bra {skip}")
            with self.block():                            # only threads t < half run this
                other = partner(half)
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
    """Compile at the highest optimization level that works and fits the runner's size limit.

    Level 0 is the plain lowering, without the optional transformations, so an IR shape that
    trips an optimization still compiles.
    """
    for level in (2, 1, 0):
        try:
            ptx = KernelCompiler(graph, kernel, launch, target, level=level).compile()
        except CompileError:
            raise                                         # unsupported IR: no level helps
        except Exception:
            if level == 0:
                raise
            continue
        if len(ptx) <= PTX_SIZE_LIMIT or level == 0:
            return ptx
    return ptx
