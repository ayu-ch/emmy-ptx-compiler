"""Tensor-core statements: mma.sync on m16n8k16 cells, f16 / bf16 operands, f32 accumulators.

Emmy lowers a matmul cell to per-lane register fragments (`RegFragment`), fills them from
shared memory with `ldmatrix` or straight from global memory (`LdmatrixLoad`), multiplies with
`mma.sync` (`MmaSyncPtx`) and writes the accumulators out (`RegStore`). A fragment holds the
lane's share of a cell, in the PTX ISA's lane map for mma.m16n8k16 (g = lane / 4, t = lane % 4):

    A 16x16 (row-major)   a[i] = A[g + 8 * (i & 1)][2t + 8 * (i >> 1) + {0, 1}]   4 packed pairs
    B 16x8  (col-major)   b[i] = B[2t + 8 * i + {0, 1}][g]                         2 packed pairs
    C 16x8                c[i] = C[g + 8 * (i >> 1)][2t + (i & 1)]                 4 f32

Other layouts Emmy has (m8n8k4, fp8, f16 accumulators, block scales) and fused store
epilogues stop with a CompileError.
"""

from __future__ import annotations

import ast
import math
import re

from .intmath import c_div
from .loops import walk
from .ptx import Val, const_int, const_pred

SHAPE = (16, 8, 16)
NREGS = {"a": 4, "b": 2, "c": 4}
STORE_TYPE = {"f32": "f32", "f16": "b16", "bf16": "b16"}
TO_HALF = {"f16": "cvt.rn.f16.f32", "bf16": "cvt.rn.bf16.f32"}
_REF = re.compile(r"\s*([A-Za-z_]\w*)\s*(?:\[(.+)\])?\s*$")      # `_c0_` or `_c0_[_r0 + 1]`
_REF_FIELDS = {"LdmatrixLoad": ("frag", "pair_frag"), "MmaSyncPtx": ("c_frag", "a_frag", "b_frag"),
               "RegStore": ("frag",)}
_INDEX_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
              ast.Div: c_div, ast.FloorDiv: c_div, ast.Mod: lambda a, b: a - b * c_div(a, b)}


def fragment_refs(stmt) -> list[str]:
    """The fragment references a tensor-core statement names."""
    return [r for r in (getattr(stmt, f, None) for f in _REF_FIELDS.get(type(stmt).__name__, ())) if r]


def _parse_index(text: str):
    """A fragment index (C) as a Python expression tree; None if it is not one."""
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


class TensorCoreLowering:
    """Mixin: the tensor-core statements. Part of `KernelCompiler`."""

    def fragment_index_vars(self, body) -> set[str]:
        """Names used in fragment indices (`_b[_r0]`) in `body`. A loop over one of them is
        unrolled whatever its size: a register cannot be picked at run time."""
        names: set[str] = set()
        for s in walk(body):
            for ref in fragment_refs(s):
                m = _REF.match(ref)
                tree = _parse_index(m.group(2)) if m and m.group(2) else None
                if tree is not None:
                    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        return names

    def visit_RegFragment(self, s) -> None:
        """Registers for `count` fragments: packed 16-bit pairs (a, b) or f32 (c, zeroed)."""
        what = s.pretty()[0].strip()
        dtype = getattr(s.dtype, "name", s.dtype)
        if tuple(s.shape) != SHAPE or s.nregs is not None or s.role not in NREGS:
            raise self.error(f"{what}: only m16n8k16 fragments are supported")
        if dtype not in (("f32",) if s.role == "c" else ("f16", "bf16")):
            raise self.error(f"{what}: supported are f16 / bf16 operands with f32 accumulators")
        if self.sm < 80:
            raise self.error(f"{what}: mma.sync m16n8k16 needs sm_80 or newer")
        ty = "f32" if s.role == "c" else "b32"
        regs = [[self.em.reg(ty) for _ in range(NREGS[s.role])] for _ in range(s.count)]
        if s.role == "c":
            for r in (r for member in regs for r in member):
                self.em.emit(f"mov.f32 {r}, 0f00000000")
        self.frags[s.name] = (s.role, dtype, regs)

    def visit_LdmatrixLoad(self, s) -> None:
        what = s.pretty()[0].strip()
        if s.fragment_layout != "m16n8k16" or s.byte_slab or s.scale_buffer is not None:
            raise self.error(f"{what}: only m16n8k16 fragments of 16-bit elements are supported")
        dtype, regs = self._frag(s.frag, s.role)
        buf = self.buffer(s.src_buffer)
        flat = self.flat_index(buf, s.src_index)
        ldm = int(s.ldm) or buf.shape[-1]
        if s.staged:
            if buf.space != "shared" or buf.esize != 2:
                raise self.error(f"{what}: ldmatrix reads 16-bit elements from shared memory")
            if s.pair_frag is not None:
                regs = regs + self._frag(s.pair_frag, "b")[1]
            self._ldmatrix(s, buf, flat, ldm, regs)
        else:
            self._gmem_fragment(s, buf, flat, ldm, dtype, regs)

    def _ldmatrix(self, s, buf, flat: Val, ldm: int, regs: list[Val]) -> None:
        """One ldmatrix: each lane gives the address of one 8-element row (lanes 0-7 the rows of
        the first 8x8 matrix, 8-15 the second, ...) and receives its share of every matrix,
        which is the fragment map above (.trans for a B slab stored K-major)."""
        lane, k = self.lane, const_int
        rows16, half = self.iop("%", lane, k(16)), self.iop("/", lane, k(16))
        rows8, col8 = self.iop("%", lane, k(8)), self.iop("*", self.iop("%", self.iop("/", lane, k(8)), k(2)), k(8))
        if s.role == "a":                                 # 16 rows x 2 column halves
            row, col, kind = rows16, self.iop("*", half, k(8)), "x4"
        elif s.pair_frag is not None and not s.b_trans:   # two B fragments, N-adjacent columns
            row, col, kind = rows16, self.iop("*", half, k(8)), "x4.trans"
        elif s.pair_frag is not None:                     # two B fragments of an N-major slab
            row, col, kind = self.iop("+", rows8, self.iop("*", half, k(8))), col8, "x4"
        elif s.b_trans:                                   # B stored N x K: rows are B's columns
            row, col, kind = rows8, col8, "x2"
        else:                                             # B stored K x N
            row, col, kind = rows16, k(0), "x2.trans"
        elem = self.swizzle(self.iop("+", flat, self.iop("+", self.iop("*", row, k(ldm)), col)), s.swizzle)
        self.require_aligned(s, buf, elem, 16)
        self.em.emit(f"ldmatrix.sync.aligned.m8n8.{kind}.shared.b16 {{{', '.join(map(str, regs))}}}, "
                     f"{self.address(buf, elem)}")

    def _gmem_fragment(self, s, buf, flat: Val, ldm: int, dtype: str, regs: list[Val]) -> None:
        """A fragment read straight from global memory in the lane map above. `gmem_guard` clamps
        the lane's A row / B column into the tensor (the duplicates it reads are never stored);
        `k_zero` reads the K elements past the bound as zeros, which add nothing to the sum."""
        k = const_int
        g, t = self._lane_group()
        last = k_left = None
        if s.gmem_guard is not None:
            base, bound = (self.to_s32(self.expr(e)) for e in s.gmem_guard)
            last = self.iop("max", self.iop("-", self.iop("-", bound, base), k(1)), k(0))
        if s.k_zero is not None:
            base, bound = (self.to_s32(self.expr(e)) for e in s.k_zero)
            k_left = self.iop("-", bound, base)
        contiguous = s.role == "a" or s.b_trans           # the pair (k, k + 1) is adjacent in memory
        for i, reg in enumerate(regs):
            if s.role == "a":
                mn, kk = self.iop("+", g, k(8 * (i & 1))), self.iop("+", self.iop("*", t, k(2)), k(8 * (i >> 1)))
            else:
                mn, kk = g, self.iop("+", self.iop("*", t, k(2)), k(8 * i))
            if last is not None:
                mn = self.iop("min", mn, last)
            elems = [self.iop("+", flat, self._at(mn, self.iop("+", kk, k(j)), ldm, contiguous)) for j in (0, 1)]
            if (k_left is None and contiguous and buf.dtype == dtype
                    and self.known_alignment(buf, elems[0]) >= 4):  # both halves in one 32-bit load
                self.em.emit(f"ld.{buf.space}.b32 {reg}, {self.address(buf, elems[0])}")
                continue
            halves = []
            for j, elem in enumerate(elems):
                inside = None if k_left is None else self._less(self.iop("+", kk, k(j)), k_left)
                halves.append(self._fragment_half(buf, elem, dtype, inside))
            self.em.emit(f"mov.b32 {reg}, {{{halves[0]}, {halves[1]}}}")   # low half first

    def _at(self, mn: Val, kk: Val, ldm: int, contiguous: bool) -> Val:
        """Element offset of (row or column `mn`, K index `kk`) from the cell's base."""
        k = const_int
        if contiguous:                                    # A[m][k], or B stored as [n][k]
            return self.iop("+", self.iop("*", mn, k(ldm)), kk)
        return self.iop("+", self.iop("*", kk, k(ldm)), mn)

    def _fragment_half(self, buf, elem: Val, dtype: str, inside: Val | None) -> Val:
        """One 16-bit fragment element; zero where `inside` is false."""
        em = self.em
        h = em.reg("b16")
        if inside is not None and inside.is_const and not inside.const:
            em.emit(f"mov.b16 {h}, 0")
            return h
        when = ""
        if inside is not None and not inside.is_const:
            em.emit(f"mov.b16 {h}, 0")
            when = f"@{inside} "
        addr = self.address(buf, elem)
        if buf.dtype == dtype:
            em.emit(f"{when}ld.{buf.space}.b16 {h}, {addr}")
        elif buf.dtype == "f32":                          # an f32 operand, rounded into the fragment
            f = em.reg("f32")
            em.emit(f"{when}ld.{buf.space}.f32 {f}, {addr}")
            em.emit(f"{when}{TO_HALF[dtype]} {h}, {f}")
        else:
            raise self.error(f"{dtype} fragment from a {buf.dtype} buffer {buf.name!r} is not supported")
        return h

    def visit_MmaSyncPtx(self, s) -> None:
        """c = a @ b + c on one 16x8x16 cell, the whole warp together."""
        what = s.pretty()[0].strip()
        if (tuple(s.shape) != SHAPE or s.c_dtype != "f32" or s.ab_dtype not in ("f16", "bf16") or s.b_row_major
                or s.sfa_frag is not None):
            raise self.error(f"{what}: only m16n8k16 with f16 / bf16 operands and f32 accumulators is supported")
        _, c = self._frag(s.c_frag, "c")
        (a_dt, a), (b_dt, b) = self._frag(s.a_frag, "a"), self._frag(s.b_frag, "b")
        if a_dt != s.ab_dtype or b_dt != s.ab_dtype:
            raise self.error(f"{what}: fragments are {a_dt} / {b_dt}, the mma is {s.ab_dtype}")
        ab = s.ab_dtype
        regs = [", ".join(map(str, r)) for r in (c, a, b, c)]
        self.em.emit(f"mma.sync.aligned.m16n8k16.row.col.f32.{ab}.{ab}.f32 "
                     f"{{{regs[0]}}}, {{{regs[1]}}}, {{{regs[2]}}}, {{{regs[3]}}}")

    def visit_RegStore(self, s) -> None:
        """Write an accumulator fragment: the lane's two row pairs (rows g and g + 8, columns
        2t, 2t + 1), converted to the buffer's type, each under the store's row / column guards."""
        what = s.pretty()[0].strip()
        if (s.epilogue is not None or s.extra_frags or s.swizzle != "NONE" or s.fragment_layout != "m16n8k16"
                or s.volta_interleaved):
            raise self.error(f"{what}: fused store epilogues and swizzled or Volta stores are not supported")
        _, c = self._frag(s.frag, "c")
        buf = self.buffer(s.dst_buffer)
        if buf.dtype not in STORE_TYPE or (s.atomic and buf.dtype != "f32"):
            raise self.error(f"{what}: stores go to f32 / f16 / bf16 buffers (atomic ones to f32)")
        k = const_int
        flat = self.flat_index(buf, s.dst_index)
        stride = lambda dim: math.prod(buf.shape[dim + 1:])
        ldm = int(s.ldm) or (stride(s.row_dim) if s.row_dim is not None else buf.shape[-1])
        ldn = stride(s.col_dim) if s.col_dim is not None else 1
        g, t = self._lane_group()
        rows = [g, self.iop("+", g, k(8))]
        cols = [self.iop("*", t, k(2)), self.iop("+", self.iop("*", t, k(2)), k(1))]
        row_ok, col_ok = [None, None], [None, None]
        if s.m_guard is not None:
            base, bound = (self.to_s32(self.expr(e)) for e in s.m_guard)
            row_ok = [self._less(self.iop("+", base, r), bound) for r in rows]
        if s.n_guard is not None:
            base, bound = (self.to_s32(self.expr(e)) for e in s.n_guard)
            col_ok = [self._less(self.iop("+", base, cc), bound) for cc in cols]
        vals = [self._narrow(buf.dtype, v) for v in c]
        ty = STORE_TYPE[buf.dtype]
        for h, row in enumerate(rows):
            start = self.iop("+", flat, self.iop("+", self.iop("*", row, k(ldm)), self.iop("*", cols[0], k(ldn))))
            pair = vals[2 * h:2 * h + 2]
            if (not s.atomic and s.n_guard is None and ldn == 1
                    and self.known_alignment(buf, start) >= 2 * buf.esize):     # one 2-element store
                when = self._when(row_ok[h])
                if when is not None:
                    self.em.emit(f"{when}st.{buf.space}.v2.{ty} {self.address(buf, start)}, {{{pair[0]}, {pair[1]}}}")
                continue
            for j, v in enumerate(pair):
                when = self._when(row_ok[h], col_ok[j])
                if when is None:
                    continue
                addr = self.address(buf, self.iop("+", start, k(j * ldn)))
                if s.atomic:
                    self.em.emit(f"{when}red.{buf.space}.add.f32 {addr}, {v}")
                else:
                    self.em.emit(f"{when}st.{buf.space}.{ty} {addr}, {v}")

    # ---- helpers --------------------------------------------------------------------------
    def _frag(self, ref: str, role: str) -> tuple[str, list[Val]]:
        """(element type, registers) of a fragment reference: `name`, or `name[index]` into a
        family, where the index must be a compile-time constant."""
        m = _REF.match(ref)
        if m is None or m.group(1) not in self.frags:
            raise self.error(f"unknown register fragment {ref!r}")
        frag_role, dtype, regs = self.frags[m.group(1)]
        if frag_role != role:
            raise self.error(f"fragment {ref!r} has role {frag_role!r}, used as {role!r}")
        if m.group(2) is None:
            if len(regs) != 1:
                raise self.error(f"fragment family {ref!r} used without an index")
            return dtype, regs[0]
        tree = _parse_index(m.group(2))
        if tree is None:
            raise self.error(f"fragment {ref!r}: unsupported index expression")
        i = self._const_index(tree, ref)
        if not 0 <= i < len(regs):
            raise self.error(f"fragment {ref!r}: index {i} out of range")
        return dtype, regs[i]

    def _const_index(self, node, ref: str) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            v = self.env.get(node.id)
            if v is None or not v.is_const:
                raise self.error(f"fragment {ref!r}: {node.id} is not a compile-time constant")
            return int(v.const)
        if isinstance(node, ast.BinOp) and type(node.op) in _INDEX_OPS:
            return _INDEX_OPS[type(node.op)](self._const_index(node.left, ref), self._const_index(node.right, ref))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._const_index(node.operand, ref)
        raise self.error(f"fragment {ref!r}: unsupported index expression")

    def _lane_group(self) -> tuple[Val, Val]:
        """(g, t) = (lane / 4, lane % 4)."""
        return self.iop("/", self.lane, const_int(4)), self.iop("%", self.lane, const_int(4))

    def _less(self, a: Val, b: Val) -> Val:
        if a.is_const and b.is_const:
            return const_pred(a.const < b.const)
        p = self.em.reg("pred")
        self.em.emit(f"setp.lt.s32 {p}, {self.em.materialize(a)}, {b}")
        return p

    def _when(self, *preds) -> str | None:
        """The `@p ` prefix for the conjunction of `preds` (None: no condition); None if it is
        known false."""
        live = []
        for p in preds:
            if p is None or (p.is_const and p.const):
                continue
            if p.is_const:
                return None
            live.append(p)
        if not live:
            return ""
        p = live[0]
        for q in live[1:]:
            r = self.em.reg("pred")
            self.em.emit(f"and.pred {r}, {p}, {q}")
            p = r
        return f"@{p} "
