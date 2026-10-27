"""Integer arithmetic for indices and addresses.

Almost all integer work in a kernel is index arithmetic: decoding the thread id into tile
axes and flattening `buf[i, j]` into an element offset. `IntMath.iop` emits it with four
optimizations that matter for the generated code:

- Constant folding and identities: `(a0 * 2) + 0` costs one multiply.
- Affine splitting: every `+ constant` / `* constant` result is remembered as
  `base + constant` (`self.affine`), so an address can be `[addr(base) + constant]` and the
  copies of an unrolled loop share one address register.
- Common-subexpression elimination, scoped by block (`self.cse`): an identical instruction
  computed earlier in an enclosing block is reused.
- Facts about values: ranges (`self.ranges`) and known divisors (`self.mults`). A value
  known to be non-negative divides unsigned (a shift for powers of two), a loop whose start
  is a thread index still has a known trip count, and a known divisor of an element offset
  proves the alignment a vector access needs.

Integer registers are written once (see `KernelCompiler.define`), which is what makes caching
them sound; loop counters, the one exception, are `forget`-ed before they change.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

from .ptx import Val, const_int


def c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero), for constant folding."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


INT_INSTR = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32", "/": "div.s32", "//": "div.s32",
             "%": "rem.s32", "<<": "shl.b32", ">>": "shr.s32", "&": "and.b32", "|": "or.b32",
             "^": "xor.b32", "max": "max.s32", "min": "min.s32"}
INT_FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
            "/": c_div, "//": c_div, "%": lambda a, b: a - b * c_div(a, b),
            "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b, "&": lambda a, b: a & b,
            "|": lambda a, b: a | b, "^": lambda a, b: a ^ b, "max": max, "min": min}
COMMUTATIVE = {"+", "*", "&", "|", "^", "max", "min", "==", "!="}
_HUGE = 1 << 30                                           # "divisible by anything" (the value 0)


class IntMath:
    """Mixin: integer instructions with folding, affine tracking, CSE, ranges and divisors.

    Expects `self.em` (the Emitter) and `self.error` (the exception type to raise).
    """

    def init_intmath(self) -> None:
        self.cse: list[dict] = [{}]                     # one table per open block
        self.affine: dict[str, tuple[Val, int]] = {}    # register -> (base register, constant)
        self.ranges: dict[str, tuple[int, int]] = {}    # register -> (lo, hi)
        self.mults: dict[str, int] = {}                 # register -> known divisor
        self.sums: dict[str, tuple[Val, Val]] = {}      # register -> (x, y) when it is x + y

    # ---- the entry point ----------------------------------------------------------------
    def iop(self, op: str, a: Val, b: Val) -> Val:
        """`a op b` for integers."""
        if op not in INT_INSTR:
            raise self.error(f"integer operator {op!r} is not supported")
        if a.is_const and b.is_const:
            return const_int(INT_FOLD[op](a.const, b.const))
        if a.is_const and op in COMMUTATIVE:
            a, b = b, a
        if b.is_const:
            k = b.const
            if (op in ("+", "-", "|", "^", "<<", ">>") and k == 0) or (op in ("*", "/", "//") and k == 1):
                return a
            if (op == "*" and k == 0) or (op == "%" and k == 1):
                return const_int(0)
        result = self._emit_iop(op, a, b)
        if not result.is_const:
            if result.text not in self.ranges:
                r = _range_after(op, self.range_of(a), self.range_of(b))
                if r is not None:
                    self.ranges[result.text] = r
            if result.text not in self.mults:
                m = self._mult_after(op, a, b)
                if m > 1:
                    self.mults[result.text] = m
        return result

    def _emit_iop(self, op: str, a: Val, b: Val) -> Val:
        if op in ("+", "-"):
            (ra, ka), (rb, kb) = self.split(a), self.split(b)
            sign = 1 if op == "+" else -1
            if rb is None:                                # x +/- constant
                return self._offset(ra, ka + sign * kb)
            if ra is None and op == "+":
                return self._offset(rb, ka + kb)
            if ra is None:                                # constant - x
                return self._offset(self.raw("sub.s32", const_int(0), rb), ka - kb)
            if op == "-" and ra.text in self.sums:        # (x + y) - x = y
                x, y = self.sums[ra.text]
                if rb.text in (x.text, y.text):
                    return self._offset(y if rb.text == x.text else x, ka - kb)
            r = self.raw(INT_INSTR[op], ra, rb)
            if op == "+":
                self.sums[r.text] = (ra, rb)
            return self._offset(r, ka + sign * kb)
        if op == "*" and b.is_const:                      # (base + c) * k = base * k + c * k
            ra, ka = self.split(a)
            return self._offset(self.raw("mul.lo.s32", ra, b), ka * b.const)
        if op in ("/", "//", "%") and b.is_const and b.const > 0:
            r = self.range_of(a)
            if r is not None and r[0] >= 0:               # non-negative: unsigned is cheaper
                k = int(b.const)
                if k & (k - 1) == 0:                      # a power of two: a shift or a mask
                    if op == "%":
                        return self.raw("and.b32", a, const_int(k - 1))
                    return self.raw("shr.u32", a, const_int(k.bit_length() - 1))
                return self.raw("rem.u32" if op == "%" else "div.u32", a, b)
        return self.raw(INT_INSTR[op], a, b)

    # ---- affine values ------------------------------------------------------------------
    def split(self, v: Val) -> tuple[Val | None, int]:
        """`v` as (register base, constant offset); a constant is (None, value)."""
        if v.is_const:
            return None, int(v.const)
        return self.affine.get(v.text, (v, 0))

    def _offset(self, base: Val, k: int) -> Val:
        """The register `base + k`, remembered as affine so addresses can split it again.

        Emitted lazily: when every use folds `k` into an address, the add never appears.
        """
        if k == 0:
            return base
        key = ("add.s32", base.text, str(k))
        r = self._cse_get(key)
        if r is None:
            r = self.em.reg("s32")
            self.em.emit_lazy(r, f"add.s32 {r}, {base}, {k}")
            self.cse[-1][key] = r
        self.affine[r.text] = (base, k)
        return r

    # ---- common subexpressions ----------------------------------------------------------
    def raw(self, instr: str, a: Val, b: Val) -> Val:
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

    def remember(self, key: tuple, value: Val) -> None:
        self.cse[-1][key] = value

    def _cse_get(self, key):
        for table in reversed(self.cse):
            if key in table:
                return table[key]
        return None

    def forget(self, reg: Val) -> None:
        """`reg` is about to change: drop every cached result and affine fact that read it."""
        for table in self.cse:                            # (over-forgetting is safe)
            for key in [k for k, v in table.items() if reg.text in repr((k, v))]:
                del table[key]
        for key in [k for k, (base, _) in self.affine.items() if k == reg.text or base.text == reg.text]:
            del self.affine[key]
        for key in [k for k, (x, y) in self.sums.items() if reg.text in (k, x.text, y.text)]:
            del self.sums[key]

    @contextmanager
    def block(self):
        """Code that may not run (a loop body, a branch): its cached values die with it."""
        self.cse.append({})
        try:
            yield
        finally:
            self.cse.pop()

    # ---- facts about values -------------------------------------------------------------
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

    def mult_of(self, v: Val) -> int:
        """A known divisor of an integer value (1 when nothing is known)."""
        if v.is_const:
            return abs(int(v.const)) or _HUGE
        known = self.mults.get(v.text, 1)
        if v.text in self.affine:                         # e.g. (x + 32) * 512 = mul(x, 512) + 16384:
            base, k = self.affine[v.text]                 # both facts hold, keep the stronger one
            return max(known, math.gcd(self.mult_of(base), abs(k) or _HUGE))
        return known

    def _mult_after(self, op: str, a: Val, b: Val) -> int:
        ma, mb = self.mult_of(a), self.mult_of(b)
        if op in ("+", "-", "%", "max", "min"):           # a % b = a - b * q
            return math.gcd(ma, mb)
        if op == "*":
            return min(ma * mb, _HUGE)
        if op == "<<" and b.is_const:
            return min(ma << int(b.const), _HUGE)
        # Bitwise ops, by the low bits known to be zero: a bit of a & b is zero where either
        # operand's is, a bit of a | b or a ^ b where both are (swizzles are XORs).
        if op == "&":
            return max(ma & -ma, mb & -mb)
        if op in ("|", "^"):
            return min(ma & -ma, mb & -mb)
        return 1


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
