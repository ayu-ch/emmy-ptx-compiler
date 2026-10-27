"""PTX building blocks: typed values, virtual registers, labels and the instruction stream.

Everything the compiler computes is a `Val`: either a virtual register (`%f3`) or a
compile-time constant (`42`, `0f3F800000`). Keeping constants as Python numbers until an
instruction really needs a register lets the compiler fold index arithmetic such as
`(a0 * 4) + 0` at compile time, and lets `[%rd5+16]` addressing absorb constant offsets.

PTX registers are virtual and unlimited, so every new value gets a fresh one and `ptxas`
does the real register allocation.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

# Value types the compiler works with, and the register class each one lives in.
#   s32  - integers: indices, loop counters, axis variables (signed, like C `int`)
#   u64  - addresses of global memory
#   f32  - all floating-point math
#   pred - the 1-bit results of comparisons
#   b16  - 16-bit elements on their way to or from memory (f16, bf16, bool)
#   b32  - tensor-core fragments: two 16-bit elements packed in one register
REG_CLASS = {
    "pred": ("p", ".pred"),
    "s32": ("r", ".b32"),
    "u64": ("rd", ".b64"),
    "f32": ("f", ".f32"),
    "b16": ("h", ".b16"),
    "b32": ("b", ".b32"),
}


def f32_bits(x: float) -> str:
    """A PTX float literal: `0f` followed by the IEEE-754 bits, so it can never round wrongly."""
    return "0f" + struct.pack(">f", float(x)).hex().upper()


@dataclass(frozen=True)
class Val:
    """A typed operand: a register, or a constant whose Python value is in `const`."""

    text: str
    ty: str
    const: int | float | bool | None = None

    @property
    def is_const(self) -> bool:
        return self.const is not None

    def __str__(self) -> str:
        return self.text


def const_int(v: int) -> Val:
    return Val(str(int(v)), "s32", int(v))


def const_f32(v: float) -> Val:
    return Val(f32_bits(v), "f32", float(v))


def const_pred(v: bool) -> Val:
    # Predicates have no literal syntax in PTX; `materialize` turns this into a setp.
    return Val("1" if v else "0", "pred", bool(v))


_REGISTER = re.compile(r"%[a-z]+\d+")


@dataclass
class _Lazy:
    dst: str
    text: str


class Emitter:
    """Collects the kernel body's instructions and hands out registers and labels."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counts = {ty: 0 for ty in REG_CLASS}
        self.n_labels = 0

    def reg(self, ty: str) -> Val:
        self.counts[ty] += 1
        prefix, _ = REG_CLASS[ty]
        return Val(f"%{prefix}{self.counts[ty]}", ty)

    def emit(self, instr: str) -> None:
        self.lines.append(f"    {instr};")

    def emit_lazy(self, dst: Val, instr: str) -> None:
        """An instruction that is kept only if some other instruction reads its result `dst`.

        The compiler computes some values speculatively (e.g. the full value of an index whose
        uses may all fold into address offsets, or a product that may become part of an FMA).
        Dropping the unused ones here keeps the PTX small; ptxas would delete them anyway.
        """
        self.lines.append(_Lazy(dst.text, f"    {instr};"))

    def render(self) -> list[str]:
        """The body lines, without the lazy instructions nobody reads."""
        used: set[str] = set()
        for line in self.lines:
            if isinstance(line, str):
                used.update(_REGISTER.findall(line))
        changed = True
        while changed:                                    # a kept lazy line may read another one
            changed = False
            for line in self.lines:
                if isinstance(line, _Lazy) and line.dst in used:
                    new = set(_REGISTER.findall(line.text.split(",", 1)[1])) - used
                    if new:
                        used |= new
                        changed = True
        return [line.text if isinstance(line, _Lazy) else line
                for line in self.lines if not isinstance(line, _Lazy) or line.dst in used]

    def label(self, hint: str) -> str:
        self.n_labels += 1
        return f"$L{self.n_labels}_{hint}"

    def place(self, label: str) -> None:
        self.lines.append(f"{label}:")

    def declarations(self) -> list[str]:
        """One `.reg` line per register class that was used, sized to the highest number."""
        out = []
        for ty, (prefix, ptx_type) in REG_CLASS.items():
            if self.counts[ty]:
                out.append(f"    .reg {ptx_type} %{prefix}<{self.counts[ty] + 1}>;")
        return out

    # ---- turning constants into registers when an instruction needs one ----
    def materialize(self, v: Val) -> Val:
        """Return `v` as a register: constants are moved into a fresh one."""
        if not v.is_const:
            return v
        r = self.reg(v.ty)
        if v.ty == "pred":
            self.emit(f"setp.ne.s32 {r}, {1 if v.const else 0}, 0")
        else:
            self.emit(f"mov.{'b32' if v.ty == 's32' else v.ty} {r}, {v}")
        return r
