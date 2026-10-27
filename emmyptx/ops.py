"""Elementwise operations: one IR op name -> the PTX instructions computing it.

`lower_op(em, name, args, dst)` writes `name(args)` into the register `dst`. Every value is
f32, the dtype of every Emmy computation in this hackathon.

PTX has only a handful of math instructions (`ex2.approx`, `lg2.approx`, `rcp`, `sqrt`,
`sin.approx`, ...), so the transcendental functions are built from them the way a C math
library is built. Plain `add.f32` / `mul.f32` are emitted without a rounding modifier, which
lets ptxas fuse a multiply and an add into an FMA exactly as nvcc does for Emmy's own kernels.
"""

from __future__ import annotations

import math

from .ptx import Emitter, Val, const_f32

LOG2_E = 1.4426950408889634   # exp(x) = 2^(x * log2(e))
LN_2 = 0.6931471805599453     # log(x) = log2(x) * ln(2)
TWO_PI = 2.0 * math.pi


class OpError(NotImplementedError):
    pass


def _reg(em: Emitter, v: Val) -> Val:
    return em.materialize(v)


def _bin(em: Emitter, instr: str, dst: Val, a: Val, b: Val) -> None:
    """Two-operand instruction. PTX takes a constant in the second slot, so the first is a register."""
    em.emit(f"{instr} {dst}, {_reg(em, a)}, {b}")


def _exp(em: Emitter, dst: Val, x: Val) -> None:
    t = em.reg("f32")
    em.emit(f"mul.f32 {t}, {_reg(em, x)}, {const_f32(LOG2_E)}")
    em.emit(f"ex2.approx.f32 {dst}, {t}")


def _log(em: Emitter, dst: Val, x: Val) -> None:
    t = em.reg("f32")
    em.emit(f"lg2.approx.f32 {t}, {_reg(em, x)}")
    em.emit(f"mul.f32 {dst}, {t}, {const_f32(LN_2)}")


def _select01(em: Emitter, dst: Val, cmp: str, a: Val, b: Val) -> None:
    """Comparisons produce 1.0 / 0.0 floats (Emmy's convention for masks)."""
    p = em.reg("pred")
    em.emit(f"setp.{cmp}.f32 {p}, {_reg(em, a)}, {b}")
    em.emit(f"selp.f32 {dst}, {const_f32(1.0)}, {const_f32(0.0)}, {p}")


def _truthy(em: Emitter, x: Val) -> Val:
    """A predicate that is true where x != 0 (C truthiness of a float, NaN included)."""
    p = em.reg("pred")
    em.emit(f"setp.neu.f32 {p}, {_reg(em, x)}, {const_f32(0.0)}")
    return p


def _tanh(em: Emitter, dst: Val, x: Val) -> None:
    # tanh(x) = sign(x) * (1 - e) / (1 + e) with e = exp(-2|x|): no overflow for any x.
    ax, e, num, den, t = (em.reg("f32") for _ in range(5))
    em.emit(f"abs.f32 {ax}, {_reg(em, x)}")
    em.emit(f"mul.f32 {ax}, {ax}, {const_f32(-2.0)}")
    _exp(em, e, ax)
    em.emit(f"sub.f32 {num}, {const_f32(1.0)}, {e}")
    em.emit(f"add.f32 {den}, {e}, {const_f32(1.0)}")
    em.emit(f"div.rn.f32 {t}, {num}, {den}")
    em.emit(f"copysign.f32 {dst}, {_reg(em, x)}, {t}")


def _erf(em: Emitter, dst: Val, x: Val) -> None:
    # Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7):
    #   erf(|x|) = 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) exp(-x^2),  t = 1 / (1 + p |x|)
    a = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
    xr = _reg(em, x)
    ax, t, poly, x2, e, y = (em.reg("f32") for _ in range(6))
    em.emit(f"abs.f32 {ax}, {xr}")
    em.emit(f"fma.rn.f32 {t}, {ax}, {const_f32(0.3275911)}, {const_f32(1.0)}")
    em.emit(f"rcp.rn.f32 {t}, {t}")
    em.emit(f"mov.f32 {poly}, {const_f32(a[4])}")
    for coef in reversed(a[:4]):                       # Horner: (((a5 t + a4) t + a3) t + a2) t + a1
        em.emit(f"fma.rn.f32 {poly}, {poly}, {t}, {const_f32(coef)}")
    em.emit(f"mul.f32 {poly}, {poly}, {t}")
    em.emit(f"mul.f32 {x2}, {ax}, {ax}")
    em.emit(f"neg.f32 {x2}, {x2}")
    _exp(em, e, x2)
    em.emit(f"mul.f32 {y}, {poly}, {e}")
    em.emit(f"sub.f32 {y}, {const_f32(1.0)}, {y}")
    em.emit(f"copysign.f32 {dst}, {xr}, {y}")


def _sincos(em: Emitter, dst: Val, x: Val, fn: str) -> None:
    # sin.approx / cos.approx are accurate on [-pi, pi]: reduce by the nearest multiple of 2*pi.
    k, r = em.reg("f32"), em.reg("f32")
    xr = _reg(em, x)
    em.emit(f"mul.f32 {k}, {xr}, {const_f32(1.0 / TWO_PI)}")
    em.emit(f"cvt.rni.f32.f32 {k}, {k}")
    em.emit(f"fma.rn.f32 {r}, {k}, {const_f32(-6.28125)}, {xr}")              # 2*pi split in two
    em.emit(f"fma.rn.f32 {r}, {k}, {const_f32(-(TWO_PI - 6.28125))}, {r}")    # parts for accuracy
    em.emit(f"{fn}.approx.f32 {dst}, {r}")


def _pow(em: Emitter, dst: Val, a: Val, b: Val) -> None:
    # a^b = 2^(b * log2|a|), negated for a negative base with an odd integer exponent,
    # NaN for a negative base with a fractional one, and 1 whenever b == 0 (as numpy).
    ar, br = _reg(em, a), _reg(em, b)
    la, r, bi, half, hf = (em.reg("f32") for _ in range(5))
    em.emit(f"abs.f32 {la}, {ar}")
    em.emit(f"lg2.approx.f32 {la}, {la}")
    em.emit(f"mul.f32 {la}, {la}, {br}")
    em.emit(f"ex2.approx.f32 {r}, {la}")
    neg, isint, odd, p = (em.reg("pred") for _ in range(4))
    em.emit(f"setp.lt.f32 {neg}, {ar}, {const_f32(0.0)}")
    em.emit(f"cvt.rzi.f32.f32 {bi}, {br}")
    em.emit(f"setp.eq.f32 {isint}, {bi}, {br}")
    em.emit(f"mul.f32 {half}, {br}, {const_f32(0.5)}")
    em.emit(f"cvt.rzi.f32.f32 {hf}, {half}")
    em.emit(f"setp.neu.f32 {odd}, {hf}, {half}")              # b/2 not an integer: b is odd
    em.emit(f"and.pred {p}, {neg}, {odd}")
    em.emit(f"@{p} neg.f32 {r}, {r}")
    em.emit(f"not.pred {isint}, {isint}")
    em.emit(f"and.pred {p}, {neg}, {isint}")
    em.emit(f"@{p} mov.f32 {r}, 0f7FFFFFFF")
    em.emit(f"setp.eq.f32 {p}, {br}, {const_f32(0.0)}")
    em.emit(f"selp.f32 {dst}, {const_f32(1.0)}, {r}, {p}")


_COMPARE = {"equal": "eq", "not_equal": "neu", "greater": "gt", "less": "lt",
            "greater_equal": "ge", "less_equal": "le"}


def lower_op(em: Emitter, name: str, args: list[Val], dst: Val) -> None:
    """Emit `dst = name(*args)` for an f32 elementwise op."""
    a = args[0] if args else None
    b = args[1] if len(args) > 1 else None
    if name == "add":
        if a.is_const and not b.is_const:                 # commutative: keep the constant second
            a, b = b, a
        _bin(em, "add.f32", dst, a, b)
    elif name == "subtract":
        _bin(em, "sub.f32", dst, a, b)
    elif name == "multiply":
        if a.is_const and not b.is_const:
            a, b = b, a
        _bin(em, "mul.f32", dst, a, b)
    elif name == "divide":
        _bin(em, "div.rn.f32", dst, a, b)
    elif name == "maximum":
        _bin(em, "max.f32", dst, a, b)
    elif name == "minimum":
        _bin(em, "min.f32", dst, a, b)
    elif name == "relu":
        _bin(em, "max.f32", dst, a, const_f32(0.0))
    elif name == "square":
        _bin(em, "mul.f32", dst, a, a)
    elif name in ("copy", "pad", "from_f8e4m3", "from_f8e5m2", "to_f8e4m3", "to_f8e5m2"):
        em.emit(f"mov.f32 {dst}, {a}")
    elif name == "negative":
        em.emit(f"neg.f32 {dst}, {_reg(em, a)}")
    elif name == "abs":
        em.emit(f"abs.f32 {dst}, {_reg(em, a)}")
    elif name == "reciprocal":
        em.emit(f"rcp.rn.f32 {dst}, {_reg(em, a)}")
    elif name == "sqrt":
        em.emit(f"sqrt.rn.f32 {dst}, {_reg(em, a)}")
    elif name == "rsqrt":
        t = em.reg("f32")
        em.emit(f"sqrt.rn.f32 {t}, {_reg(em, a)}")
        em.emit(f"rcp.rn.f32 {dst}, {t}")
    elif name in ("exp", "exp_fast"):
        _exp(em, dst, a)
    elif name == "log":
        _log(em, dst, a)
    elif name == "sigmoid":                              # 1 / (1 + exp(-x))
        t = em.reg("f32")
        em.emit(f"neg.f32 {t}, {_reg(em, a)}")
        _exp(em, t, t)
        em.emit(f"add.f32 {t}, {t}, {const_f32(1.0)}")
        em.emit(f"rcp.rn.f32 {dst}, {t}")
    elif name == "tanh":
        _tanh(em, dst, a)
    elif name == "erf":
        _erf(em, dst, a)
    elif name in ("sin", "cos"):
        _sincos(em, dst, a, name)
    elif name == "pow":
        _pow(em, dst, a, b)
    elif name in _COMPARE:
        _select01(em, dst, _COMPARE[name], a, b)
    elif name == "where":                                # where(c, a, b) = c ? a : b
        em.emit(f"selp.f32 {dst}, {args[1]}, {args[2]}, {_truthy(em, a)}")
    elif name in ("bitwise_and", "bitwise_or"):          # logical and/or of 1.0/0.0 masks
        p = em.reg("pred")
        em.emit(f"{'and' if name == 'bitwise_and' else 'or'}.pred {p}, {_truthy(em, a)}, {_truthy(em, b)}")
        em.emit(f"selp.f32 {dst}, {const_f32(1.0)}, {const_f32(0.0)}, {p}")
    elif name == "bitwise_not":
        _select01(em, dst, "eq", a, const_f32(0.0))
    elif name in ("floor_divide", "mod", "remainder"):   # Python semantics: floor(a / b), a - b*floor(a/b)
        q = em.reg("f32")
        em.emit(f"div.rn.f32 {q}, {_reg(em, a)}, {b}")
        em.emit(f"cvt.rmi.f32.f32 {q}, {q}")
        if name == "floor_divide":
            em.emit(f"mov.f32 {dst}, {q}")
        else:
            em.emit(f"mul.f32 {q}, {q}, {b}")
            em.emit(f"sub.f32 {dst}, {_reg(em, a)}, {q}")
    else:
        raise OpError(f"elementwise op {name!r} is not supported")


# ---- integer elementwise ops (Assigns whose dtype is an integer type) ----
_INT_OPS = {"add": "add.s32", "subtract": "sub.s32", "multiply": "mul.lo.s32",
            "floor_divide": "div.s32", "divide": "div.s32", "mod": "rem.s32", "remainder": "rem.s32",
            "maximum": "max.s32", "minimum": "min.s32", "bitwise_and": "and.b32",
            "bitwise_or": "or.b32", "bitwise_xor": "xor.b32", "left_shift": "shl.b32",
            "right_shift": "shr.s32"}


def lower_int_op(em: Emitter, name: str, args: list[Val], dst: Val) -> None:
    if name == "copy":
        em.emit(f"mov.b32 {dst}, {args[0]}")
    elif name == "abs":
        em.emit(f"abs.s32 {dst}, {_reg(em, args[0])}")
    elif name in _INT_OPS:
        _bin(em, _INT_OPS[name], dst, args[0], args[1])
    else:
        raise OpError(f"integer elementwise op {name!r} is not supported")
