"""Cases for tools/check_torch.py, one module per category, each with a list CASES.

A case is torch code `<preamble>; <final expression>` (or several lines, the last one the
expression). Emmy traces only the final expression, so every tensor it reads must be a
preamble variable: the float64 reference rebinds each one to the seeded input the GPU gets.
A tensor made inside the expression (`torch.ones(n).tril()`) becomes a graph input too, and
is fed random data like any other. The code is run as Python.

Tolerances are against that float64 PyTorch reference:
  ROUND  ops one f32 rounding away from exact
  TRANS  transcendental functions (emmyptx and Emmy both use fast approximations, e.g. ex2.approx)
  RED    reductions of a few thousand terms (the summation order differs from torch's)
  MM     matmul and attention (dot products of length K in f32)
  HALF   f16 storage
"""

ROUND = dict(rtol=1e-6, atol=1e-6)
TRANS = dict(rtol=1e-5, atol=1e-5)
RED = dict(rtol=1e-4, atol=1e-4)
MM = dict(rtol=1e-3, atol=1e-3)
HALF = dict(rtol=1e-2, atol=1e-2)
REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def case(id: str, code: str, tol: dict, knobs: str = "") -> dict:
    """`knobs` is passed to Emmy as EMMY_KNOBS (e.g. to pick a reduction schedule)."""
    return dict(id=id, code=code, knobs=knobs, **tol)
