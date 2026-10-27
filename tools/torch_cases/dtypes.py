"""Buffers other than f32."""

from torch_cases import HALF, ROUND, case

CASES = [
    case("d_f16_add", "x = torch.randn(1024, dtype=torch.float16); y = torch.randn(1024, dtype=torch.float16); "
                      "x + y", HALF),
    case("d_f16_gelu", "x = torch.randn(1024, dtype=torch.float16); F.gelu(x)", HALF),
    case("d_bf16_add", "x = torch.randn(1024, dtype=torch.bfloat16); y = torch.randn(1024, dtype=torch.bfloat16); "
                       "x + y", HALF),
    case("d_f32_to_f16", "x = torch.randn(1024); x.half()", HALF),
    case("d_f64_add", "x = torch.randn(1024, dtype=torch.float64); x * 2 + 1", ROUND),
    case("d_i32_arith", "a = torch.randint(-4, 4, (1024,), dtype=torch.int32); a * 3 + 1", ROUND),
    # torch's // and % round toward minus infinity; PTX's div.s32 / rem.s32 (and C, which
    # Emmy's CUDA uses) truncate toward zero. They differ for negative operands.
    case("d_i32_floordiv_mod", "a = torch.randint(-8, 8, (1024,), dtype=torch.int32); a // 3 + (a % 3) * 100", ROUND),
]
