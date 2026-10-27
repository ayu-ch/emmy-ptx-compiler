"""Matrix products: plain, batched, broadcast, transposed, matrix-vector, with a bias."""

from torch_cases import MM, ROUND, case

# 16-bit matmuls run on tensor cores; the error is the output's rounding to f16 / bf16.
H, B = "dtype=torch.float16", "dtype=torch.bfloat16"
HALF_MM, BF16_MM = dict(rtol=2e-2, atol=1e-1), dict(rtol=5e-2, atol=5e-1)

CASES = [
    case("m_mm", "a = torch.randn(128, 96); b = torch.randn(96, 80); a @ b", MM),
    case("m_bmm", "a = torch.randn(4, 64, 32); b = torch.randn(4, 32, 48); torch.bmm(a, b)", MM),
    case("m_bcast_matmul", "a = torch.randn(4, 64, 32); b = torch.randn(32, 48); a @ b", MM),
    case("m_linear_bias", "x = torch.randn(64, 128); w = torch.randn(96, 128); b = torch.randn(96); "
                          "F.linear(x, w, b)", MM),
    case("m_einsum", "a = torch.randn(64, 32); b = torch.randn(32, 48); torch.einsum('ij,jk->ik', a, b)", MM),
    case("m_addmm", "c = torch.randn(64, 48); a = torch.randn(64, 32); b = torch.randn(32, 48); "
                    "torch.addmm(c, a, b)", MM),
    # emmy-ml 0.3.11's schedule copies 16 bytes (cp.async) from an address that is not 16-byte
    # aligned, in its own CUDA as well. emmyptx cannot prove the alignment and stops at compile.
    case("m_transpose_lhs", "a = torch.randn(96, 128); b = torch.randn(96, 80); a.t() @ b", MM),
    # emmy-ml 0.3.11's IR reads only v[0] and has output shape [256, 128], not (256,).
    case("m_matvec", "a = torch.randn(256, 128); v = torch.randn(128); a @ v", MM),
    case("m_outer", "a = torch.randn(64); b = torch.randn(96); torch.outer(a, b)", ROUND),
    # The fragment paths emmy-ml 0.3.11 picks on sm_86: straight from global memory with a
    # zero-filled K tail (m_f16_mm), with row / column guards (_odd); ldmatrix from swizzled
    # shared memory (_staged, bf16, bmm), with a transposed B (linear).
    case("m_f16_mm", f"a = torch.randn(256, 256, {H}); b = torch.randn(256, 256, {H}); a @ b", HALF_MM),
    case("m_f16_mm_odd", f"a = torch.randn(100, 200, {H}); b = torch.randn(200, 72, {H}); a @ b", HALF_MM),
    case("m_f16_mm_staged", f"a = torch.randn(512, 512, {H}); b = torch.randn(512, 512, {H}); a @ b", HALF_MM),
    case("m_bf16_mm", f"a = torch.randn(512, 512, {B}); b = torch.randn(512, 512, {B}); a @ b", BF16_MM),
    case("m_f16_bmm", f"a = torch.randn(4, 128, 64, {H}); b = torch.randn(4, 64, 96, {H}); torch.bmm(a, b)", HALF_MM),
    case("m_f16_linear", f"x = torch.randn(64, 256, {H}); w = torch.randn(512, 256, {H}); F.linear(x, w)", HALF_MM),
]
