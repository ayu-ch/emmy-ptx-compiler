"""Matrix products: plain, batched, broadcast, transposed, matrix-vector, with a bias."""

from torch_cases import MM, ROUND, case

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
    case("m_f16_mm", "a = torch.randn(256, 256, dtype=torch.float16); b = torch.randn(256, 256, dtype=torch.float16); "
                     "a @ b", dict(rtol=2e-2, atol=1e-1)),
]
