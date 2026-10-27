"""Patterns Emmy fuses into one or a few kernels: softmax, norms, attention, losses."""

from torch_cases import MM, case

CASES = [
    case("f_softmax", "x = torch.randn(64, 1000); torch.softmax(x, -1)", dict(rtol=1e-3, atol=1e-5)),
    case("f_softmax_dim0", "x = torch.randn(256, 64); torch.softmax(x, 0)", dict(rtol=1e-3, atol=1e-5)),
    case("f_log_softmax", "x = torch.randn(48, 256); torch.log_softmax(x, -1)", dict(rtol=1e-3, atol=1e-4)),
    case("f_layer_norm_affine", "x = torch.randn(64, 768); w = torch.randn(768); b = torch.randn(768); "
                                "F.layer_norm(x, (768,), w, b)", dict(rtol=1e-3, atol=1e-4)),
    case("f_rms_norm", "x = torch.randn(32, 512); w = torch.randn(512); F.rms_norm(x, (512,), w, 1e-6)",
         dict(rtol=1e-3, atol=1e-4)),
    case("f_normalize", "x = torch.randn(64, 256); F.normalize(x, dim=-1)", dict(rtol=1e-4, atol=1e-5)),
    case("f_cosine_sim", "a = torch.randn(64, 256); b = torch.randn(64, 256); F.cosine_similarity(a, b, dim=-1)",
         dict(rtol=1e-4, atol=1e-5)),
    case("f_batch_norm_eval", "x = torch.randn(32, 64); m = torch.randn(64); v = torch.randn(64); w = torch.randn(64); "
                              "b = torch.randn(64); F.batch_norm(x, m, v.abs() + 0.5, w, b, training=False)",
         dict(rtol=1e-4, atol=1e-5)),
    case("f_soft_xent", "x = torch.randn(32, 100); p = torch.randn(32, 100); "
                        "-(F.log_softmax(x, -1) * torch.softmax(p, -1)).sum(-1).mean()", dict(rtol=1e-3, atol=1e-4)),
    case("f_sdpa", "q = torch.randn(1, 2, 64, 32); k = torch.randn(1, 2, 64, 32); v = torch.randn(1, 2, 64, 32); "
                   "F.scaled_dot_product_attention(q, k, v)", MM),
    # Launched with 512-thread blocks: fully unrolling its dot products used to need more
    # registers than such a block can have, and the launch failed.
    case("f_sdpa_f16", "q = torch.randn(1, 4, 128, 64, dtype=torch.float16); "
                       "k = torch.randn(1, 4, 128, 64, dtype=torch.float16); "
                       "v = torch.randn(1, 4, 128, 64, dtype=torch.float16); F.scaled_dot_product_attention(q, k, v)",
         dict(rtol=1e-2, atol=1e-2)),
    # A causal mask made inside the expression (torch.ones(...).tril()) would be fed random data
    # like any graph input, so the mask is taken from the scores: s.triu(1) != 0 above the diagonal.
    case("f_masked_attn_manual",
         "def attn(q, k, v):\n"
         "    s = q @ k.t() * 0.17677669\n"
         "    return torch.softmax(s.masked_fill(s.triu(1) != 0, float('-inf')), -1) @ v\n"
         "q = torch.randn(64, 32); k = torch.randn(64, 32); v = torch.randn(64, 32)\n"
         "attn(q, k, v)", MM),
]
