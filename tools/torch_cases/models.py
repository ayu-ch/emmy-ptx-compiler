"""Small models as a user would write them: functions over several ops, and an nn.Module."""

from torch_cases import MM, case

CASES = [
    # A Llama-style feed-forward sublayer: RMSNorm, SwiGLU MLP, residual. The weights are
    # inputs; the constant scales are 1/sqrt(fan_in), keeping every intermediate O(1).
    case("t_swiglu_ffn",
         "def ffn(x, g, w1, w3, w2):\n"
         "    h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g\n"
         "    a = F.silu((h @ w1) * 0.0625) * ((h @ w3) * 0.0625)\n"
         "    return x + (a @ w2) * 0.04419417\n"
         "x = torch.randn(64, 256); g = torch.randn(256)\n"
         "w1 = torch.randn(256, 512); w3 = torch.randn(256, 512); w2 = torch.randn(512, 256)\n"
         "ffn(x, g, w1, w3, w2)", MM),
    # Causal multi-head attention through torch's own fused entry point.
    case("t_causal_sdpa",
         "q = torch.randn(1, 4, 128, 64); k = torch.randn(1, 4, 128, 64); v = torch.randn(1, 4, 128, 64)\n"
         "F.scaled_dot_product_attention(q, k, v, is_causal=True)", MM),
    # An nn.Module: Emmy keeps its weights as named graph constants, which the harness fills
    # from the module the reference builds.
    case("t_module_mlp",
         "torch.manual_seed(0)\n"
         "mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 64))\n"
         "x = torch.randn(32, 128)\n"
         "mlp(x)", MM),
]
