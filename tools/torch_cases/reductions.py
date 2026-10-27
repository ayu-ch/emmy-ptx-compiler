"""Reductions over the whole tensor, a row, a column, or several dimensions."""

from torch_cases import RED, REDUCE_KNOBS, ROUND, case

CASES = [
    case("r_sum", "x = torch.randn(8192); x.sum()", RED, knobs=REDUCE_KNOBS),
    case("r_big_sum_default", "x = torch.randn(1 << 20); x.sum()", dict(rtol=1e-3, atol=1e-2)),
    case("r_mean_rows", "x = torch.randn(128, 300); x.mean(-1)", RED),
    case("r_sum_dim0", "x = torch.randn(256, 64); x.sum(0)", RED),
    case("r_sum_multi_dim", "x = torch.randn(8, 32, 64); x.sum((0, 2))", RED),
    case("r_amax_amin", "x = torch.randn(96, 200); x.amax(-1) - x.amin(-1)", ROUND),
    case("r_prod", "x = torch.randn(64, 32); torch.prod(1 + 0.01 * x, -1)", RED),
    case("r_var_std", "x = torch.randn(64, 512); x.var(-1) + x.std(-1)", RED),
    case("r_vector_norm", "x = torch.randn(64, 512); torch.linalg.vector_norm(x, dim=-1)", RED),
    case("r_logsumexp", "x = torch.randn(64, 512); torch.logsumexp(x, -1)", RED),
    case("r_argmax", "x = torch.randn(64, 512); x.argmax(-1)", ROUND),
    case("r_max_with_idx", "x = torch.randn(64, 512); x.max(-1)", ROUND),
    case("r_cumsum", "x = torch.randn(32, 256); torch.cumsum(x, -1)", RED),
]
