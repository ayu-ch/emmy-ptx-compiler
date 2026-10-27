"""Data movement: transposes, views, slices, concatenation, indexing."""

from torch_cases import ROUND, case

CASES = [
    case("s_transpose", "x = torch.randn(64, 96); x.t().contiguous()", ROUND),
    # emmy-ml 0.3.11 lowers this permute as a reshape (its kernel is named k_reshape): the
    # output is x.reshape(32, 8, 16) * 2.
    case("s_permute", "x = torch.randn(8, 16, 32); x.permute(2, 0, 1).contiguous() * 2", ROUND),
    case("s_reshape", "x = torch.randn(4096); x.reshape(64, 64) + 1", ROUND),
    case("s_slice", "x = torch.randn(64, 128); x[:, 10:50] * 2", ROUND),
    # emmy-ml 0.3.11 loses the step: its IR reads x[0:512], so the output is x[:512] + 1.
    case("s_strided_slice", "x = torch.randn(1024); x[::2] + 1", ROUND),
    case("s_cat", "a = torch.randn(32, 64); b = torch.randn(32, 48); torch.cat([a, b], dim=-1)", ROUND),
    case("s_stack", "a = torch.randn(32, 64); b = torch.randn(32, 64); torch.stack([a, b])", ROUND),
    case("s_flip", "x = torch.randn(64, 128); x.flip(-1)", ROUND),
    case("s_roll", "x = torch.randn(64, 128); torch.roll(x, 3, -1)", ROUND),
    case("s_repeat", "x = torch.randn(32, 64); x.repeat(2, 1) + 0", ROUND),
    case("s_expand", "x = torch.randn(1, 128); x.expand(64, 128) + 0", ROUND),
    case("s_tril", "x = torch.randn(64, 64); torch.tril(x)", ROUND),
    case("s_embedding", "idx = torch.randint(0, 100, (64,)); table = torch.randn(100, 32); "
                        "table[idx.abs() % 100]", ROUND),
    case("s_gather", "x = torch.randn(32, 64); idx = torch.randint(0, 64, (32, 8)); "
                     "torch.gather(x, 1, idx.abs() % 64)", ROUND),
]
