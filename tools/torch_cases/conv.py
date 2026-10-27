"""Convolution and pooling."""

from torch_cases import MM, ROUND, case

CASES = [
    case("c_conv1d", "x = torch.randn(1, 16, 128); w = torch.randn(32, 16, 3); F.conv1d(x, w)", MM),
    case("c_conv2d", "x = torch.randn(1, 8, 32, 32); w = torch.randn(16, 8, 3, 3); F.conv2d(x, w)", MM),
    case("c_avg_pool1d", "x = torch.randn(1, 16, 128); F.avg_pool1d(x, 4)", ROUND),
    case("c_max_pool2d", "x = torch.randn(1, 8, 32, 32); F.max_pool2d(x, 2)", ROUND),
]
