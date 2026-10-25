"""Extra problems for testing generalization: the open set's five categories with other shapes
and ops, in the format of the organizers' tools/expressions_open.py.

    python <hackathon>/tools/generate.py --emmy <venv>/bin/emmy --out out/extra tools/expressions_extra.py
    python tools/run_all.py <hackathon> out/extra
"""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def f64(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _erf(x):
    from math import erf
    return np.vectorize(erf)(x)


def _f32(*arrays):
    return [np.asarray(a).astype(np.float32) for a in arrays]


OPEN = [
    # ---- pointwise: odd sizes, 2-D shapes, broadcasting, more ops ----------------------
    dict(id="p10_odd_add", code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p11_tiny", code="torch.randn(7) * torch.randn(7)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p12_2d_mul", code="torch.randn(33, 77) * torch.randn(33, 77)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p13_bcast_row", code="torch.randn(64, 128) + torch.randn(128)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="p14_bcast_outer", code="torch.randn(48, 1) * torch.randn(1, 96)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="p15_sigmoid", code="torch.sigmoid(torch.randn(3000))",
         ref=lambda x: _f32(1 / (1 + np.exp(-f64(x)))), rtol=1e-5, atol=1e-6),
    dict(id="p16_tanh", code="torch.tanh(torch.randn(4096))",
         ref=lambda x: _f32(np.tanh(f64(x))), rtol=1e-5, atol=1e-6),
    dict(id="p17_gelu", code="torch.nn.functional.gelu(torch.randn(2048))",
         ref=lambda x: _f32(0.5 * f64(x) * (1 + _erf(f64(x) / np.sqrt(2)))), rtol=1e-5, atol=1e-6),
    dict(id="p18_silu", code="x = torch.randn(4096); x * torch.sigmoid(x)",
         ref=lambda x: _f32(f64(x) / (1 + np.exp(-f64(x)))), rtol=1e-5, atol=1e-6),
    dict(id="p19_exp_abs_neg", code="torch.exp(-torch.abs(torch.randn(2048)))",
         ref=lambda x: _f32(np.exp(-np.abs(f64(x)))), rtol=1e-5, atol=1e-6),
    dict(id="p20_div_sqrt", code="x = torch.randn(1024); y = torch.randn(1024); x / torch.sqrt(y * y + 1.0)",
         ref=lambda x, y: _f32(f64(x) / np.sqrt(f64(y) ** 2 + 1)), rtol=1e-5, atol=1e-6),
    dict(id="p21_maxmin", code="x = torch.randn(1536); y = torch.randn(1536); torch.maximum(x, y) - torch.minimum(x, y)",
         ref=lambda x, y: [np.maximum(x, y) - np.minimum(x, y)], rtol=0, atol=0),
    dict(id="p22_leaky", code="x = torch.randn(2000); torch.where(x > 0, x, x * 0.1)",
         ref=lambda x: [np.where(x > 0, x, x * np.float32(0.1))], rtol=1e-6, atol=1e-7),
    dict(id="p23_log", code="x = torch.randn(1024); torch.log(x * x + 1.0)",
         ref=lambda x: _f32(np.log(f64(x) ** 2 + 1)), rtol=1e-5, atol=1e-6),
    dict(id="p24_clamp", code="torch.clamp(torch.randn(3333), -0.5, 0.5)",
         ref=lambda x: [np.clip(x, -0.5, 0.5)], rtol=0, atol=0),
    dict(id="p25_square_3d", code="x = torch.randn(4, 16, 40); x * x * x",
         ref=lambda x: [x * x * x], rtol=1e-6, atol=1e-7),

    # ---- reductions ---------------------------------------------------------------------
    dict(id="r10_sum_odd", code="torch.randn(5000).sum()",
         ref=lambda x: _f32(f64(x).sum()), rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="r11_max", code="torch.randn(4096).max()",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="r12_row_mean", code="torch.randn(128, 300).mean(-1)",
         ref=lambda x: _f32(f64(x).mean(-1)), rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r13_sumsq", code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: _f32((f64(x) ** 2).sum()), rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="r14_row_max", code="torch.randn(96, 200).amax(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="r15_col_sum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: _f32(f64(x).sum(0)), rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r16_abs_sum_rows", code="torch.abs(torch.randn(32, 1000)).sum(-1)",
         ref=lambda x: _f32(np.abs(f64(x)).sum(-1)), rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r17_sum_default", code="torch.randn(20000).sum()",
         ref=lambda x: _f32(f64(x).sum()), rtol=1e-3, atol=1e-3),
    dict(id="r18_rows_default", code="torch.randn(512, 256).sum(-1)",
         ref=lambda x: _f32(f64(x).sum(-1)), rtol=1e-4, atol=1e-4),

    # ---- matmul -------------------------------------------------------------------------
    dict(id="m10_256x128x64", code="torch.randn(256, 128) @ torch.randn(128, 64)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="m11_64x512x32", code="torch.randn(64, 512) @ torch.randn(512, 32)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="m12_odd", code="torch.randn(100, 72) @ torch.randn(72, 120)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="m13_1024", code="torch.randn(1024, 1024) @ torch.randn(1024, 1024)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="m14_tall", code="torch.randn(2048, 64) @ torch.randn(64, 128)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="m15_bias_relu", code="a = torch.randn(128, 96); b = torch.randn(96, 64); c = torch.randn(64); torch.relu(a @ b + c)",
         ref=lambda a, b, c: _f32(np.maximum(f64(a) @ f64(b) + f64(c), 0)), rtol=1e-3, atol=1e-3),

    # ---- fused --------------------------------------------------------------------------
    dict(id="f10_softmax_wide", code="torch.softmax(torch.randn(32, 1000), dim=-1)",
         ref=lambda x: _f32(_softmax(f64(x))), rtol=1e-3, atol=1e-5),
    dict(id="f11_layernorm", code="torch.nn.functional.layer_norm(torch.randn(64, 768), (768,))",
         ref=lambda x: _f32((f64(x) - f64(x).mean(-1, keepdims=True))
                            / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5)), rtol=1e-3, atol=1e-4),
    dict(id="f16_layernorm_inline",
         code="x = torch.randn(32, 512); (x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True) + 1e-5)",
         ref=lambda x: _f32((f64(x) - f64(x).mean(-1, keepdims=True))
                            / np.sqrt(f64(x).var(-1, keepdims=True) + 1e-5)), rtol=1e-3, atol=1e-4),
    dict(id="f12_log_softmax", code="torch.log_softmax(torch.randn(48, 256), dim=-1)",
         ref=lambda x: _f32(np.log(_softmax(f64(x)))), rtol=1e-3, atol=1e-4),
    dict(id="f13_rmsnorm_wide",
         code="x = torch.randn(16, 2048); w = torch.randn(2048); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: _f32(f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6) * f64(w)),
         rtol=1e-3, atol=1e-4),
    dict(id="f14_center", code="x = torch.randn(128, 128); x - x.max(-1, keepdim=True).values",
         ref=lambda x: [x - x.max(-1, keepdims=True)], rtol=1e-6, atol=1e-6),
    dict(id="f15_softmax_small", code="torch.softmax(torch.randn(256, 16), dim=-1)",
         ref=lambda x: _f32(_softmax(f64(x))), rtol=1e-3, atol=1e-5),

    # ---- attention-like -------------------------------------------------------------------
    dict(id="a10_attn_32x16", code="q = torch.randn(32, 16); kt = torch.randn(16, 32); v = torch.randn(32, 16); "
                                   "torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: _f32(_softmax(f64(q) @ f64(kt) / 4.0) @ f64(v)), rtol=1e-3, atol=1e-4),
    dict(id="a11_attn_rect", code="q = torch.randn(96, 64); kt = torch.randn(64, 200); v = torch.randn(200, 48); "
                                  "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: _f32(_softmax(f64(q) @ f64(kt) / 8.0) @ f64(v)), rtol=1e-3, atol=1e-4),
    dict(id="a12_attn_256", code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); "
                                 "torch.softmax(q @ kt * 0.125, dim=-1) @ v",
         ref=lambda q, kt, v: _f32(_softmax(f64(q) @ f64(kt) * 0.125) @ f64(v)), rtol=1e-3, atol=1e-4),
]

# ---- multi-kernel problems and large shapes ------------------------------------------------
OPEN += [
    dict(id="x10_chain_matmul", code="a = torch.randn(64, 96); b = torch.randn(96, 80); c = torch.randn(80, 48); a @ b @ c",
         ref=lambda a, b, c: _f32(f64(a) @ f64(b) @ f64(c)), rtol=1e-3, atol=1e-3),
    dict(id="x11_matmul_softmax", code="torch.softmax(torch.randn(128, 64) @ torch.randn(64, 256), dim=-1)",
         ref=lambda a, b: _f32(_softmax(f64(a) @ f64(b))), rtol=1e-3, atol=1e-5),
    dict(id="x12_matmul_rowsum", code="(torch.randn(256, 128) @ torch.randn(128, 64)).sum(-1)",
         ref=lambda a, b: _f32((f64(a) @ f64(b)).sum(-1)), rtol=1e-3, atol=1e-3),
    dict(id="x13_big_sum_default", code="torch.randn(1 << 20).sum()",
         ref=lambda x: _f32(f64(x).sum()), rtol=1e-3, atol=1e-2),
    dict(id="x14_matmul_2048", code="torch.randn(2048, 512) @ torch.randn(512, 1024)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
    dict(id="x15_softmax_4096", code="torch.softmax(torch.randn(16, 4096), dim=-1)",
         ref=lambda x: _f32(_softmax(f64(x))), rtol=1e-3, atol=1e-6),
    dict(id="x16_attn_512", code="q = torch.randn(512, 64); kt = torch.randn(64, 512); v = torch.randn(512, 64); "
                                 "torch.softmax(q @ kt * 0.125, dim=-1) @ v",
         ref=lambda q, kt, v: _f32(_softmax(f64(q) @ f64(kt) * 0.125) @ f64(v)), rtol=1e-3, atol=1e-4),
    dict(id="x17_norm_then_matmul",
         code="x = torch.randn(64, 256); w = torch.randn(256, 128); "
              "(x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)) @ w",
         ref=lambda x, w: _f32((f64(x) / np.sqrt((f64(x) ** 2).mean(-1, keepdims=True) + 1e-6)) @ f64(w)),
         rtol=1e-3, atol=1e-3),
    dict(id="x18_mean_cols_default", code="torch.randn(4096, 32).mean(0)",
         ref=lambda x: _f32(f64(x).mean(0)), rtol=1e-4, atol=1e-4),
    dict(id="x19_batched_matmul", code="torch.randn(4, 64, 32) @ torch.randn(4, 32, 48)",
         ref=lambda a, b: _f32(f64(a) @ f64(b)), rtol=1e-3, atol=1e-3),
]

# ---- f16 buffers (computed in f32, rounded on store) ----------------------------------------
OPEN += [
    dict(id="h00_half_add", code="torch.randn(1024, dtype=torch.float16) + torch.randn(1024, dtype=torch.float16)",
         ref=lambda a, b: [(a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)], rtol=1e-3, atol=1e-3),
    dict(id="h01_half_rowsum", code="torch.randn(64, 256, dtype=torch.float16).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float16)], rtol=1e-2, atol=1e-2, knobs=REDUCE_KNOBS),
]
