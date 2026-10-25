# emmyptx: Emmy Kernel IR → PTX

A compiler from [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR straight to PTX, written
for the Emmy Hackathon 2026. Emmy normally renders its Kernel IR to CUDA C++ and builds it with
`nvcc`. This compiler skips the C++ step: it walks the IR and prints PTX, which `ptxas` assembles.

```bash
pip install emmy-ml==0.3.11            # used only to load the IR (Graph.from_dict)
python compile.py path/to/<kernel>.ir.json > out.ptx
python compile.py path/to/<kernel>.ir.json --target sm_86     # other GPUs, e.g. for local runs
```

`meta.json` next to the IR supplies the parameter order (`arg_order`), the launch geometry
and the dynamic shared-memory size. The lowering is written from scratch: nothing calls
Emmy's renderer, `emmy compile`, `nvcc` or any other compiler.

## Results

| | |
|---|---|
| Open problem set | **15 / 15** pass on the runner (RTX 5090, sm_120) and on an RTX 3060 (sm_86) |
| Generalization set | **51 / 51** pass on both GPUs: [`tools/expressions_extra.py`](tools/expressions_extra.py), the five categories with other shapes and ops (GELU, SiLU, tanh, layer norm, column sums, batched / chained / fused matmuls, attention up to 512 keys, f16 buffers) |
| Kernel speed | level with `nvcc` compiling Emmy's own CUDA for the same IR: geometric mean 0.99× on the open set (0.78× to 1.24×) and 1.04× on the generalization set (0.50× to 3.48×) |

Speed was measured on the runner (10 warm-up and 50 timed launches, median), against PTX that
`nvcc -O3` produced from each problem's `.ref.cu`, purely as a benchmark:

| Problem | emmyptx | nvcc from Emmy's CUDA | | Problem | emmyptx | nvcc | |
|---|---|---|---|---|---|---|---|
| p00 scalar add | 3.10 µs | 3.10 µs | 1.00× | f01 RMSNorm | 4.74 µs | 4.77 µs | 1.01× |
| p01 vector add | 3.09 µs | 3.10 µs | 1.01× | f02 softmax | 3.04 µs | 3.07 µs | 1.01× |
| p02 scale-shift | 3.17 µs | 3.17 µs | 1.00× | f03 L2 norm | 3.10 µs | 3.04 µs | 0.98× |
| p03 ReLU-mul | 3.10 µs | 3.10 µs | 1.00× | m01 matmul 512 | 15.0 µs | 15.0 µs | 1.00× |
| r01 array sum | 3.10 µs | 3.07 µs | 0.99× | m02 matmul rect | 5.06 µs | 5.04 µs | 1.00× |
| r02 big sum | 10.1 µs | 8.96 µs | 0.89× | a01 attention | 51.8 µs | 64.2 µs | 1.24× |
| r03 mean | 6.75 µs | 6.05 µs | 0.90× | a02 attention 128 | 37.5 µs | 29.3 µs | 0.78× |
| r04 row sum | 3.17 µs | 3.26 µs | 1.03× | | | | |

## How it works

```
.ir.json --Graph.from_dict--> KernelOp --KernelCompiler--> PTX text --ptxas--> SASS
```

`KernelCompiler` visits the IR once, top to bottom, with one `visit_<Kind>` method per
statement. Every IR name maps to a `Val`: a virtual register, or a compile-time constant
that is folded into later arithmetic.

| Module | Role |
|---|---|
| [`compile.py`](compile.py) | command line: load the IR and `meta.json`, compile, print |
| [`emmyptx/kernel.py`](emmyptx/kernel.py) | the compiler: prologue, names, expressions, addresses, and every statement except loops |
| [`emmyptx/loops.py`](emmyptx/loops.py) | `Loop` / `StridedLoop`: how each loop is laid out (unrolled, guarded, rotated, hoisted) |
| [`emmyptx/intmath.py`](emmyptx/intmath.py) | integer index arithmetic: folding, affine splitting, CSE, value ranges, divisibility |
| [`emmyptx/ops.py`](emmyptx/ops.py) | elementwise ops: exp, log, tanh, erf, sigmoid, pow, comparisons, ... built from PTX instructions |
| [`emmyptx/ptx.py`](emmyptx/ptx.py) | values, registers, labels, and the instruction stream (with lazily kept instructions) |

How each Kernel IR statement is lowered:

| IR | PTX |
|---|---|
| `Tile` | `gid = ctaid * ntid + tid`; an exit guard when the launch has more threads than cells (from meta.json; not in the IR); axis decode `(gid / stride) % extent`, with Emmy's grouped raster order when set |
| `Load` / `Write` | flattened index → `ld` / `st` in `.global` or `.shared`; `.v2` / `.v4` for grouped names; `red.global.add.f32` for atomic writes; f16 buffers converted to and from f32 |
| `Assign` | the op table in `ops.py` |
| `Accum` | `acc = op(base, value)` on the accumulator's register, seeded with the op's identity before the loop that contains it; `acc + a * b` becomes one `fma.rn.f32` |
| `Loop` / `StridedLoop` | labels and a bottom-tested branch, or compile-time unrolling (see below) |
| `Cond`, `Select`, `TernaryExpr` | predicated branches, `selp` |
| `Smem` | `.shared` arrays, or one `.extern .shared` pool when `meta.json` asks for dynamic shared memory (over 48 KiB) |
| `Sync` | `bar.sync` (and named barriers, `bar.warp.sync`) |
| `WarpShuffle` | `shfl.sync.bfly.b32` per state component and offset, then the IR's own `combine_states` program; that one mechanism covers sums, maxes and online softmax's coupled (max, sum) merge |
| `TreeHalve` | a halving tree over the shared slabs with a barrier per level, then a broadcast read |
| `CpAsyncCopy` / `Commit` / `Wait` | `cp.async.cg/ca.shared.global`, `cp.async.commit_group`, `cp.async.wait_group` |

### Registers

Every value gets a fresh virtual register (SSA), so `ptxas` sees independent values as
independent. The exception is *carried* names (`Accum` and `Init` targets, `WarpShuffle` /
`TreeHalve` states): their value flows across loop iterations and combine steps, so they keep
one register that is updated in place. Declarations are counted at the end, and `ptxas` does
the real register allocation.

### Optimizations

Correctness comes from the straightforward lowering above. These make the code fast:

- **Folding and affine addressing** (`intmath.py`). Index arithmetic is folded at compile time,
  and every `+ c` / `* c` result is remembered as `base + constant`. An address becomes
  `[addr(base) + 4c]`, so the 64 copies of an unrolled matmul inner loop share one address
  register. The full value is emitted lazily and disappears when nothing else reads it.
- **Scoped common-subexpression elimination**: integer instructions are reused from enclosing
  blocks, never from a loop body or branch that may not have run.
- **Value ranges and divisibility**: non-negative values divide unsigned (shifts and masks for
  powers of two); a cooperative loop `for i in tid..N:step` gets its exact trip count from the
  range of `tid`; even offsets prove the alignment that merging two scalar loads into one `.v2`
  load needs.
- **Loop layout** (`loops.py`). Small constant loops unroll completely. Thread-dependent loops
  with a small trip count unroll with exit checks. Other loops are bottom-tested and unrolled
  to keep about 16 loads in flight per thread. Loop-invariant global loads are hoisted out of
  runtime loops (attention keeps its query row in registers).
- **FMA contraction** of accumulations, and **lazy instructions** that are dropped when unused.
- **Size control**: the runner accepts PTX up to 2 MB, so a kernel whose unrolled PTX would
  be larger is recompiled at a lower optimization level.

## Testing

```bash
# every problem through the organizers' runner (needs their repo for submit.py)
python tools/run_all.py ../emmy-hackathon-2026
python tools/run_all.py ../emmy-hackathon-2026 out/extra

# the same checks on a local NVIDIA GPU, no network: same input generation, same tolerances
python tools/run_local.py ../emmy-hackathon-2026/problems

# regenerate the generalization set (needs torch) with the organizers' generator
python ../emmy-hackathon-2026/tools/generate.py --emmy "$(which emmy)" --out out/extra tools/expressions_extra.py
```

## Limitations

- Computation is in f32 throughout: f16 buffers are converted on load and rounded once on
  store. (Emmy's own CUDA computes f16 values in `__half` arithmetic; f32 is at least as
  accurate, and passes the f16 tests.) No tensor cores, TMA or `wgmma`, none of which the
  hackathon's problems use.
- The statements Emmy documents as unused by the hackathon (`Pre`, `Carry`, `ZeroPrologue`,
  the tensor-core and TMA family) are not supported, and the compiler stops with a clear error.
  `Select`, documented as unused, appeared in a generated matmul-plus-reduction and is supported.
- Tiles above 2^31 cells (which would need 64-bit indexing) are rejected.
