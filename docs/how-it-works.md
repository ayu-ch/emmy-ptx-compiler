# How emmyptx works, by example

This walks through what the compiler does with one problem from each category, from the IR
to the PTX it prints. Every PTX excerpt is real output of `compile.py`.

## 0. What comes in

`problems/p00_scalar_add/k_add_pointwise.ir.json` holds a small graph. Its one `KernelOp`
has a body that `Graph.from_dict` turns into objects (`Tile`, `Load`, `Assign`, `Write`, ...):

```
Tile[a0] (N=64)
    in0__u0 = load x0[((a0 * 2) + 0)]
    in1__u0 = load x1[((a0 * 2) + 0)]
    f32 v0__u0 = add(in0__u0, in1__u0)
    in0__u1 = load x0[((a0 * 2) + 1)]
    in1__u1 = load x1[((a0 * 2) + 1)]
    f32 v0__u1 = add(in0__u1, in1__u1)
    add[((a0 * 2) + 0)] = (v0__u0, v0__u1)
```

Emmy has already made every scheduling decision: 64 threads, each handling 2 elements.
`meta.json` adds what the IR leaves out: the entry's parameters in order (`x0, x1, add`) and
the launch, 1 block of 256 threads.

`compile.py` loads both and calls `compile_kernel`, which runs `KernelCompiler` over the body.
The compiler is a single pass: `body()` visits statements in order and dispatches each to
`visit_<Kind>`.

## 1. Pointwise (p00): names, registers, addresses

```
    ld.param.u64 %rd1, [x0];                 // prologue: every pointer, in arg_order,
    cvta.to.global.u64 %rd1, %rd1;           // converted to a global address
    ...
    mov.u32 %r1, %tid.x;                     // tid, lane = tid & 31, warp = tid >> 5
    and.b32 %r2, %r1, 31;
    shr.u32 %r3, %r1, 5;
    mov.u32 %r4, %ctaid.x;                   // Tile: gid = ctaid * ntid + tid
    mov.u32 %r6, %ntid.x;
    mad.lo.s32 %r5, %r4, %r6, %r1;
    setp.ge.s32 %p1, %r5, 64;                // 256 threads launched, 64 cells: the guard
    @%p1 bra $L_exit;
    mul.lo.s32 %r7, %r5, 2;                  // a0 * 2 (the "+ 0" folded away)
    mul.wide.s32 %rd4, %r7, 4;               // element -> byte offset, once for all buffers
    add.s64 %rd5, %rd1, %rd4;
    ld.global.v2.f32 {%f1, %f2}, [%rd5];     // in0__u0 and in0__u1, merged into one load
    add.s64 %rd6, %rd2, %rd4;
    ld.global.v2.f32 {%f3, %f4}, [%rd6];
    add.f32 %f5, %f1, %f3;
    add.f32 %f6, %f2, %f4;
    add.s64 %rd7, %rd3, %rd4;
    st.global.v2.f32 [%rd7], {%f5, %f6};     // the grouped Write: one vector store
```

The ideas behind this:

- **`Val`** (`ptx.py`). Everything the compiler computes is a `Val`: a register such as
  `%f3`, or a constant such as `2` or `0f3F800000`. Constants stay constants, so `(a0 * 2) + 0`
  costs one multiply.
- **`self.env`** maps each IR name to its `Val`. A `Load` *defines* its names, and an
  `Assign` looks up its arguments and defines its result.
- **Fresh registers (SSA).** Each definition gets a new register; `ptxas` does the real
  allocation (12 hardware registers here). The exception is carried names (section 2).
- **The guard is not in the IR.** It comes from `N = 64` against 1 × 256 launched threads.
  When the launch matches N exactly (p02), no guard is emitted.
- **`flat_index` + `address`.** An index tuple is flattened row-major against the buffer's
  shape (`x[a0, a1]` becomes `a0 * 512 + a1`). Then `address` makes it an operand:
  `mul.wide` to a 64-bit byte offset, plus the buffer's base.
- **Paired loads.** `in0__u0` reads element `e`, and `in0__u1`, two statements later, reads
  `e + 1`. `body()` merges them into one `.v2` load, after proving the address is 8-byte
  aligned (`a0 * 2` is even, see section 3).

## 2. Reductions (r01): loops, carried state, combines

```
Tile[a0_co] (N=128)
    for a0 in a0_co..1024:128
        in0 = load x[a0]
        f32 acc0 <- add(acc0, in0)
    WarpShuffle(acc0, length=32)
    Smem float acc0_smem[4]
    if ((lane == 0)):
        acc0_smem[warp] = acc0
    Sync
    TreeHalve(acc0, length=4, tid=warp)
    if ((a0_co == 0)):
        sum_1[] = acc0
```

```
    mov.f32 %f1, 0f00000000;                 // acc0 = identity of add, before the loop
    mul.wide.s32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];               // the strided loop, unrolled: 8 copies,
    add.f32 %f1, %f1, %f2;                   // a0 = a0_co + 128k folded into [%rd4 + 512k]
    ld.global.f32 %f3, [%rd4+512];
    add.f32 %f1, %f1, %f3;
    ...
    shfl.sync.bfly.b32 %f10, %f1, 16, 31, 4294967295;   // WarpShuffle: offsets 16, 8, 4, 2, 1
    add.f32 %f11, %f1, %f10;                            // then combine_states: acc0 = acc0 + acc0__o
    mov.f32 %f1, %f11;
    ...
    setp.eq.s32 %p1, %r3, 0;                 // if (lane == 0)
    @!%p1 bra $L3_endif;
    mad.lo.s32 %r15, %r4, 4, %r1;            // &acc0_smem[warp] in shared memory
    st.shared.f32 [%r15], %f1;
$L3_endif:
    bar.sync 0;                              // Sync
    setp.ge.s32 %p2, %r4, 2;                 // TreeHalve, level 1: warps 0-1 add warp+2
    @%p2 bra $L4_tree_skip;
    ...
$L4_tree_skip:
    bar.sync 0;                              // every thread reaches the barrier
    ...
    ld.shared.f32 %f1, [%r1];                // broadcast: every thread reads the total
```

- **Carried names.** `acc0` is an `Accum` target, so it keeps one register (`%f1`) that every
  fold updates in place. The `Accum` sits inside the loop, but its seed (the op's identity: 0
  for add, -1e30 for max) is emitted before the loop that directly contains it.
- **The loop was unrolled completely**, although its start `a0_co` depends on the thread.
  `a0_co` is a tile axis, so its range is known: [0, 127]. Every thread therefore runs exactly
  8 iterations, and each copy's index is `a0_co + 128k`. As an affine value that's an
  immediate offset of `512k` bytes. (`loops.py`, `_short_loop`; if trip counts differed
  between threads, the later copies would get exit checks.)
- **`WarpShuffle` and `TreeHalve` don't know they are summing.** They move the state and then
  compile the IR's own `combine_states` program, which is ordinary `Assign`s. In r01 that's
  `acc0 = add(acc0, acc0__o)`. In f02 (softmax) it's eight assignments merging two
  (max, sum) pairs, and the same code handles it.
- **Barriers sit outside conditional code.** Threads that skip a tree level jump *to* the
  barrier, never past it.

## 3. Matmul (m01): shared memory, cp.async, affine addressing

m01 copies tiles of A and B into shared memory with `cp.async`, in a 3-stage pipeline, then
runs an inner `for _ki in 0..64` loop marked `unroll`:

```
    cp.async.cg.shared.global [%r28], [%rd7], 16;
    cp.async.cg.shared.global [%r28+8192], [%rd7+65536], 16;   // same registers, constant offsets
    ...
    cp.async.wait_group 2;
    bar.sync 0;
    ...
    ld.shared.v2.f32 {%f5, %f6}, [%r86];     // B[k, 2u] and B[k, 2u+1]: one paired load
    ...
    ld.shared.f32 %f7, [%r96];               // A[2v, k]
    fma.rn.f32 %f1, %f5, %f7, %f1;           // acc += a * b
    fma.rn.f32 %f2, %f6, %f7, %f2;
    ld.shared.f32 %f10, [%r96+256];          // A[2v+1, k]: the next row, a constant away
```

- **Dynamic shared memory.** The two tiles need 72 KiB, above the 48 KiB that static
  `.shared` arrays allow. `meta.json` then sets `smem_bytes`, and the arrays are carved out of
  one `.extern .shared` pool.
- **Affine values** (`intmath.py`). Every integer `+ c` or `* c` result is recorded as
  `base + constant`. In the 64 unrolled copies of the inner loop, `_ki` is a constant, so every
  shared-memory index is the *same* base plus a different constant. `address()` puts the
  constant into the immediate (`[%r98+256]`), and all copies share one address register. That
  took m01 from 68 µs to 21 µs.
- **CSE, scoped by block.** Identical integer instructions are reused from the same or an
  enclosing block, but never from a loop body or branch that might not have run.
- **Facts about values.** Ranges tell whether a value is non-negative (unsigned division,
  shifts). Divisibility proves alignment: `a1_u * 2` is even, so B's two columns can be one
  8-byte load.
- **FMA.** A `multiply` result is emitted lazily, and an `Accum` that adds it becomes one
  `fma.rn.f32`. The lazy multiply is dropped since nothing else reads it.

## 4. Fused and attention (f02, a01): combine programs and invariant loads

f02's online softmax keeps two accumulators, a running max `acc0` and a running sum `acc1`.
The sum is an `Accum` with a `base`: `acc1 <- add(sa, sb)` means `acc1 = sa + sb`, where `sa`
is the old sum rescaled by `exp(old max - new max)`. It's compiled exactly like any other
`Accum`. The cross-thread merge is the 8-statement `combine_states` program.

`exp` is not a PTX instruction. `ops.py` computes `ex2.approx.f32(x * log2(e))`, and builds
tanh, erf (for GELU), sigmoid, log and pow out of the few math instructions PTX has.

a01 computes, for each key `a1` in a runtime loop, four dot products `q[row, :] . kt[:, a1]`.
The q values don't depend on `a1`, and nothing writes q, so `_hoist_invariant_loads` issues
them once before the loop and the loop body reuses the registers. That made a01 1.24× faster
than nvcc's build of Emmy's CUDA.

## 5. Layout of loops

`loops.py` picks one of three forms for every loop:

| Form | When | Shape |
|---|---|---|
| complete unrolling | known trip count, small body (or the IR marks it `unroll`) | straight-line copies; the variable is a constant |
| guarded unrolling | trip count depends on the thread but is known to be small | copies, an exit check before each copy that might not run |
| runtime loop | everything else | bottom-tested; a plain body unrolled to about 16 loads in flight per thread, then a remainder loop |

The runtime form matters for reductions over big arrays (r02, 512 iterations per thread): a
single load per iteration leaves the thread waiting on memory. With 16 independent loads per
trip, r02 went from 23 µs to 10 µs.

If unrolling makes the PTX bigger than the runner accepts (2 MB), `compile_kernel` compiles
again at a lower optimization level.

## 6. Extending it

- **A new elementwise op**: add a branch to `ops.lower_op`, writing the result into `dst`.
- **A new statement**: add `visit_<Kind>` to `KernelCompiler`. If it can contain code, make
  `walk` see its bodies (`loops.children`) and wrap code that may not run in `self.block()`.
- **Testing**: add a torch expression to `tools/expressions_extra.py`, generate with the
  organizers' `tools/generate.py`, and run `tools/run_local.py` or `tools/run_all.py`.
