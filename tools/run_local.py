"""Run compile.py on every problem on a local NVIDIA GPU, with no network: a stand-in for the
organizers' runner, using the CUDA driver API through ctypes.

    python tools/run_local.py <problems dir> [--only PREFIX]

Inputs are made the way the runner makes them (a seeded `standard_normal` per `seed`, exact
values for `data`, zeroed outputs and `zero_outputs`), each kernel runs under its meta.json
launch, and outputs are compared with expected.npz within the problem's rtol / atol. The PTX
target is the local GPU's (e.g. sm_86 on an RTX 30xx), passed to compile.py as --target.
"""

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
cuda = ctypes.CDLL("libcuda.so.1")
# Pointers and sizes are passed as c_void_p / c_size_t explicitly: by default ctypes would pass
# Python ints as 32-bit C ints, truncating the host addresses of large numpy arrays.
DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64}
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 75, 76


def check(err: int, what: str) -> None:
    if err:
        msg = ctypes.c_char_p()
        cuda.cuGetErrorString(err, ctypes.byref(msg))
        raise RuntimeError(f"{what}: {msg.value.decode() if msg.value else err}")


def init() -> str:
    """Create a context on GPU 0 and return its PTX target, e.g. 'sm_86'."""
    check(cuda.cuInit(0), "cuInit")
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "context")
    check(cuda.cuCtxSetCurrent(ctx), "context")
    major, minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    cuda.cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    return f"sm_{major.value}{minor.value}"


def load(ptx: str, entry: str):
    log = ctypes.create_string_buffer(16384)
    opts = (ctypes.c_int * 2)(5, 6)                  # CU_JIT_ERROR_LOG_BUFFER, ..._SIZE_BYTES
    vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p).value, 16384)
    module, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if cuda.cuModuleLoadDataEx(ctypes.byref(module), ptx.encode() + b"\0", 2, opts, vals):
        raise RuntimeError("PTX rejected:\n" + log.value.decode())
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), module, entry.encode()), f"entry {entry}")
    return module, fn


def run_kernel(ptx: str, k: dict, produced: dict) -> dict:
    """Launch one kernel of a problem; return its outputs by name."""
    module, fn = load(ptx, k["entry"])
    if k["smem_bytes"] > 48 * 1024:
        check(cuda.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k["smem_bytes"]), "smem")
    arrays, ptrs, holders = {}, [], []
    for a in k["args"]:
        dtype, shape = DTYPES[a["dtype"]], a["shape"] or [1]
        if a.get("scalar"):
            holders.append(ctypes.c_float(a["value"]) if a["dtype"] == "f32" else ctypes.c_int32(a["value"]))
        else:
            if "seed" in a:
                host = np.random.default_rng(a["seed"]).standard_normal(shape).astype(dtype)
            elif "data" in a:
                host = np.asarray(a["data"], dtype=dtype).reshape(shape)
            elif a.get("intermediate"):
                host = np.asarray(produced[a["name"]], dtype=dtype).reshape(shape)
            else:
                host = np.zeros(shape, dtype)
            dev = ctypes.c_uint64()
            check(cuda.cuMemAlloc_v2(ctypes.byref(dev), ctypes.c_size_t(max(host.nbytes, 1))), "alloc")
            check(cuda.cuMemcpyHtoD_v2(dev, ctypes.c_void_p(host.ctypes.data), ctypes.c_size_t(host.nbytes)), "copy in")
            arrays[a["name"]] = (dev, host)
            holders.append(dev)
        ptrs.append(ctypes.cast(ctypes.byref(holders[-1]), ctypes.c_void_p))
    params = (ctypes.c_void_p * len(ptrs))(*ptrs)
    check(cuda.cuLaunchKernel(fn, *k["grid"], *k["block"], k["smem_bytes"], None, params, None), "launch")
    check(cuda.cuCtxSynchronize(), "kernel")
    outputs = {}
    for a in k["args"]:
        if a.get("output"):
            dev, host = arrays[a["name"]]
            out = np.empty_like(host)
            check(cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(out.ctypes.data), dev, ctypes.c_size_t(out.nbytes)), "copy out")
            outputs[a["name"]] = out
    for dev, _ in arrays.values():
        cuda.cuMemFree_v2(dev)
    cuda.cuModuleUnload(module)
    return outputs


def run_problem(pdir: Path, target: str) -> tuple[bool, str]:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced, report = {}, []
    for k in meta["kernels"]:
        ir = pdir / f"{k['entry']}.ir.json"
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir), "--target", target],
                           capture_output=True, text=True)
        if r.returncode:
            return False, f"compile.py failed: {(r.stderr.strip().splitlines() or ['?'])[-1]}"
        try:
            outs = run_kernel(r.stdout, k, produced)
        except RuntimeError as e:
            return False, str(e).splitlines()[0]
        for name, got in outs.items():
            produced[name] = got
            if name in expected:
                ref = expected[name]
                err = float(np.max(np.abs(got.reshape(ref.shape) - ref))) if ref.size else 0.0
                if not np.allclose(got.reshape(ref.shape), ref, rtol=meta["rtol"], atol=meta["atol"]):
                    return False, f"{name}: max abs err {err:.3e} (rtol={meta['rtol']} atol={meta['atol']})"
                report.append(f"{name}: max abs err {err:.3e}")
    return True, "; ".join(report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", type=Path, help="a problems folder")
    ap.add_argument("--only", default="", help="only problems whose name starts with this")
    args = ap.parse_args()
    target = init()
    problems = sorted(p for p in args.problems.iterdir() if (p / "meta.json").exists() and p.name.startswith(args.only))
    passed = 0
    for p in problems:
        ok, detail = run_problem(p, target)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {p.name:28s} {detail}", flush=True)
    print(f"\n{passed}/{len(problems)} passed on the local GPU ({target})")
    return 0 if passed == len(problems) else 1


if __name__ == "__main__":
    sys.exit(main())
