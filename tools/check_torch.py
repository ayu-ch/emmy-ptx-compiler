"""Check emmyptx end to end against PyTorch itself, not only against Emmy's CUDA.

    python tools/check_torch.py                       # every category in tools/torch_cases/
    python tools/check_torch.py pointwise fused       # some categories
    python tools/check_torch.py --only m_mm,f_soft    # case ids, or id prefixes
    python tools/check_torch.py my_cases.py           # a cases file of your own

Each case is torch code `<preamble>; <final expression>` (see tools/torch_cases/__init__.py).
It goes through every stage on the local GPU, and a failure is reported with the stage that
failed, so it lands on the right component:

  trace      Emmy traces the final expression and schedules it (`emmy compile --ir cuda`)
  package    the dump becomes <entry>.ir.json + meta.json, the files compile.py reads
  reference  the case's own torch code re-run in float64 on the inputs the GPU gets
  compile    compile.py (emmyptx) -> PTX
  run        the PTX loaded by the CUDA driver and launched with meta.json's geometry
  check      every output compared with the reference within the case's rtol / atol

A test against Emmy's `.ref.cu` cannot see a wrong answer that is already in Emmy's IR (a lost
slice step, a permute lowered as a reshape): emmyptx compiles that IR faithfully, and both
agree. The float64 torch reference does see it. To tell whose bug a `check` failure is, read
the case's <entry>.ir.txt and <entry>.ref.cu in --out.

Every case runs on the GPU in its own process, because a kernel fault such as a misaligned
address is sticky: it would fail every later launch in the same CUDA context. Results are in
<out>/results.json, and each case's IR, PTX and meta.json in <out>/<id>/.
Needs torch and emmy-ml in this Python; the package step reads emmy-ml 0.3.11's dump layout.
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import torch_cases  # noqa: E402

DTYPES = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i32": np.int32, "i64": np.int64}

# `emmy` in this Python. Emmy imports fcntl, which Windows lacks, only to flock its
# working-golden cache; every trace here gets its own cache files, so a no-op lock is enough.
EMMY = """import sys, types
try:
    import fcntl
except ImportError:
    sys.modules["fcntl"] = types.SimpleNamespace(LOCK_SH=1, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=lambda fd, op: None)
from emmy.emmy import main
sys.argv[0] = "emmy"
main()
"""


class StageError(Exception):
    def __init__(self, stage: str, msg: str):
        super().__init__(msg)
        self.stage = stage


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "?"


def load_cases(names: list[str]) -> list[dict]:
    """The cases of the named categories (all of them by default), or of cases files by path."""
    names = names or sorted(m.name for m in pkgutil.iter_modules(torch_cases.__path__))
    cases = []
    for name in names:
        path = Path(name)
        if path.suffix == ".py":
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(f"torch_cases.{name}")
        cases += [dict(c, category=path.stem) for c in mod.CASES]
    return cases


# ---- trace -------------------------------------------------------------------------------------
def trace(case: dict, work: Path, target: str) -> Path:
    dump = work / "dump"
    env = {**os.environ,
           "PYTHONUTF8": "1",                                 # Emmy's dumps are not all ASCII
           "EMMY_DUMP_DIR": str(dump),
           "EMMY_TUNE_DB": str(work / "autotune.db"),         # no tuning evidence from earlier runs
           "EMMY_ONLINE_FILE": str(work / "online.json"),
           "EMMY_GOLDEN_FILE": "",
           "EMMY_KNOBS": case.get("knobs", "")}
    r = subprocess.run([sys.executable, "-c", EMMY, "compile", "-c", case["code"], "--ir", "cuda", "--target", target],
                       cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900)
    if r.returncode:
        raise StageError("trace", last_line(r.stderr))
    return dump


# ---- package -----------------------------------------------------------------------------------
def _flat(dims) -> int:
    if isinstance(dims, int):
        return dims
    if not all(isinstance(f, (int, float)) for f in dims):
        raise StageError("package", f"symbolic launch dim {dims!r}")
    return math.prod(int(f) for f in dims)


def package(dump: Path, pdir: Path) -> dict:
    """Emmy's dump -> one <entry>.ir.json (plus .ir.txt and .ref.cu) per kernel, and the
    launch descriptors, in the format of the organizers' problem folders."""
    try:
        kernel_json = next(dump.glob("*_lowering_kernel.json"))
        cuda_json = next(dump.glob("*_lowering_cuda.json"))
    except StopIteration:
        raise StageError("package", "no kernel or cuda stage in Emmy's dump") from None
    graph = json.loads(kernel_json.read_text(encoding="utf-8"))
    cuda = json.loads(cuda_json.read_text(encoding="utf-8"))
    kdir = kernel_json.with_suffix(".kernels")
    nodes = graph["nodes"]
    tensors = {t["name"]: t for n in nodes.values() for t in n.get("outputs", [])}
    kernels = []
    for nid, node in cuda["nodes"].items():
        if node["op"] != "CudaOp":
            continue
        of = node["op_fields"]
        outs = {t["name"] for t in node["outputs"]}
        args = []
        for name in of["arg_order"]:
            t = tensors.get(name)
            if t is None:
                raise StageError("package", f"argument {name!r} is not in the graph")
            a = dict(name=name, dtype=t["dtype"], shape=t["shape"] or [1])
            n = nodes.get(name)
            if name in outs:
                a["output"] = True
            elif n is not None and n["op"] == "ConstantOp":
                f = n["op_fields"]
                if f.get("value") is not None:              # a value folded into the graph
                    a["data"] = np.asarray(f["value"], dtype=np.float64).ravel().tolist()
                else:                                       # a module parameter, by name: its
                    a["source_path"] = f["source_path"]     # values come from `reference`
                    a["load_ops"] = f.get("load_ops") or []
            elif n is not None and n["op"] == "InputOp":
                a["seed"] = graph["inputs"].index(name)
            else:
                a["intermediate"] = True
            args.append(a)
        smem = of.get("smem_bytes", 0)
        k = dict(node=nid, entry=of["kernel_name"], arg_order=of["arg_order"],
                 grid=[_flat(d) for d in of["grid"]], block=[_flat(d) for d in of["block"]],
                 smem_bytes=smem if smem > 48 * 1024 else 0,       # only dynamic shared memory
                 zero_outputs=of.get("zero_outputs", []), args=args)
        src = kdir / f"{nid}.json"
        if not src.exists():
            src = kdir / f"{k['entry']}.json"
        ir = json.loads(src.read_text(encoding="utf-8"))
        kops = [n for n in ir["nodes"].values() if n["op"] == "KernelOp"]
        if len(kops) != 1:
            raise StageError("package", f"{len(kops)} KernelOps for {k['entry']}")
        kops[0]["op_fields"]["name"] = k["entry"]                  # what compile.py matches meta.json by
        (pdir / f"{k['entry']}.ir.json").write_text(json.dumps(ir), encoding="utf-8")
        if src.with_suffix(".txt").exists():
            shutil.copy(src.with_suffix(".txt"), pdir / f"{k['entry']}.ir.txt")
        (pdir / f"{k['entry']}.ref.cu").write_text(of.get("kernel_source", ""), encoding="utf-8")
        kernels.append(k)
    if not kernels:
        raise StageError("package", "no CudaOp in Emmy's dump")
    return dict(graph_inputs=graph["inputs"], graph_outputs=graph["outputs"], kernels=kernels,
                input_dtypes={n: nodes[n]["outputs"][0]["dtype"] for n in graph["inputs"]},
                input_shapes={n: nodes[n]["outputs"][0]["shape"] for n in graph["inputs"]})


# ---- reference ---------------------------------------------------------------------------------
def seeded_input(seed: int, shape: list, dtype: str):
    """Graph input number `seed` as run_local.py makes it, as a torch tensor."""
    import torch
    if dtype == "bf16":                                 # numpy has no bf16: round f32 values
        return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32)).bfloat16()
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(DTYPES[dtype]))


def reference(case: dict, pkg: dict) -> dict[str, np.ndarray]:
    """Run the case's torch code in float64 on the seeded inputs; module parameters that the
    graph names by path are copied into the kernels' arguments."""
    import torch
    import torch.nn.functional as F
    tree = ast.parse(case["code"], mode="exec")
    scope = {"torch": torch, "nn": torch.nn, "F": F}
    exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<case>", "exec"), scope)
    modules = [v for v in scope.values() if isinstance(v, torch.nn.Module)]
    for k in pkg["kernels"]:
        for a in k["args"]:
            if "source_path" not in a:
                continue
            if len(modules) != 1:
                raise StageError("reference", f"parameter {a['source_path']!r} needs exactly one nn.Module "
                                              "in the preamble")
            t = modules[0].state_dict()[a["source_path"]].detach()
            for op in a["load_ops"]:
                if op["__op__"] != "TransposeOp":
                    raise StageError("reference", f"load op {op['__op__']} on {a['source_path']!r}")
                t = t.transpose(*op["fields"]["axes"])
            if list(t.shape) != list(a["shape"]):
                raise StageError("reference", f"{a['source_path']}: shape {list(t.shape)}, the graph has {a['shape']}")
            a["data"] = t.contiguous().reshape(-1).double().tolist()
    for module in modules:
        module.double()                                 # the same weights, evaluated in f64
    # `module(x)` is traced through module.forward, whose parameter names (e.g. `input`) name
    # the graph inputs: those are matched to the call's arguments by position.
    final = tree.body[-1].value
    call_args = [a.id for a in final.args if isinstance(a, ast.Name)] if isinstance(final, ast.Call) else []
    for i, name in enumerate(pkg["graph_inputs"]):
        var = name if name in scope else (call_args[i] if i < len(call_args) else None)
        if var is None or not isinstance(scope.get(var), torch.Tensor):
            raise StageError("reference", f"graph input {name!r} is not a tensor variable of the preamble")
        t = seeded_input(i, pkg["input_shapes"][name] or [1], pkg["input_dtypes"][name])
        t = t.double() if t.is_floating_point() else t.long()
        scope[var] = t.reshape(scope[var].shape)
    with torch.no_grad():
        out = eval(compile(ast.Expression(final), "<case>", "eval"), scope)
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    if len(outs) != len(pkg["graph_outputs"]):
        raise StageError("reference", f"torch gives {len(outs)} outputs, the graph has {len(pkg['graph_outputs'])}")
    return {name: o.detach().numpy() for name, o in zip(pkg["graph_outputs"], outs)}


def prepare(case: dict, pdir: Path, target: str) -> dict:
    """trace + package + reference for one case. The outcome is saved for --skip-trace."""
    shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            pkg = package(trace(case, Path(td), target), pdir)
        np.savez(pdir / "expected.npz", **reference(case, pkg))
        meta = dict(id=case["id"], code=case["code"], rtol=case["rtol"], atol=case["atol"], **pkg)
        (pdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        res = dict(stage="prepared")
    except StageError as e:
        res = dict(stage=e.stage, detail=str(e))
    except Exception as e:  # noqa: BLE001 - one broken case must not stop the others
        res = dict(stage="harness", detail=f"{type(e).__name__}: {e}"[:300])
    (pdir / "prepare.json").write_text(json.dumps(res), encoding="utf-8")
    return res


# ---- compile, run, check -----------------------------------------------------------------------
def compare(name: str, got: np.ndarray, ref: np.ndarray, rtol: float, atol: float) -> dict:
    if got.size != ref.size:
        return dict(name=name, ok=False,
                    detail=f"{name}: the kernel wrote {got.size} values, torch gives shape {list(ref.shape)}")
    g, r = got.reshape(ref.shape).astype(np.float64), ref.astype(np.float64)
    close = np.isclose(g, r, rtol=rtol, atol=atol, equal_nan=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where((g == r) | (np.isnan(g) & np.isnan(r)), 0.0, np.abs(g - r))
        err = np.nan_to_num(err, nan=np.inf)
        ratio = float(np.max(np.where(err == 0, 0.0, err / (atol + rtol * np.abs(r))), initial=0.0))
    # A kernel that writes nothing leaves the zeroed output buffer: not a pass.
    vacuous = bool(np.all(g == 0)) and not bool(np.all(r == 0))
    off = int(np.count_nonzero(~close))
    detail = f"{name}: max|err| {float(np.max(err, initial=0.0)):.2e} ({ratio:.3g}x tol)"
    if off:
        detail += f", {off} of {ref.size} values off"
    if vacuous:
        detail += ", ALL ZERO"
    return dict(name=name, ok=not off and not vacuous, detail=detail)


def run_case(pdir: Path, target: str) -> dict:
    """compile, run and check one prepared case, in this process."""
    import run_local
    run_local.init()
    meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
    expected = dict(np.load(pdir / "expected.npz"))
    produced, checks = {}, []
    for k in meta["kernels"]:
        r = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(pdir / f"{k['entry']}.ir.json"),
                            "--target", target], capture_output=True, text=True)
        if r.returncode:
            return dict(stage="compile", detail=f"{k['entry']}: {last_line(r.stderr)}")
        (pdir / f"{k['entry']}.ptx").write_text(r.stdout, encoding="utf-8")
        try:
            outs = run_local.run_kernel(r.stdout, k, produced)
        except RuntimeError as e:
            return dict(stage="run", detail=f"{k['entry']}: {str(e).splitlines()[0]}")
        for name, got in outs.items():
            produced[name] = got
            if name in expected:
                checks.append(compare(name, got, expected[name], meta["rtol"], meta["atol"]))
    checks += [dict(name=n, ok=False, detail=f"{n}: no kernel writes it") for n in expected if n not in produced]
    ok = all(c["ok"] for c in checks)
    return dict(stage="pass" if ok else "check", detail="; ".join(c["detail"] for c in checks))


def run_isolated(pdir: Path, target: str) -> dict:
    r = subprocess.run([sys.executable, __file__, "--run-one", str(pdir), "--target", target],
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        return dict(stage="harness", detail=last_line(r.stderr))
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="categories (tools/torch_cases/<name>.py) or cases files; default: all")
    ap.add_argument("--only", default="", help="comma-separated case ids or id prefixes")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "torch", help="problem folders and results.json")
    ap.add_argument("--target", help="GPU to schedule and compile for (default: the local GPU's, e.g. sm_86)")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="cases traced at once")
    ap.add_argument("--skip-trace", action="store_true", help="reuse the cases already prepared in --out")
    ap.add_argument("--run-one", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.run_one:
        print(json.dumps(run_case(args.run_one, args.target)))
        return 0

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if any(c["id"].startswith(p) for p in args.only.split(","))]
    if args.target is None:
        import run_local
        args.target = run_local.init()
    args.out.mkdir(parents=True, exist_ok=True)

    results = {c["id"]: dict(id=c["id"], category=c["category"], code=c["code"]) for c in cases}
    if args.skip_trace:
        for c in cases:
            saved = args.out / c["id"] / "prepare.json"
            results[c["id"]].update(json.loads(saved.read_text(encoding="utf-8")) if saved.exists()
                                    else dict(stage="trace", detail="(not traced yet)"))
    else:
        print(f"tracing {len(cases)} cases for {args.target}, {args.jobs} at a time")
        with ThreadPoolExecutor(args.jobs) as pool:
            for c, res in zip(cases, pool.map(lambda c: prepare(c, args.out / c["id"], args.target), cases)):
                results[c["id"]].update(res)
                print(f"  {c['id']:28s} {res['stage']} {res.get('detail', '')[:120]}", flush=True)

    print(f"running on {args.target}, one process per case\n")
    for c in cases:
        res = results[c["id"]]
        if res["stage"] == "prepared":
            res.update(run_isolated(args.out / c["id"], args.target))
        status = "PASS" if res["stage"] == "pass" else "FAIL"
        print(f"{status}  {c['id']:28s} [{res['stage']}] {res.get('detail', '')[:160]}", flush=True)
    (args.out / "results.json").write_text(json.dumps(list(results.values()), indent=1), encoding="utf-8")

    stages = {}
    for res in results.values():
        stages[res["stage"]] = stages.get(res["stage"], 0) + 1
    passed = stages.pop("pass", 0)
    failed = ", ".join(f"{n} at {s}" for s, n in sorted(stages.items(), key=lambda kv: -kv[1]))
    print(f"\n{passed}/{len(results)} passed against PyTorch on {args.target}" + (f" ({failed})" if failed else ""))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
