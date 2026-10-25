"""Compile one Emmy Kernel IR file to PTX.

    python compile.py path/to/<kernel>.ir.json > out.ptx
    python compile.py path/to/<kernel>.ir.json --target sm_86     # e.g. to run on a local RTX 30xx

The launch descriptor `meta.json` next to the IR supplies the parameter order (`arg_order`),
the launch geometry and the dynamic shared-memory size. Emmy is used only to load the IR.
"""

import argparse
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emmyptx.kernel import compile_kernel  # noqa: E402


def find_launch(meta: dict | None, kernel, graph) -> dict:
    """This kernel's entry in meta.json, or a best guess from the IR when there is none."""
    if meta is not None:
        for k in meta.get("kernels", []):
            if k.get("entry") == kernel.name:
                return k
        if len(meta.get("kernels", [])) == 1:
            return meta["kernels"][0]
    node = next(n for n in graph.nodes.values() if n.op is kernel)
    outputs = [t.name for t in node.outputs]
    return {"arg_order": [*node.inputs, *outputs], "block": [256, 1, 1], "grid": [0, 1, 1], "smem_bytes": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir", help="a Kernel IR json file (<kernel>.ir.json)")
    ap.add_argument("--target", default="sm_89", help="PTX target architecture (default: sm_89)")
    args = ap.parse_args()

    ir_path = Path(args.ir)
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    kernels = [n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        sys.exit(f"expected one KernelOp in {ir_path}, found {len(kernels)}")
    kernel = kernels[0]
    meta_path = ir_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    launch = find_launch(meta, kernel, graph)

    sys.stdout.write(compile_kernel(graph, kernel, launch, target=args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
