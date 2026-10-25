"""Print the PTX for a kernel with each IR statement as a comment above the code it produced.

    python tools/trace.py path/to/<kernel>.ir.json [--level 0|1|2]

A reading aid: the compiler is unchanged, it just leaves `// ...` markers as it visits each
statement. Statements that produce no code of their own (a constant `Let`, an `Smem`
declaration) show up as a marker with nothing under it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmy.compiler.graph import Graph  # noqa: E402

from compile import find_launch  # noqa: E402
from emmyptx.kernel import KernelCompiler  # noqa: E402


def _head(s) -> str:
    text = s.pretty()[0].strip() if hasattr(s, "pretty") else type(s).__name__
    return text.encode("ascii", "replace").decode()              # PTX comments must be ASCII


class TracingCompiler(KernelCompiler):
    depth = 0

    def statement_in(self, stmts, i):
        self.em.lines.append(f"    // {'  ' * self.depth}>> {_head(stmts[i])}")
        self.depth += 1
        try:
            absorbed = super().statement_in(stmts, i)
        finally:
            self.depth -= 1
        for j in sorted(absorbed):
            self.em.lines.append(f"    // {'  ' * self.depth}>> {_head(stmts[j])}   (merged into the load above)")
        return absorbed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ir", type=Path)
    ap.add_argument("--level", type=int, default=2, help="optimization level (default 2)")
    ap.add_argument("--target", default="sm_89")
    args = ap.parse_args()
    graph = Graph.from_dict(json.loads(args.ir.read_text()))
    kernel = next(n.op for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    meta_path = args.ir.parent / "meta.json"
    launch = find_launch(json.loads(meta_path.read_text()) if meta_path.exists() else None, kernel, graph)
    print(TracingCompiler(graph, kernel, launch, args.target, level=args.level).compile())
    return 0


if __name__ == "__main__":
    sys.exit(main())
