"""Run compile.py on every problem in a problems folder through the organizers' submit.py.

    python tools/run_all.py <emmy-hackathon-2026 checkout> [problems dir] [--only PREFIX]

Prints one line per problem (PASS / FAIL with the reason) and a summary. Problems run in
parallel, a few at a time, since each one is a round trip to the remote runner.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(submit: Path, problem: Path) -> tuple[str, bool, str]:
    r = subprocess.run([sys.executable, str(submit), str(problem), "--compiler", str(ROOT / "compile.py")],
                       capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and "PASS" in out and "FAIL" not in out
    detail = [line for line in out.splitlines() if "PASS" in line or "FAIL" in line or "rror" in line]
    return problem.name, ok, (detail[-1] if detail else out.splitlines()[-1] if out else "(no output)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hackathon", type=Path, help="the emmy-hackathon-2026 checkout (for submit.py)")
    ap.add_argument("problems", type=Path, nargs="?", help="problems folder (default: the checkout's)")
    ap.add_argument("--only", default="", help="only problems whose name starts with this")
    ap.add_argument("-j", type=int, default=4, help="problems in flight at once")
    args = ap.parse_args()
    pdir = args.problems or args.hackathon / "problems"
    problems = sorted(p for p in pdir.iterdir() if (p / "meta.json").exists() and p.name.startswith(args.only))
    with ThreadPoolExecutor(args.j) as pool:
        results = list(pool.map(lambda p: run(args.hackathon / "submit.py", p), problems))
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
