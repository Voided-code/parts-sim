#!/usr/bin/env python3
"""Cross-check the native and the JavaScript .psim code against each other.

  python3 native/tests/psim_cross.py [path/to/psim_tool] [--samples hook,wrench]

(a) the JS writes files (node scripts/psim.mjs fixtures, and built sample parts with a synthetic result from
    scripts/psim-sample.mjs); psim_tool verifies them and `psim_tool dump` equals `node scripts/psim.mjs dump`
    line by line, and `psim_tool inspect` equals `node scripts/psim.mjs inspect`;
(b) the native code writes files (psim_tool make-fixture, with and without CAD face ids, and psim_tool bench on the
    big sample files, which rewrites them); `node scripts/psim.mjs verify` accepts them and `dump` is identical to
    psim_tool's dump of the same file, and make-fixture's content equals the JS fixture's content.
Exit status 0 when everything agrees.
"""

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test" / "fixtures" / "psim" / "v1-tube-stride.psim"
failures = []


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, **kw)


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name)
    if not ok:
        failures.append(name)
        if detail:
            print(detail)


def diff_text(a, b, an, bn):
    lines = list(difflib.unified_diff(a.splitlines(), b.splitlines(), an, bn, lineterm="", n=1))
    return "\n".join(lines[:30])


def js(*args):
    return run(["node", "scripts/psim.mjs", *args])


def same_dump(tool, path, skip_table=False):
    n = run([tool, "dump", path])
    j = js("dump", path)
    if n.returncode or j.returncode:
        return False, f"native rc {n.returncode}: {n.stderr}\nJS rc {j.returncode}: {j.stderr}"
    a, b = n.stdout, j.stdout
    if skip_table:
        a = "\n".join(l for l in a.splitlines() if not l.startswith("table ")) + "\n"
        b = "\n".join(l for l in b.splitlines() if not l.startswith("table ")) + "\n"
    return a == b, diff_text(a, b, "native", "js")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?", default=str(ROOT / "native" / "build-web" / "psim_tool"))
    ap.add_argument("--samples", default="hook,wrench", help="sample ids for the JS-written real parts (comma list, 'all' for every sample)")
    ap.add_argument("--big", default="", help="comma list of sample ids to rewrite natively (default: the same as --samples)")
    args = ap.parse_args()
    tool = Path(args.tool)
    if not tool.is_file():
        sys.exit(f"{tool} not found: build the psim_tool target first")

    tmp = Path(tempfile.mkdtemp(prefix="psim-cross-"))
    saved = FIXTURE.read_bytes() if FIXTURE.exists() else None
    try:
        # ---- (a) JS writes, native reads
        r = js("fixtures")
        check("(a) node scripts/psim.mjs fixtures", r.returncode == 0, r.stderr)
        files = {"v1-tube (JS fixtures command)": FIXTURE}
        ids = [] if args.samples == "all" else [s for s in args.samples.split(",") if s]
        r = run(["node", "scripts/psim-sample.mjs", tmp / "js", *ids])
        check("(a) node scripts/psim-sample.mjs", r.returncode == 0, r.stderr)
        for line in r.stdout.splitlines():
            print("      " + line)
        for p in sorted((tmp / "js").glob("*.psim")):
            files[f"sample {p.stem} (JS)"] = p
        for name, p in files.items():
            v = run([tool, "verify", p])
            check(f"(a) psim_tool verify, {name}", v.returncode == 0, v.stdout + v.stderr)
            ok, d = same_dump(tool, p)
            check(f"(a) dump equals the JS dump, {name}", ok, d)
            n = run([tool, "inspect", p]).stdout.replace(str(p), "FILE")
            j = js("inspect", p).stdout.replace(str(p), "FILE")
            check(f"(a) inspect equals the JS inspect, {name}", n == j, diff_text(n, j, "native", "js"))

        # ---- (b) native writes, JS reads
        out = tmp / "native-fixture.psim"
        r = run([tool, "make-fixture", out])
        check("(b) psim_tool make-fixture", r.returncode == 0, r.stderr)
        v = js("verify", out)
        check("(b) node scripts/psim.mjs verify accepts it", v.returncode == 0, v.stdout + v.stderr)
        ok, d = same_dump(tool, out)
        check("(b) native dump equals JS dump of the native file", ok, d)
        n = run([tool, "dump", out]).stdout
        j = js("dump", FIXTURE).stdout
        a = "\n".join(l for l in n.splitlines() if not l.startswith("table ")) + "\n"
        b = "\n".join(l for l in j.splitlines() if not l.startswith("table ")) + "\n"
        check("(b) make-fixture content equals the JS fixture content (all numbers)", a == b, diff_text(a, b, "native", "js"))
        # the same, with CAD face ids: the JS side writes its own with sampleContent({ brep: true })
        jsbrep, natbrep = tmp / "js-brep.psim", tmp / "native-brep.psim"
        script = (
            "import {writePsim} from './src/core/psim.js';import {sampleContent,CREATED} from './test/helpers/psim-content.mjs';"
            "import {writeFileSync} from 'node:fs';"
            f"writeFileSync({str(jsbrep)!r}, await writePsim(sampleContent({{brep:true}}), {{created: CREATED}}));"
        )
        r = run(["node", "--input-type=module", "-e", script])
        check("(b) JS writes a CAD-faces file", r.returncode == 0, r.stderr)
        r = run([tool, "make-fixture", natbrep, "--brep"])
        check("(b) psim_tool make-fixture --brep", r.returncode == 0, r.stderr)
        v = js("verify", natbrep)
        check("(b) JS verify accepts the native CAD-faces file", v.returncode == 0, v.stdout + v.stderr)
        n = run([tool, "dump", natbrep]).stdout
        j = js("dump", jsbrep).stdout
        a = "\n".join(l for l in n.splitlines() if not l.startswith("table ")) + "\n"
        b = "\n".join(l for l in j.splitlines() if not l.startswith("table ")) + "\n"
        check("(b) CAD-faces content equals between the apps", a == b and "geom.faceFnv" in a, diff_text(a, b, "native", "js"))

        # real parts: the native code reads and rewrites the big JS files
        big = [s for s in (args.big or ("" if args.samples == "all" else args.samples)).split(",") if s]
        for p in sorted((tmp / "js").glob("*.psim")):
            if big and p.stem not in big:
                continue
            rewritten = tmp / f"native-{p.name}"
            r = run([tool, "bench", p, rewritten])
            check(f"(b) psim_tool bench rewrites {p.stem}", r.returncode == 0, r.stderr)
            print("      " + r.stdout.strip())
            v = js("verify", rewritten)
            check(f"(b) node verify accepts the native {p.stem}", v.returncode == 0, v.stdout + v.stderr)
            ok, d = same_dump(tool, rewritten)
            check(f"(b) dump of the native {p.stem} equal in both apps", ok, d)
            # same content as the JS file: the values pass through one more quantisation, so compare the counts only
            a = run([tool, "dump", p]).stdout
            b = run([tool, "dump", rewritten]).stdout
            keys = ("geom.vertices", "geom.triangles", "geom.faceCount", "rfea.array")
            ka = [l.split(" min=")[0] for l in a.splitlines() if l.startswith(keys)]
            kb = [l.split(" min=")[0] for l in b.splitlines() if l.startswith(keys)]
            check(f"(b) the native {p.stem} has the same counts and arrays", ka == kb and len(ka) > 3, diff_text("\n".join(ka), "\n".join(kb), "js", "native"))
    finally:
        if saved is not None:
            FIXTURE.write_bytes(saved)  # `fixtures` rewrites the tracked file; leave it as it was
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
