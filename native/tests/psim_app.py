#!/usr/bin/env python3
"""Run the real app (PARTS_SIM_SCRIPT) through the .psim flows and check the numbers against the JavaScript tools.

  python3 native/tests/psim_app.py "native/build-web/Parts Sim.app/Contents/MacOS/Parts Sim" [--js-file smoke.psim]

1. sample -> run static (CPU) -> savepsim; node scripts/psim.mjs verify/dump accepts it;
2. restart with open:<that file> -> assert:psim, same maximum von Mises as before;
3. a thermal run saved and reopened;
4. a file written by the JavaScript code (scripts/psim-sample.mjs, plus --js-file, e.g. one saved by the website)
   opens in the app and shows the stored numbers; the app can save it again and the JS tool verifies that.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
fails = []


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name)
    if not ok:
        fails.append(name)
        print(detail)


def app(exe, script):
    r = subprocess.run([exe], env={**__import__("os").environ, "PARTS_SIM_SCRIPT": script}, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


def js(*a):
    return subprocess.run(["node", "scripts/psim.mjs", *map(str, a)], cwd=ROOT, capture_output=True, text=True)


def num(out, key):
    m = re.search(key + r" ([0-9.eE+-]+)", out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("--js-file", default="")
    a = ap.parse_args()
    t = Path(tempfile.mkdtemp(prefix="psim-app-"))
    beam, th = t / "beam.psim", t / "th.psim"
    rc, out = app(a.exe, f"sample:beam;idle;engine:cpu;run;idle;assert:static;result;savepsim:{beam};quit")
    vm = num(out, "max von Mises")
    check("1 run, save", rc == 0 and beam.exists() and vm, out)
    check("1 JS verifies the native file", js("verify", beam).returncode == 0, js("verify", beam).stderr)
    d = js("dump", beam).stdout
    m = re.search(r"rfea.meta.static.maxVM = (\S+)", d)
    check("1 JS dump shows the same maximum stress", m and abs(float(m.group(1)) - vm) <= 1e-6 * vm, d[:300])
    rc, out = app(a.exe, f"open:{beam};idle;assert:part;assert:psim;fileinfo;quit")
    vm2 = num(out, "max von Mises")
    check("2 open: results shown without a solve, same number", rc == 0 and "PARTS_SIM_PSIM_OK" in out and vm2 == vm, out)
    rc, out = app(a.exe, f"sample:hook;idle;tab:thermal;thtemp:0:100;thconv:1:25;run;idle;savepsim:{th};quit")
    check("3 thermal save", rc == 0 and th.exists(), out)
    rc, out = app(a.exe, f"open:{th};idle;assert:psim;status;quit")
    check("3 thermal reopen", rc == 0 and "Opened th.psim" in out, out)
    # 4: JS-written files
    subprocess.run(["node", "scripts/psim-sample.mjs", t / "js", "hook"], cwd=ROOT, capture_output=True)
    files = list((t / "js").glob("*.psim")) + ([Path(a.js_file)] if a.js_file else [])
    for f in files:
        d = js("dump", f).stdout
        want = re.search(r"rfea.meta.static.maxVM = (\S+)", d)
        again = t / ("re-" + f.name)
        rc, out = app(a.exe, f"open:{f};idle;assert:psim;result;savepsim:{again};quit")
        got = num(out, "max von Mises")
        check(f"4 app opens {f.name} with the stored maximum stress", rc == 0 and want and got and abs(got - float(want.group(1))) <= 1e-6 * got, out)
        check(f"4 JS verifies the app's copy of {f.name}", js("verify", again).returncode == 0, js("verify", again).stderr)
    print("\n%d failed" % len(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


sys.exit(main())
