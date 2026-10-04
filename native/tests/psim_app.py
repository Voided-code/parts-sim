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
    # 3b: the other studies round trip (save, JS verify, reopen shows results without a solve)
    for study, extra in [("modal", ""), ("buckling", ""), ("fatigue", ""), ("nonlinear", "mesh:6000;"), ("break", "mesh:20000;")]:
        f = t / (study + ".psim")
        run = "break" if study == "break" else f"study:{study};run"
        rc, out = app(a.exe, f"sample:{'lbracket' if study == 'break' else 'beam'};idle;engine:cpu;{extra}{run};idle;savepsim:{f};quit")
        check(f"3b {study} save", rc == 0 and f.exists(), out)
        check(f"3b JS verifies {study}", js("verify", f).returncode == 0, js("verify", f).stderr)
        rc, out = app(a.exe, f"open:{f};idle;wait:800;assert:psim;status;quit")
        check(f"3b {study} reopens from the file", rc == 0 and "results shown from the file" in out, out)
    # 3c: airflow at ~50k cells on the CPU engine
    af = t / "air.psim"
    rc, out = app(a.exe, f"sample:ahmed;idle;tab:airflow;airengine:cpu;cells:50000;run;idle;airwait:300;idle;aero;savepsim:{af};quit")
    cd = re.search(r"Cd (\S+) ", out)
    check("3c airflow save", rc == 0 and af.exists() and cd, out)
    check("3c JS verifies airflow", js("verify", af).returncode == 0, js("verify", af).stderr)
    rc, out = app(a.exe, f"open:{af};idle;wait:1500;aero;assert:psim;quit")
    cd2 = re.search(r"Cd (\S+) ", out)
    check("3c airflow reopens with the same Cd, no solver", rc == 0 and cd and cd2 and cd.group(1) == cd2.group(1), out)
    # 4: JS-written files
    subprocess.run(["node", "scripts/psim-sample.mjs", t / "js", "hook"], cwd=ROOT, capture_output=True)
    files = list((t / "js").glob("*.psim")) + ([Path(a.js_file)] if a.js_file else [])
    for f in files:
        if "app-airflow" in f.name or "airflow" in f.name:
            rc, out = app(a.exe, f"open:{f};idle;wait:1500;aero;assert:psim;quit")
            want = re.search(r"rair.meta.results.cd = (\S+)", js("dump", f).stdout)
            got = re.search(r"Cd (\S+) ", out)
            check(f"4 app opens {f.name} with the stored Cd", rc == 0 and want and got and abs(float(got.group(1)) - float(want.group(1))) < 5e-4, out)
            continue
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
