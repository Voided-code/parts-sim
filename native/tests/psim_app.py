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
    for study, extra in [("modal", ""), ("buckling", ""), ("fatigue", ""), ("nonlinear", "mesh:6000;"), ("break", "mesh:20000;"), ("optimize-topology", "mesh:3000;"), ("optimize-sizing", "mesh:3000;")]:
        f = t / (study + ".psim")
        run = "break" if study == "break" else f"study:{study};run"
        if study.startswith("optimize"):
            run = f"study:optimize;studyopt:{study.split('-')[1]};run"
        rc, out = app(a.exe, f"sample:{'lbracket' if study == 'break' else 'beam'};idle;engine:cpu;{extra}{run};idle;savepsim:{f};quit")
        check(f"3b {study} save", rc == 0 and f.exists(), out)
        check(f"3b JS verifies {study}", js("verify", f).returncode == 0, js("verify", f).stderr)
        rc, out = app(a.exe, f"open:{f};idle;wait:800;assert:psim;status;quit")
        check(f"3b {study} reopens from the file", rc == 0 and "results shown from the file" in out, out)
    # 3d: study options round trip, one case per study (set non-default, save, reopen, read back)
    options = {
        "nonlinear": {"opts": {"large": False, "plastic": False, "mode": "failure", "steps": 17}, "view": {"plot": "disp", "exaggerate": 7}},
        "modal": {"opts": {"nev": 7}, "view": {"amp": 0.2, "animate": False}},
        "buckling": {"opts": {"nev": 5}, "view": {"amp": 0.11, "animate": True}},
        "fatigue": {"opts": {"loading": "custom", "R": -0.5, "cycles": 2e7, "finish": "polished", "scale": 1.5}, "view": {"plot": "fos"}},
        "drop": {"opts": {"height": 2.5}, "view": {"plot": "frame", "exaggerate": 3}},
        "dynamic": {"opts": {"source": "base", "dir": 2, "type": "sine", "nev": 6, "amp": 2, "zeta": 3, "pulse": 8, "sineF": 80, "duration": 0.5, "quake": 12}, "view": {"plot": "disp", "animate": False}},
        "optimize": {"opts": {"goal": "sizing", "keep": 55, "iters": 45, "fos": 3, "maxDisp": 2, "objective": "cost", "family": "metals"}, "view": {"level": 0.6}},
    }
    for study, want in options.items():
        f = t / ("opt-" + study + ".psim")
        j = __import__("json").dumps(want, separators=(",", ":"))
        rc, out = app(a.exe, f"sample:beam;idle;study:{study};setopt:{study}:{j};savepsim:{f};quit")
        check(f"3d {study} options save", rc == 0 and f.exists(), out)
        rc, out = app(a.exe, f"open:{f};idle;wait:500;getopt:{study};quit")
        m = re.search(r"options \S+: (\{.*\})", out)
        got = __import__("json").loads(m.group(1)) if m else {}
        ok = all(got.get(part, {}).get(k) == v or (isinstance(v, float) and abs(got.get(part, {}).get(k, 1e99) - v) < 1e-9) for part in want for k, v in want[part].items())
        check(f"3d {study} options come back after reopening", rc == 0 and m and ok, out + " got " + str(got))
    # 3f: the linear dynamic result is not stored: INFO.notStored says so, and opening says it in the status line
    dyn = t / "dyn.psim"
    rc, out = app(a.exe, f"sample:beam;idle;engine:cpu;mesh:3000;study:dynamic;run;idle;savepsim:{dyn};quit")
    check("3f dynamic run and save", rc == 0 and dyn.exists(), out)
    check("3f INFO.notStored is ['dynamic']", 'info.notStored[0] = "dynamic"' in js("dump", dyn).stdout, js("dump", dyn).stdout[:400])
    rc, out = app(a.exe, f"open:{dyn};idle;wait:500;status;quit")
    check("3f the status line names the missing dynamic result", rc == 0 and "dynamic results are not stored" in out, out)
    # 3e: a file whose options were written by another app (the JS tools) opens with them applied
    js_file = t / "opt-js.psim"
    script = (
        "import {readPsim, writePsim} from './src/core/psim.js'; import {readFileSync, writeFileSync} from 'node:fs';"
        f"const f = await readPsim(new Uint8Array(readFileSync({str(t / 'opt-modal.psim')!r})));"
        "f.setup.structural.options.modal = {opts: {nev: 9, bogus: 1}, view: {amp: 0.25, mode: 3}};"
        "f.setup.structural.options.drop = {opts: {height: 'tall'}, view: {}};"
        "f.setup.structural.options.fatigue = {opts: {R: 7}, view: {plot: 'nope'}};"
        "f.setup.structural.study = 'modal';"
        "const geometry = f.geometry; const rfea = f.rfea ? {meta: f.rfea.meta, arrays: []} : undefined;"
        f"writeFileSync({str(js_file)!r}, await writePsim({{info: f.info, geometry: {{...geometry, faceOf: null}}, setup: f.setup, view: f.view}}, {{created: f.info.created}}));"
    )
    r = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True)
    check("3e JS rewrites the setup with other options", r.returncode == 0 and js_file.exists(), r.stderr)
    rc, out = app(a.exe, f"open:{js_file};idle;wait:500;getopt:modal;getopt:drop;getopt:fatigue;quit")
    check("3e modal options from the file are applied (nev 9, amp 0.25)", '"nev":9' in out and '"amp":0.25' in out, out)
    check("3e wrong types and out-of-range values are ignored", '"height":1' in out and '"R":0' in out, out)
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
