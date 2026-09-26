#!/usr/bin/env python3
"""Run a bounded beam/static GUI smoke test against an installed or packaged app.

Examples:
  python3 native/tests/smoke_app.py '/Applications/Parts Sim.app'
  xvfb-run -a python3 native/tests/smoke_app.py dist/Parts_Sim.AppImage --screenshot smoke.png
  python native/tests/smoke_app.py 'C:\\Parts Sim\\parts-sim.exe' --timeout 180
"""

import argparse
import math
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import sys
import tempfile


MARKER = "PARTS_SIM_SMOKE_OK\n"


def executable_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir() and path.suffix.lower() == ".app":
        with (path / "Contents" / "Info.plist").open("rb") as stream:
            name = plistlib.load(stream)["CFBundleExecutable"]
        path = path / "Contents" / "MacOS" / name
    if not path.is_file():
        raise ValueError(f"App executable not found: {path}")
    return path


def stop_process_tree(process):
    if os.name == "nt":
        # Qt workers and AppImage launchers must not outlive a timed-out smoke test.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("app", help="App executable, AppImage, or macOS .app bundle")
    parser.add_argument("--timeout", type=float, default=180, help="Maximum runtime in seconds (default: 180)")
    parser.add_argument("--screenshot", type=Path, help="Save the completed static result as a PNG")
    parser.add_argument("--log", type=Path, help="Save captured application output")
    parser.add_argument("--isolated", action="store_true", help="Remove SDK/library search paths from the child environment")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")

    try:
        app = executable_path(args.app)
        steps = ["sample:beam", "idle", "assert:part", "tab:structural", "study:static", "run", "idle", "assert:static"]
        screenshot = args.screenshot.expanduser().resolve() if args.screenshot else None
        if screenshot:
            if ";" in str(screenshot):
                raise ValueError("Screenshot path cannot contain ';' (the script command separator)")
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            steps.append(f"shot:{screenshot}")
        steps.append("quit")
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="parts-sim-smoke-") as temporary:
            marker_path = Path(temporary) / "result.txt"
            env = os.environ.copy()
            if args.isolated:
                for key in ("QT_ROOT_DIR", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML_IMPORT_PATH",
                            "QML2_IMPORT_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
                    env.pop(key, None)
                if os.name == "nt":
                    windows = Path(env.get("SystemRoot", r"C:\Windows"))
                    env["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))
            env["PARTS_SIM_SCRIPT"] = ";".join(steps)
            env["PARTS_SIM_SCRIPT_RESULT"] = str(marker_path)
            if app.suffix.lower() == ".appimage":
                env.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")
            print(f"Smoke testing {app} (timeout {args.timeout:g}s)", flush=True)
            process = subprocess.Popen(
                [str(app)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", start_new_session=os.name != "nt",
            )
            timed_out = False
            try:
                output, _ = process.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process_tree(process)
                try:
                    output, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired as error:
                    output = error.output or b""
                    if isinstance(output, bytes):
                        output = output.decode("utf-8", errors="replace")
                    process.stdout.close()
            except KeyboardInterrupt:
                stop_process_tree(process)
                process.wait(timeout=10)
                raise

            if output:
                print(output, end="" if output.endswith("\n") else "\n", flush=True)
            if args.log:
                args.log.write_text(output, encoding="utf-8")
            if timed_out:
                raise RuntimeError(f"Application exceeded the {args.timeout:g}s timeout")
            if process.returncode != 0:
                raise RuntimeError(f"Application exited with code {process.returncode}")
            if not marker_path.is_file() or marker_path.read_text(encoding="utf-8") != MARKER:
                raise RuntimeError("Application did not confirm a loaded part and successful static solve")
            if screenshot and (not screenshot.is_file() or screenshot.stat().st_size == 0):
                raise RuntimeError(f"Application did not save a screenshot: {screenshot}")
    except (OSError, ValueError, KeyError, RuntimeError, plistlib.InvalidFileException) as error:
        print(f"SMOKE FAILED: {error}", file=sys.stderr)
        return 1
    print("SMOKE PASSED: beam loaded and static study completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
