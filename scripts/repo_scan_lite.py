"""Lite repo scan (placeholder for kit repo_scan). Stdlib only.

Fails on: data files (*.nc); weights files (*.pt/.onnx/.npz/...) outside weights/, stored as Git LFS pointers
(weights are plain git files) or > 95 MB; any other file > 50 MB; disallowed imports in the inference files."""
import ast, os, subprocess, sys, sys as _s

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"torch", "numpy", "scipy", "pandas", "xarray", "sklearn", "lightgbm",
           "onnxruntime", "fastapi", "flask", "uvicorn", "gunicorn"}  # R1_CONTRACT.md list
LOCAL = {"common", "twin", "serve", "postproc"}
CHECKED = ["serve.py", "common.py", "twin.py", "postproc.py"]
MAX = 50 * 1024 * 1024          # any other file
WEIGHTS_MAX = 95 * 1024 * 1024  # weights are plain git files; GitHub rejects files > 100 MB
WEIGHT_EXT = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npz")
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def main():
    errs = []
    files = [f for f in git("ls-files", "-co", "--exclude-standard", "-z").split("\0") if f]
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p):
            continue
        size = os.path.getsize(p)
        if f.endswith(".nc"):
            errs.append(f"data file: {f}")
        if f.lower().endswith(WEIGHT_EXT):
            # Weights: plain git files under weights/ (no Git LFS), each < 95 MB.
            if not f.startswith("weights/"):
                errs.append(f"weights file outside weights/: {f}")
            with open(p, "rb") as fh:
                if fh.read(len(LFS_POINTER)) == LFS_POINTER:
                    errs.append(f"weights file is a Git LFS pointer, not the real bytes: {f}")
            if size > WEIGHTS_MAX:
                errs.append(f"weights file > 95 MB: {f} ({size / 2**20:.1f} MB)")
        elif size > MAX:
            errs.append(f">50MB file: {f}")
    stdlib = getattr(_s, "stdlib_module_names", set())
    for name in CHECKED:
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            print(f"note: {name} missing, skipped")
            continue
        tree = ast.parse(open(p, encoding="utf-8").read(), name)
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [n.module]
            for m in mods:
                top = m.split(".")[0]
                if top not in ALLOWED | LOCAL and top not in stdlib and top != "__future__":
                    errs.append(f"{name}:{n.lineno} disallowed import '{m}'")
    for e in errs:
        print("FAIL:", e)
    print(f"scanned {len(files)} files: {'FAIL' if errs else 'PASS'}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
