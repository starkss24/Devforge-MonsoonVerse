"""Lite repo scan (placeholder for kit repo_scan). Stdlib only."""
import ast, os, subprocess, sys, sys as _s

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED = {"torch", "numpy", "scipy", "pandas", "xarray", "sklearn", "lightgbm",
           "onnxruntime", "fastapi", "flask", "uvicorn"}
LOCAL = {"common", "twin", "serve"}
CHECKED = ["serve.py", "common.py", "twin.py"]
MAX = 50 * 1024 * 1024


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def main():
    errs = []
    files = [f for f in git("ls-files", "-co", "--exclude-standard", "-z").split("\0") if f]
    lfs = {}
    if files:
        out = subprocess.run(["git", "check-attr", "filter", "--stdin", "-z"], cwd=ROOT,
                             input="\0".join(files), capture_output=True, text=True).stdout.split("\0")
        lfs = {out[i]: out[i + 2] == "lfs" for i in range(0, len(out) - 2, 3)}
    for f in files:
        p = os.path.join(ROOT, f)
        if not os.path.isfile(p):
            continue
        in_lfs = lfs.get(f, False)
        if f.endswith(".nc"):
            errs.append(f"data file: {f}")
        if f.endswith(".pt") and not in_lfs:
            errs.append(f".pt not in LFS: {f}")
        if os.path.getsize(p) > MAX and not in_lfs:
            errs.append(f">50MB outside LFS: {f}")
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
