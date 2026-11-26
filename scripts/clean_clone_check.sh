#!/usr/bin/env bash
# Clean-clone check: clone, venv, install, serve, validate.
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; PID=""
cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT
fail() { echo "FAIL: $*"; exit 1; }
git -C "$SRC" rev-parse -q --verify HEAD >/dev/null || fail "no commits yet: the clone check tests committed content only"
[ -n "$(git -C "$SRC" status --porcelain)" ] && echo "WARN: uncommitted changes are NOT part of this check"
git clone -q "$SRC" "$TMP/repo" || fail clone
cd "$TMP/repo"
# Weights are plain git files (no Git LFS): a fresh clone must contain the real bytes, never LFS pointers.
for w in weights/*; do
  [ -f "$w" ] || continue
  head -c 40 "$w" | grep -q "^version https://git-lfs" && fail "$w is a Git LFS pointer in the clean clone"
done
echo "weights: $(ls weights | grep -v '^\.gitkeep$' | tr '\n' ' ')(no LFS pointers)"
python -m venv .venv || fail venv
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
"$PY" -m pip install -q -r requirements.txt || fail install
PORT=$("$PY" -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
PORT=$PORT "$PY" serve.py --port "$PORT" > serve.log 2>&1 & PID=$!
# Grader allows 120 s wall-clock for /health: measure seconds, not tries.
START=$SECONDS
until "$PY" -c "import urllib.request as u;u.urlopen(u.Request('http://127.0.0.1:$PORT/health',method='POST'),timeout=2)" 2>/dev/null; do
  [ $((SECONDS - START)) -ge 120 ] && { cat serve.log; fail "health not ready within 120 s"; }
  sleep 1
done
echo "health ready after $((SECONDS - START)) s"
"$PY" tools/validate.py --url "http://127.0.0.1:$PORT" || fail validate
echo PASS
