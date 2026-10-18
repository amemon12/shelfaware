#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
setup=0
case "${1:-}" in
  --setup) setup=1 ;;
  "") ;;
  *) printf 'Usage: bash scripts/start.sh [--setup]\n' >&2; exit 2 ;;
esac
command -v node >/dev/null || { echo 'Install Node 22.12+ (22.x) or Node 24 before starting.' >&2; exit 1; }
command -v npm >/dev/null || { echo 'npm is missing from PATH.' >&2; exit 1; }
node -e 'const [major,minor]=process.versions.node.split(".").map(Number); if (!((major===22&&minor>=12)||major===24||major>=26)) { console.error("Use Node 22.12+ (22.x), Node 24, or Node 26+; installed:",process.versions.node); process.exit(1); }'
if [ ! -f .venv/bin/python ]; then
  python_bin="${PYTHON_BIN:-python3}"
  "$python_bin" -c 'import sys; assert sys.version_info >= (3,11), "Use Python 3.11+; set PYTHON_BIN=python3.12 if the macOS default is older."'
  "$python_bin" -m venv .venv
  setup=1
fi
if [ "$setup" = 1 ]; then
  .venv/bin/python -m pip install -r backend/requirements.lock.txt
  (
    cd web
    npm ci
  )
fi
if [ ! -f web/node_modules/vite/bin/vite.js ]; then echo 'Run bash scripts/start.sh --setup first.' >&2; exit 1; fi
.venv/bin/python -c 'import fastapi, uvicorn, httpx, dotenv' || { echo 'Backend dependencies are missing; use --setup.' >&2; exit 1; }
# Refuse duplicate servers instead of silently connecting the UI to an old backend.
.venv/bin/python - <<'PY'
import socket
for port in (8000, 5173):
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is already in use. Reuse that server or stop it explicitly before starting another.')
PY
bind_address="${SHELF_BIND_ADDRESS:-127.0.0.1}"
case "$bind_address" in
  127.0.0.1|0.0.0.0) ;;
  *) echo 'SHELF_BIND_ADDRESS must be 127.0.0.1 or 0.0.0.0.' >&2; exit 2 ;;
esac
mkdir -p .runtime
api_pid=''
web_pid=''
cleanup() {
  if [ -n "$api_pid" ]; then kill "$api_pid" 2>/dev/null || true; fi
  if [ -n "$web_pid" ]; then kill "$web_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(
  cd backend
  exec ../.venv/bin/python -m uvicorn app.main:app --host "$bind_address" --port 8000
) > .runtime/backend.log 2>&1 &
api_pid=$!
(
  cd web
  exec node node_modules/vite/bin/vite.js --host "$bind_address" --port 5173 --strictPort
) > .runtime/web.log 2>&1 &
web_pid=$!
.venv/bin/python - <<'PY'
import json
import time
from urllib.request import urlopen
for name, url in [('API', 'http://127.0.0.1:8000/api/health'), ('web', 'http://127.0.0.1:5173/')]:
    deadline = time.monotonic() + 20
    while True:
        try:
            with urlopen(url, timeout=2) as response:
                if name == 'API' and json.loads(response.read(4096)).get('status') != 'ok':
                    raise ValueError('Unexpected health response')
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise SystemExit(f'{name} did not become ready. Inspect .runtime/backend.log and .runtime/web.log locally.')
            time.sleep(0.25)
PY
printf 'Shelf Aware ready: http://localhost:5173\nAPI PID: %s, web PID: %s. Ctrl+C stops both. Logs in .runtime/\n' "$api_pid" "$web_pid"
if [ "$bind_address" = '0.0.0.0' ]; then
  echo 'Trusted LAN mode: use the Mac LAN IP on the iPhone. Keep the development servers off the public internet.'
fi
# macOS ships Bash 3.2, so do not rely on newer `wait -n`.
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do sleep 1; done
echo 'A server exited. Both processes will stop; inspect .runtime logs locally.' >&2
exit 1

