"""Read-only handoff checks; never prints credential values or installs software."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=15, check=False)
        return result.stdout.strip()[:300] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mac', action='store_true', help='Include Xcode and XcodeGen checks.')
    parser.add_argument('--api-url', help='Optionally check backend health, e.g. http://127.0.0.1:8000')
    args = parser.parse_args()
    failed: list[str] = []

    def check(label: str, ok: bool, detail: str) -> None:
        print(f'{"PASS" if ok else "NEEDS ACTION"} | {label}: {detail}')
        if not ok:
            failed.append(label)

    print(f'Shelf Aware handoff checks | {platform.system()} | {ROOT}')
    check('Python', sys.version_info >= (3, 11), platform.python_version())
    node = version(['node', '--version'])
    match = re.search(r'v?(\d+)\.(\d+)\.(\d+)', node or '')
    major, minor = (int(match[1]), int(match[2])) if match else (0, 0)
    check('Node', (major == 22 and minor >= 12) or major == 24 or major >= 26, node or 'missing; use Node 22.12+ (22.x) or 24')
    check('npm', version(['npm', '--version']) is not None, 'available' if shutil.which('npm') else 'missing')
    check('Web dependencies', (ROOT / 'web/node_modules/vite/bin/vite.js').is_file(), 'run bash scripts/start.sh --setup to install from the lockfile')
    missing = [name for name in ('fastapi', 'uvicorn', 'httpx', 'dotenv', 'pytest') if importlib.util.find_spec(name) is None]
    check('Python dependencies in this interpreter', not missing, ', '.join(missing) if missing else 'installed')
    check('Source bundle', all((ROOT / path).is_file() for path in ('CONTRACT.md', 'ios/project.yml', 'backend/app/main.py', 'web/package-lock.json')), 'backend, web, iOS and contract')

    env_path = ROOT / '.env'
    # Only state booleans; never output a value or copy this file into a package.
    configured: dict[str, bool] = {}
    if env_path.is_file():
        for line in env_path.read_text(encoding='utf-8-sig').splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() in {'PERCEPTRON_API_KEY', 'GEMINI_API_KEY', 'SHELF_API_TOKEN'}:
                configured[key.strip()] = bool(value.strip().strip('\"\''))
    print('INFO | .env present:', env_path.is_file())
    for key in ('PERCEPTRON_API_KEY', 'GEMINI_API_KEY', 'SHELF_API_TOKEN'):
        print(f'INFO | {key} in .env: {"set" if configured.get(key) else "not set"}; provider access is not tested by this check')
    print('INFO | Fixture mode needs no keys. Real OCR requires Perceptron. Gemini is optional for flexible intents.')

    if args.mac:
        check('Mac host', platform.system() == 'Darwin', platform.system())
        xcode = version(['xcodebuild', '-version'])
        xcode_match = re.search(r'Xcode (\d+)\.(\d+)', xcode or '')
        xcode_ok = bool(xcode_match and (int(xcode_match[1]), int(xcode_match[2])) >= (26, 4))
        check('Xcode for DAT1.0.1', xcode_ok, (xcode or 'full Xcode26.4+ missing/not selected').replace('\n', ' | '))
        swift = version(['xcrun', 'swift', '--version'])
        swift_match = re.search(r'Swift version (\d+)\.(\d+)', swift or '')
        check('Swift compiler', bool(swift_match and (int(swift_match[1]), int(swift_match[2])) >= (6, 3)), (swift or 'compiler unavailable').replace('\n', ' | '))
        check('XcodeGen', version(['xcodegen', '--version']) is not None, version(['xcodegen', '--version']) or 'brew install xcodegen')
        print('MANUAL | Check signing capabilities for the chosen target, device trust + Developer Mode, glasses pairing + Meta Developer Mode. Original Wi-Fi target requires paid capabilities; optional Bluetooth Personal Team target still needs real hardware verification.')

    if args.api_url:
        parsed = urlsplit(args.api_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise SystemExit('Use an HTTP(S) backend origin without credentials, query or fragment.')
        try:
            with urlopen(args.api_url.rstrip('/') + '/api/health', timeout=4) as response:
                result = json.loads(response.read(4096))
            check('Backend reachable', result.get('status') == 'ok', str(result.get('status', 'unknown')))
            flags = {key: bool(value) for key, value in result.get('providers', {}).items() if key in ('perceptron', 'gemini', 'deepseek')}
            print('INFO | Backend configured flags only:', json.dumps(flags))
        except Exception as exc:
            check('Backend reachable', False, type(exc).__name__ + '; verify process, address, firewall and LAN isolation')

    print('NOT VERIFIED | Native compilation/signing, real glasses frames/audio, cloud accuracy/latency, or stage readiness.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
