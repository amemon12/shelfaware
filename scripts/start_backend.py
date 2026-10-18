"""Backend-only launcher. No Node, no web server, and no secrets printed."""
from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import sys


def ensure_port_available(host: str = '127.0.0.1', port: int = 8000) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(f'Port {port} is unavailable. Reuse the existing backend or stop it explicitly; this launcher never kills processes.') from exc


def private_write(path: Path, text: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f'Refusing to write a private file through a symlink: {path.name}')
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        if hasattr(os, 'fchmod'):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            descriptor = -1
            output.write(text)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path.chmod(0o600)


def ensure_lan_token(root: Path, environment: dict[str, str]) -> str:
    # Imported only after --setup has installed the existing backend dependencies.
    from dotenv import dotenv_values, set_key

    env_file = root / '.env'
    if env_file.is_symlink():
        raise RuntimeError('Refusing to update .env through a symlink.')
    values = dotenv_values(env_file) if env_file.exists() else {}
    saved = (values.get('SHELF_API_TOKEN') or '').strip()
    supplied = environment.get('SHELF_API_TOKEN', '').strip()
    if saved and supplied and saved != supplied:
        raise RuntimeError('SHELF_API_TOKEN in the environment differs from .env. Unset the environment override or align them explicitly; neither value was printed or changed.')
    token = supplied or saved or secrets.token_urlsafe(32)
    if not token.isascii() or any(character.isspace() or ord(character) < 33 or ord(character) > 126 for character in token):
        raise RuntimeError('SHELF_API_TOKEN must contain visible ASCII characters without whitespace. Its value was not printed.')
    if not env_file.exists():
        private_write(env_file, '# Local credentials. Do not share this file.\n')
    if not saved:
        set_key(str(env_file), 'SHELF_API_TOKEN', token, quote_mode='always')
    env_file.chmod(0o600)
    environment['SHELF_API_TOKEN'] = token
    return token


def command_output(arguments: list[str]) -> str:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=4, check=False)
        return result.stdout if result.returncode == 0 else ''
    except (OSError, subprocess.TimeoutExpired):
        return ''


def lan_candidates(system: str | None = None) -> list[tuple[str, str]]:
    """Read actual local interface addresses; never guess a router/subnet or contact the internet."""
    system = system or platform.system()
    found: list[tuple[str, str]] = []

    def add(interface: str, address: str) -> None:
        try:
            parsed = ipaddress.IPv4Address(address.strip())
        except ipaddress.AddressValueError:
            return
        # macOS can expose a CLAT /32 on IPv6-only hotspots. It is a service
        # continuity address (RFC 7335), not an origin to advertise to peers.
        if parsed in ipaddress.IPv4Network('192.0.0.0/29'):
            return
        if not (parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast):
            item = (interface, str(parsed))
            if item not in found:
                found.append(item)

    if system == 'Darwin':
        ports = command_output(['networksetup', '-listallhardwareports'])
        interfaces = re.findall(r'^Device:\s*([A-Za-z0-9]+)\s*$', ports, re.M)
        for interface in interfaces:
            add(interface, command_output(['ipconfig', 'getifaddr', interface]))
        if not found:
            interface = 'interface'
            for line in command_output(['ifconfig']).splitlines():
                if line and not line[0].isspace() and ':' in line:
                    interface = line.split(':', 1)[0]
                match = re.search(r'\binet\s+(\d+\.\d+\.\d+\.\d+)\b', line)
                if match:
                    add(interface, match[1])
    elif system == 'Windows':
        for address in re.findall(r'IPv4[^:\n]*:\s*(\d+\.\d+\.\d+\.\d+)', command_output(['ipconfig'])):
            add('IPv4 interface', address)
    else:
        for address in command_output(['hostname', '-I']).split():
            add('local interface', address)
    return found


def save_phone_connection(root: Path, token: str, candidates: list[tuple[str, str]], port: int = 8000) -> Path:
    runtime = root / '.runtime'
    if runtime.is_symlink():
        raise RuntimeError('Refusing a symlinked .runtime directory.')
    runtime.mkdir(mode=0o700, exist_ok=True)
    runtime.chmod(0o700)
    path = runtime / 'phone-connection.txt'
    urls = '\n'.join(f'  {interface}: http://{address}:{port}' for interface, address in candidates)
    if not urls:
        urls = '  No address detected. Find this Mac IPv4 address in System Settings > Network.'
    private_write(path, 'Shelf Aware iPhone connection — private local file\n\n'
        'Backend URL candidates (choose the active interface on the same network as the iPhone):\n'
        + urls + '\n\nAPI bearer token (paste into the iPhone app token field):\n' + token
        + '\n\nUse the origin shown above, without /api. Do not share this file or .env.\n'
        'This is the backend API only; no browser dashboard is started.\n')
    return path


def launch(root: Path, lan: bool, port: int = 8000) -> int:
    from dotenv import load_dotenv

    host = '0.0.0.0' if lan else '127.0.0.1'
    ensure_port_available(host, port)
    if lan:
        token = ensure_lan_token(root, os.environ)
        candidates = lan_candidates()
        connection = save_phone_connection(root, token, candidates, port)
        print(f'Phone bearer token saved privately in: {connection}', flush=True)
        for interface, address in candidates:
            print(f'Backend URL candidate ({interface}): http://{address}:{port}', flush=True)
        if not candidates:
            print('No LAN IPv4 address detected. Check System Settings > Network.', flush=True)
        print(f'Use the same trusted network on the Mac and iPhone. Only API port {port} will listen on LAN.', flush=True)
    else:
        print(f'Backend: http://127.0.0.1:{port} (localhost only)', flush=True)
    load_dotenv(root / '.env', override=False)
    print('Public health check: /api/health. Press Ctrl+C to stop. No web server is started.', flush=True)
    os.chdir(root / 'backend')
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', host,
                            '--port', str(port), '--no-access-log'])
    return 0


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Port must be an integer from 1 to 65535.') from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError('Port must be an integer from 1 to 65535.')
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Start only the Shelf Aware backend; no Node required.')
    parser.add_argument('--setup', action='store_true', help='Create .venv if necessary and install locked backend dependencies.')
    parser.add_argument('--lan', action='store_true', help='Explicitly serve API on LAN with a local bearer token.')
    parser.add_argument('--port', type=parse_port, default=8000, help='API port (default: 8000).')
    parser.add_argument('--_venv', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11 or newer is required. Install it and run python3 scripts/start_backend.py --setup.')
    if arguments._venv:
        return launch(root, arguments.lan, arguments.port)
    ensure_port_available('0.0.0.0' if arguments.lan else '127.0.0.1', arguments.port)
    venv = root / '.venv'
    python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if arguments.setup:
        if not python.exists():
            subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
        requirements = root / 'backend' / 'requirements.lock.txt'
        if not requirements.exists():
            requirements = root / 'backend' / 'requirements.txt'
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    if not python.exists():
        raise RuntimeError('The backend virtual environment is missing. Run python3 scripts/start_backend.py --setup first.')
    command = [str(python), str(Path(__file__).resolve()), '--_venv', '--port', str(arguments.port)]
    if arguments.lan:
        command.append('--lan')
    os.execv(str(python), command)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        # Never print environment values or subprocess arguments containing credentials.
        print(f'Backend launcher: {error}', file=sys.stderr)
        raise SystemExit(1)
