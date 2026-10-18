"""Package source for the Mac team without credentials, media, caches or installed runtimes."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root.parent / 'Shelf-Aware-Mac-Handoff.zip'
excluded_dirs = {'.git', '.venv', 'node_modules', 'dist', '.runtime', '__pycache__', '.pytest_cache', 'private-media', 'DerivedData'}

with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
    count = 0
    manifest = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded_dirs or part.endswith(('.xcuserdata', '.xcodeproj', '.xcworkspace')) for part in relative.parts):
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            continue
        if path.suffix in {'.pyc', '.log', '.tsbuildinfo', '.zip'}:
            continue
        if relative.parts[:2] == ('tools', 'visual_matcher') and (
                'data' in relative.parts[2:] or path.name.endswith('.index.json')
                or path.name in {'evaluation.json', 'manifest.json', 'holdout.json'}):
            continue
        archive_name = (Path('ShelfAware') / relative).as_posix()
        contents = path.read_bytes()
        archive.writestr(archive_name, contents)
        manifest.append({'path': relative.as_posix(), 'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()})
        count += 1
    archive.writestr('ShelfAware/HANDOFF-MANIFEST.json', json.dumps({'format': 1, 'files': manifest}, indent=2) + '\n')
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
destination.with_suffix(destination.suffix + '.sha256').write_text(f'{digest}  {destination.name}\n', encoding='utf-8')
print(f'Packaged {count} source/documentation files + manifest: {destination}')
print(f'SHA256: {digest}')
