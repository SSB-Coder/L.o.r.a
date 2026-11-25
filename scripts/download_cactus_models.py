"""Prepare the pinned Cactus engine, Needle and Mirage; never download Whistle."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ['CACTUS_TELEMETRY'] = '0'
    from cactus.agent import fetch
    target = ROOT / 'models/cactus'
    target.mkdir(parents=True, exist_ok=True)
    assets = {'library': Path(fetch.fetch_library(dest_dir=fetch.engine_cache_dir())),
              'needle': Path(fetch.fetch_weights(3)),
              'mirage': Path(fetch.fetch_weights(fetch.MIRAGE))}
    report = {'package': importlib.metadata.version('cactus-compute'),
              'engine_version': fetch.ENGINE_VERSION, 'engine_revision': fetch.ENGINE_REVISION,
              'assets': {}}
    for name, source in assets.items():
        source = source.resolve()
        link = target / source.name
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(source)
        with source.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        report['assets'][name] = {'path': str(link), 'cache_path': str(source),
                                 'bytes': source.stat().st_size, 'sha256': digest}
    (target / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
