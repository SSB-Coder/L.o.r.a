"""Prepare the pinned Cactus engine, Needle and Mirage; never download Whistle."""
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_asset(source: Path, target: Path) -> None:
    if target.is_symlink() and target.resolve() != source:
        target.unlink()
    if target.exists():
        if target.samefile(source):
            return
        if target.stat().st_size == source.stat().st_size:
            with target.open('rb') as current, source.open('rb') as cached:
                if hashlib.file_digest(current, 'sha256').digest() == hashlib.file_digest(cached, 'sha256').digest():
                    return
    try:
        target.symlink_to(source)
    except OSError:
        try:
            os.link(source, target)
        except OSError:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as staging:
                staging_path = Path(staging.name)
            try:
                shutil.copy2(source, staging_path)
                staging_path.replace(target)
            finally:
                staging_path.unlink(missing_ok=True)


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
        link_asset(source, link)
        with link.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        report['assets'][name] = {'path': str(link), 'cache_path': str(source),
                                 'bytes': link.stat().st_size, 'sha256': digest}
    staging_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target, delete=False) as staging:
            staging_path = Path(staging.name)
            staging.write(json.dumps(report, indent=2) + '\n')
        staging_path.replace(target / 'manifest.json')
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
