"""Real offline Needle/Mirage smoke checks; never execute a proposed action."""
import argparse
import json
import tempfile
import time
from pathlib import Path

from lora.core.config import CactusConfig
from lora.core.paths import PROJECT_ROOT
from lora.models.cactus_backend import close_workers, validate_calls, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    config = CactusConfig()
    tools = [{'name': 'open_app', 'description': 'Open an installed desktop application.',
              'parameters': {'type': 'object', 'properties': {'name': {'type': 'string', 'enum': ['zed', 'calculator']}},
                             'required': ['name'], 'additionalProperties': False}}]
    started = time.monotonic()
    report = {'needle': worker('needle', config).call({'tools': tools, 'text': 'Open Zed.'})}
    calls = validate_calls(report['needle'], tools)
    assert calls == [{'name': 'open_app', 'arguments': {'name': 'zed'}}], report['needle']
    report['needle']['seconds'] = round(time.monotonic() - started, 3)
    negative = worker('needle', config).call({'tools': tools, 'text': 'Do not open Zed.'})
    assert validate_calls(negative, tools) == [], negative
    independent = worker('needle', config).call({'tools': tools, 'text': 'Open calculator.'})
    assert validate_calls(independent, tools) == [{'name': 'open_app', 'arguments': {'name': 'calculator'}}], independent
    report['needle_negation_and_stateless_turns'] = 'passed'
    with tempfile.TemporaryDirectory(prefix='lora-mirage-') as folder:
        if args.image:
            image = args.image.resolve()
        else:
            from PIL import Image, ImageDraw, ImageFont
            image = Path(folder) / 'test.png'
            canvas = Image.new('RGB', (1000, 320), 'white')
            ImageDraw.Draw(canvas).text((40, 90), 'LORA MIRAGE READY', font=ImageFont.load_default(size=64), fill='black')
            canvas.save(image)
        started = time.monotonic()
        report['mirage'] = worker('mirage', config).call({'path': str(image)})
        report['mirage']['seconds'] = round(time.monotonic() - started, 3)
        if args.image is None:
            recognized = ' '.join(item.get('string', '') for item in report['mirage'].get('text', []))
            assert 'LORA MIRAGE READY' in recognized.upper(), recognized
    report['status'] = 'passed'
    output = PROJECT_ROOT / 'logs/fedora-setup/cactus-model-verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    close_workers()


if __name__ == '__main__':
    main()
