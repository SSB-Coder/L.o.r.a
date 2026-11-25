"""Exercise real Cactus models through Lora; app execution is replaced with a recorder."""
import asyncio
import base64
import io
import json

from PIL import Image, ImageDraw, ImageFont

from lora.agent.runtime import AgentRuntime
from lora.core.config import CactusConfig
from lora.core.paths import PROJECT_ROOT
from lora.dialogue import ConversationHistory
from lora.models.cactus_backend import close_workers
from lora.tools.registry import ToolRegistry


async def main():
    registry = ToolRegistry()
    executed = []
    def record(name):
        executed.append(name)
        return {'status': 'success', 'message': f'Opened {name} (verification recorder).'}
    registry._tools['open_app'].handler = record
    opened_urls = []
    def record_url(url):
        opened_urls.append(url)
        return {'status': 'success', 'message': f'Opened {url} (verification recorder).'}
    registry._tools['open_url'].handler = record_url
    runtime = AgentRuntime('http://127.0.0.1:11434', None, ConversationHistory(), lambda: registry,
                           cactus_config=CactusConfig(needle_enabled=True, mirage_enabled=True))
    async def confirm(_):
        return False  # Verification never approves uncertain or privileged actions.
    runtime.confirm = confirm
    report = {}
    try:
        tool = await runtime.run_turn('Open Zed.', lambda _: None, lambda *_: None)
        report['tool_turn'] = {'text': tool.text, 'kind': tool.kind, 'meta': tool.meta, 'executed': executed}
        assert executed == ['zed'], report['tool_turn']
        website = await runtime.run_turn('Open Wikipedia', lambda _: None, lambda *_: None)
        report['website_turn'] = {'text': website.text, 'meta': website.meta, 'opened_urls': opened_urls}
        assert opened_urls == ['https://en.wikipedia.org'], report['website_turn']
        image = Image.new('RGB', (1000, 320), 'white')
        ImageDraw.Draw(image).text((40, 90), 'LORA MIRAGE READY', font=ImageFont.load_default(size=64), fill='black')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        visual = await runtime.run_turn('Read the large text in this image.', lambda _: None, lambda *_: None,
            attachments=[{'name': 'vision-test.png', 'mime': 'image/png',
                          'data_url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}])
        report['vision_turn'] = {'text': visual.text, 'kind': visual.kind, 'meta': visual.meta}
        assert visual.meta.get('vision_backend') == 'mirage' and not visual.failure, report['vision_turn']
        assert all(word in visual.text.upper() for word in ('LORA', 'MIRAGE', 'READY')), visual.text
        report['status'] = 'passed'
    finally:
        (PROJECT_ROOT / 'logs/fedora-setup/cactus-integration-verification.json').write_text(json.dumps(report, indent=2) + '\n')
        await registry.close()
        close_workers()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
