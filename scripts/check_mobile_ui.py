"""Render the actual mobile app against scripted gateway responses, without a PC session."""

import json
import mimetypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
scratch = ROOT / "scratch/mobile-browser"
scratch.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(scratch)
if os.name == "nt":
    sys.path.insert(0, str(ROOT / "tests"))
    from run_sandbox_tests import configure

    configure()

from playwright.sync_api import sync_playwright  # noqa: E402

ASSETS = ROOT / "src/lora/gateway/static/app"
OUT = ROOT / "docs/validation"
OUT.mkdir(exist_ok=True)


def check():
    results = {"viewports": [], "errors": []}
    with sync_playwright() as driver:
        executable = os.getenv("LORA_TEST_BROWSER")
        if not executable and os.name == "nt":
            chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
            executable = str(
                chrome
                if chrome.exists()
                else Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
                / "Microsoft/Edge/Application/msedge.exe"
            )
        browser = driver.chromium.launch(executable_path=executable, headless=True)
        context = browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            has_touch=True,
            device_scale_factor=2,
            color_scheme="light",
        )
        page = context.new_page()
        page.on("pageerror", lambda error: results["errors"].append(str(error)))
        page.add_init_script("""
window.WebSocket=class {
 constructor(){window.previewSocket=this;setTimeout(()=>this.onopen?.(),10)}
 send(){setTimeout(()=>this.onmessage?.({data:JSON.stringify({type:'auth_ok'})}),5)}
 close(){}
};
""")

        requests = []
        scenario = {"kind": "research", "status": "completed"}

        def respond(route):
            path = route.request.url.split("lora.test", 1)[1]
            if path == "/v1/ask":
                requests.append(route.request.post_data_json)
                route.fulfill(json={"job_id": f"preview-job-{len(requests)}", "queue_position": 0})
            elif path.startswith("/v1/jobs/"):
                route.fulfill(
                    json={
                        "status": scenario["status"],
                        "result": {
                            "kind": scenario["kind"],
                            "answer": "Redshift is the stretching of light toward longer wavelengths [S1].\n\nFor distant galaxies, the expansion of space is a major cause." if scenario["kind"] == "research" else "Sleep well. We can pick this up tomorrow.",
                            "sources": [
                                {
                                    "id": "S1",
                                    "title": "Understanding redshift",
                                    "url": "https://example.org/redshift",
                                    "quote": "Light from distant galaxies is shifted to longer wavelengths.",
                                    "tier": 2,
                                }
                            ] if scenario["kind"] == "research" else [],
                            "verification": {"source_check": {"status": "references_checked", "note": "Citation references checked; claims were not independently fact-checked."}} if scenario["kind"] == "research" else {},
                        },
                    }
                )
            elif path == "/v1/notebooks":
                route.fulfill(json={"notebooks": []})
            elif path == "/v1/stop":
                route.fulfill(json={"stopped": True})
            else:
                name = path.rsplit("/", 1)[-1] or "index.html"
                file = ASSETS / name
                if file.parent != ASSETS or not file.is_file():
                    route.fulfill(status=404)
                else:
                    route.fulfill(
                        body=file.read_bytes(),
                        content_type=mimetypes.guess_type(file)[0] or "text/plain",
                    )

        page.route("http://lora.test/**", respond)
        page.goto("http://lora.test/")
        page.get_by_text("Connected", exact=True).wait_for()
        page.screenshot(path=str(OUT / "mobile-light.png"))
        for width, height in [(320, 568), (390, 844), (768, 1024), (390, 430)]:
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(80)
            metrics = page.evaluate("""()=>({width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth>innerWidth,
              inputBottom:document.getElementById('ask-form').getBoundingClientRect().bottom})""")
            assert not metrics["overflow"] and metrics["inputBottom"] <= height + 1, metrics
            results["viewports"].append(metrics)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator(".style-menu summary").click()
        page.locator('[data-depth="simple"]').click()
        assert page.locator("#depth-label").inner_text() == "Simple"
        page.locator("#tts-toggle").click()  # no audio in a visual acceptance check
        page.locator("#query-input").fill("What is redshift?")
        page.locator("#submit-btn").click()
        assert page.locator("#submit-btn").is_disabled()
        page.locator(".citation-badge").wait_for()
        assert page.locator(".assistant-message").count() == 1
        assert page.locator("#welcome").count() == 0
        assert page.locator("#submit-btn").is_enabled()
        assert requests[-1]["run_mode"] == "quick" and requests[-1]["depth"] == "simple"
        assert page.locator(".source-check").count() == 1
        assert page.locator(".confidence-pill, .unverified-box").count() == 0
        page.screenshot(path=str(OUT / "mobile-conversation.png"))
        page.locator(".citation-badge").click()
        assert page.locator("#close-sheet-btn").evaluate("el=>document.activeElement===el")
        page.screenshot(path=str(OUT / "mobile-source.png"))
        page.keyboard.press("Escape")
        assert page.locator("#citation-sheet").is_hidden()
        # A late socket result must not duplicate a result already received by polling.
        page.evaluate("previewSocket.onmessage({data:JSON.stringify({type:'answer',job_id:'preview-job-1',text:'Duplicate'})})")
        assert page.locator(".assistant-message").count() == 1
        scenario["kind"] = "chat"
        page.locator("#query-input").fill("Good night")
        page.locator("#submit-btn").click()
        page.get_by_text("Sleep well. We can pick this up tomorrow.", exact=True).wait_for()
        assert page.locator(".assistant-message").count() == 2
        assert page.locator(".assistant-message").last.locator(".source-check, .confidence-pill, .unverified-box").count() == 0
        results["chat_has_no_verification_warning"] = True
        results["duplicate_completion_ignored"] = True
        # Render controlled time samples to check actual state-specific canvas output.
        animation = page.evaluate("""()=>{
          const canvas=document.createElement('canvas');canvas.id='motion-test';canvas.style.width='124px';canvas.style.height='124px';document.body.appendChild(canvas);
          const orb=new HydroOrbVisualizer('motion-test');const hashes={};
          for(const state of ['idle','thinking','speaking','sleep']){
            orb.setState(state);orb.draw(1);const first=canvas.toDataURL();
            orb.draw(1.7);hashes[state]=first!==canvas.toDataURL();
            orb.draw(1.7);if(state==='thinking')window.thinkingOrb=canvas.toDataURL();
            if(state==='speaking')window.speakingOrb=canvas.toDataURL();
          }
          orb.destroy();canvas.remove();return hashes;
        }""")
        assert animation == {"idle": True, "thinking": True, "speaking": True, "sleep": False}, animation
        results["orb_motion"] = animation
        for state in ("thinking", "speaking"):
            page.evaluate(f"previewSocket.onmessage({{data:JSON.stringify({{type:'state',state:'{state}'}})}})")
            page.wait_for_timeout(220)
            page.screenshot(path=str(OUT / f"mobile-{state}.png"))
        scenario["status"] = "running"
        page.locator("#query-input").fill("Search for redshift")
        page.locator("#submit-btn").click()
        page.locator("#floating-stop-btn").click()
        page.get_by_text("Response stopped.", exact=True).wait_for()
        assert page.locator("#submit-btn").is_enabled()
        page.evaluate("previewSocket.onmessage({data:JSON.stringify({type:'partial_text',job_id:'preview-job-3',text:'Late token'})})")
        assert "Late token" not in page.locator("#chat-feed").inner_text()
        results["cancel_ignores_late_tokens"] = True
        page.emulate_media(color_scheme="dark", reduced_motion="reduce")
        page.wait_for_timeout(100)
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        page.evaluate("""()=>{window.orbDraws=0;const draw=HydroOrbVisualizer.prototype.draw;
          HydroOrbVisualizer.prototype.draw=function(...args){window.orbDraws++;return draw.apply(this,args)}}""")
        page.wait_for_timeout(150)
        assert page.evaluate("window.orbDraws") == 0
        results["reduced_motion_stops_frames"] = True
        page.screenshot(path=str(OUT / "mobile-dark.png"))
        browser.close()
    assert not results["errors"], results["errors"]
    (OUT / "mobile-ui-check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    check()
