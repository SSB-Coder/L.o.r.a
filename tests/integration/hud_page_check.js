// Executes src/lora/ui/droplet.html in jsdom (fake WebGL) and prints a JSON summary.
// Needs node + `npm i jsdom` (the pytest wrapper skips itself when they are missing).
const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push('jsdomError: ' + (e.detail && e.detail.stack || e.message)));
vc.on('error', e => errors.push('console.error: ' + e));

const glProxy = () => new Proxy({}, { get: (t, k) => (k in t ? t[k] : (typeof k === 'string' && /^[A-Z_0-9]+$/.test(k) ? 1 : () => ({}))), set: (t, k, v) => (t[k] = v, true) });
const rafQ = []; let drawCalls = 0;

const dom = new JSDOM(html, {
  runScripts: 'dangerously', pretendToBeVisual: false, virtualConsole: vc,
  url: 'file:///hud/droplet.html',
  beforeParse(w) {
    Object.defineProperty(w.navigator, 'userAgent', { get: () => 'Mozilla/5.0 QtWebEngine/6.8' });
    w.HTMLCanvasElement.prototype.getContext = function () {
      const gl = glProxy(); gl.drawArrays = () => { drawCalls++; };
      gl.getShaderParameter = () => true; gl.getProgramParameter = () => true;
      return gl;
    };
    w.ResizeObserver = class { constructor(cb) { this.cb = cb; } observe() { } disconnect() { } };
    w.requestAnimationFrame = cb => { rafQ.push(cb); return rafQ.length; };
    Object.defineProperty(w.HTMLElement.prototype, 'clientWidth', { get() { return 800; } });
    Object.defineProperty(w.HTMLElement.prototype, 'clientHeight', { get() { return 600; } });
    w.matchMedia = w.matchMedia || (() => ({ matches: false, addListener() { }, addEventListener() { } }));
  },
});
const w = dom.window, d = w.document;
const step = (n, t0 = 1000, dt = 16) => { for (let i = 0; i < n; i++) { const q = rafQ.splice(0); q.forEach(cb => cb(t0 + i * dt)); } };
const $ = s => d.querySelector(s), $$ = s => Array.from(d.querySelectorAll(s));

const sent = [];
w.notifyPython = (a, p) => sent.push([a, p]);   // global function declaration: calls resolve through window
const key = (type, code, extra = {}) => { const e = new w.KeyboardEvent(type, { code, key: code === 'Space' ? ' ' : code, bubbles: true, cancelable: true, ...extra }); w.dispatchEvent(e); return e; };
const out = {};

step(5);
out.render_loop_ran = drawCalls > 0;
out.starts_empty = d.body.classList.contains('empty');

// ---- hold Space with OS key-repeat: exactly ONE press and ONE release
key('keydown', 'Space'); for (let i = 0; i < 25; i++) key('keydown', 'Space', { repeat: true }); key('keyup', 'Space');
out.ptt_events = sent.filter(s => s[0].startsWith('on_ptt')).map(s => s[0]);
w.setDropletState('idle');

// ---- Space while typing must not start PTT
sent.length = 0; $('#q').focus(); key('keydown', 'Space'); key('keyup', 'Space');
out.space_while_typing_events = sent.length; $('#q').blur(); d.body.focus();

// ---- typing a message: bubble + pending reply immediately, one event to Python, input locked while thinking
$('#q').value = 'what is **up**?'; sent.length = 0;
$('#cmd').dispatchEvent(new w.Event('submit', { cancelable: true, bubbles: true }));
out.submit_events = sent.filter(s => s[0] === 'on_user_submit').map(s => s[1]);
out.user_bubbles_after_submit = $$('.msg.user').length;
out.pending_shown = !!$('.msg.bot .dots');
out.state_after_submit = d.body.dataset.state;
out.chat_mode_on = !d.body.classList.contains('empty');
out.send_disabled_while_thinking = $('#send').disabled && $('#mic').disabled;
out.input_value_cleared = $('#q').value === '';

// ---- Python's beginStream for the same question reuses the bubble (no duplicate)
w.beginStream('what is **up**?', '');
out.user_bubbles_after_begin = $$('.msg.user').length;
out.bot_messages_after_begin = $$('.msg.bot').length;

// ---- progress line, then streamed markdown
w.setStatus('Searching the web…'); out.status_text = $('.msg.bot .dots').textContent;
const md = 'Intro with **bold** and *slanted* and `code` and [a link](https://example.org/x?a=1&b=2) and [bad](javascript:alert(1)).\n\n## Heading\n- one [S1]\n- two\n\n1. first\n\n```\n<img src=x onerror=alert(1)>\n```\nraw <img src=x onerror=alert(2)> here';
w.streamUpdate(md);
const body = $('.msg.bot .body');
out.md_bold = !!body.querySelector('strong') && body.querySelector('strong').textContent === 'bold';
out.md_code = !!body.querySelector('p code');
out.md_italic = !!body.querySelector('em');
out.md_list = body.querySelectorAll('ul li').length === 2 && body.querySelectorAll('ol li').length === 1;
out.md_heading = !!body.querySelector('h4');
out.md_good_link = !!body.querySelector('a[data-url="https://example.org/x?a=1&b=2"]');
out.md_no_js_link = !body.querySelector('a[data-url^="javascript"]') && !body.innerHTML.includes('href="javascript');
out.no_injected_img = body.querySelector('img') === null;
out.pre_has_text_not_html = body.querySelector('pre code').textContent.includes('<img src=x');

// ---- finish (streamed): sources dropdown + citation click opens the source; JS does not force a state
sent.length = 0;
w.finishTurn('what is **up**?', md, [{ id: 'S1', title: 'Example page', url: 'https://example.org/p', domain: 'example.org' }], 0, '', true);
out.state_after_finish = d.body.dataset.state;                     // Python owns the state change
const det = $('.msg.bot details.src');
out.sources_summary = det && det.querySelector('summary').textContent;
body.querySelector('sup.c').click();
det.querySelector('a[data-url]').click();
out.open_url_events = sent.filter(s => s[0] === 'on_open_url').map(s => s[1].url);
w.setDropletState('idle');
out.state_after_python_idle = d.body.dataset.state; out.send_enabled_again = !$('#send').disabled;

// ---- verification line under a checked answer (quiet; expands only when there are notes)
w.beginStream('check me', ''); 
w.finishTurn('check me', 'Answer [S1].', [], 0, '', true, { checked: { supported: 2, total: 3, unverified: ['Mamba was invented in 1962'] }, roundtable: { decision: 'revise', independent: true, changes: ['Reviewer: add the date'] }, improved: true });
const chk = $$('.msg.bot').pop().querySelector('.chk');
out.check_line = chk && chk.querySelector('summary').textContent;
out.check_notes = chk ? Array.from(chk.querySelectorAll('.s')).map(e => e.textContent) : [];
w.beginStream('clean', ''); w.finishTurn('clean', 'ok', [], 0, '', true, { checked: { supported: 4, total: 4, unverified: [] } });
const clean = $$('.msg.bot').pop().querySelector('.chk');
out.clean_line = clean && clean.textContent; out.clean_is_plain = clean && clean.tagName === 'DIV';
w.beginStream('none', ''); w.finishTurn('none', 'no meta', [], 0, '', true, {});
out.no_meta_no_line = !$$('.msg.bot').pop().querySelector('.chk');
w.setDropletState('idle');

// ---- late tokens after finish are ignored; system notices; "You" echo ignored
w.streamUpdate('LATE'); out.late_update_ignored = !d.body.textContent.includes('LATE');
const before = $$('.notice').length; w.addMessage('You', 'echo'); w.addMessage('L.O.R.A', 'Microphone problem'); 
out.notice_added = $$('.notice').length === before + 1 && !d.body.textContent.includes('echo');

// ---- voice-initiated turn (nothing typed here): Python's beginStream creates bubble + reply
w.beginStream('open chrome', '');
out.voice_turn_bubble = $$('.msg.user').pop().textContent === 'open chrome';
w.finishTurn('open chrome', 'Opening Google Chrome.', [], 0, '', true); w.setDropletState('idle');

// ---- mic status + level meter
w.setMicStatus(true, 'Realtek Mic'); out.mic_ok = $('#micst').className + '|' + $('#mictx').textContent + '|' + $('#micst').title;
w.setMicStatus(false, 'no input device found'); out.mic_bad = $('#micst').className + '|' + $('#mictx').textContent;

// ---- same-state calls are no-ops
w.setDropletState('listen'); $('#hint').textContent = 'SENTINEL'; w.setDropletState('listen');
out.same_state_noop = $('#hint').textContent === 'SENTINEL'; w.setDropletState('idle');

// ---- voice toggle + sleep
$('#morebtn').click(); out.more_menu_open = !$('#moremenu').hidden; sent.length = 0; $('#vo').click(); out.more_menu_closes_after_pick = $('#moremenu').hidden; out.voice_toggle = sent.filter(s => s[0] === 'on_voice_toggle').map(s => s[1]); out.voice_label = $('#vo').textContent;

// ---- "Add sources" menu: separate dropdown, no embedding-model wording anywhere
w.updateMediaList([{ path: 'C:/x/a.pdf', filename: 'a.pdf', chunk_count: 3, file_id: '1' }]);
out.srcbtn_label = $('#srcbtn').textContent;
out.menu_hidden_initially = $('#srcmenu').hidden;
$('#srcbtn').click(); out.menu_open = !$('#srcmenu').hidden;
sent.length = 0; $('#m-browse').click(); $('#media-list .m-del').click();
out.menu_events = sent.map(s => s[0]);
d.body.click(); out.menu_closes_on_outside_click = $('#srcmenu').hidden;
w.updateMediaList([]); out.srcbtn_label_empty = $('#srcbtn').textContent;
out.no_model_words = !/gemma|faiss|embedding|ollama|spark|ornith/i.test(d.documentElement.outerHTML);
out.no_remote_resources = !/(?:src|href)=["']https?:\/\//.test(d.documentElement.outerHTML);

// ---- WebGL context loss / restore does not throw and rendering resumes
const cv = $('#orb'); drawCalls = 0;
cv.dispatchEvent(new w.Event('webglcontextlost', { cancelable: true })); step(5, 3000); out.draws_while_lost = drawCalls;
cv.dispatchEvent(new w.Event('webglcontextrestored')); step(5, 4000); out.draws_after_restore = drawCalls > 0;

out.errors = errors;
console.log(JSON.stringify(out, null, 1));
process.exit(0);
