"""Deterministic parser for imperative requests ("open Chrome", "open wikipedia on ...", "open chatgpt and ask ...").

Handles unambiguous desktop, web-app, site-topic, external-AI, model-switch, skill-switch, depth-rewrite,
notebook, and power/stop intents while allowing compound or multi-step requests to flow into the multi-agent harness.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_POLITE = r"^(?:(?:(?:hey|hi|hello|ok|okay)[, ]+)?(?:lora|laura|lara|lola|lorra)[, ]+)?(?:please |kindly )?(?:(?:can|could|would|will) you (?:please )?|i (?:want|need|'d like) you to )?"

_OPEN = re.compile(
    _POLITE + r"(?P<verb>open|launch|start|run|activate|fire up|bring up|pull up|show me|go to|visit|browse(?: to)?|navigate to|take me to)\s+(?:up\s+)?(?:the\s+|my\s+)?(?P<target>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_NOTE = re.compile(
    _POLITE + r"(?:create|make|write|save|draft|generate|compose|put together)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+|new\s+)*"
    r"(?P<kind>markdown|md|text|txt|note|notes|file|document|doc|python|py|script|html|csv|json|svg)\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_IMAGE_GEN = re.compile(
    _POLITE + r"(?:create|make|generate|draw|plot|render|design)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+|new\s+)*"
    r"(?P<kind>image|picture|diagram|chart|plot|graphic|svg|illustration|poster|visual)\s+(?:of|for|about|showing|titled|called)?\s*(?P<prompt>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SAVE_NOTEBOOK = re.compile(
    _POLITE + r"(?:save|promote|store|add)\s+(?:this|that|it|the\s+(?:research|answer|session))\s+(?:to|in|into|as)\s+(?:my\s+|a\s+)?notebook(?:\s+(?:called|named|titled)\s+(?P<title>.+?))?[\s.!?]*$",
    re.IGNORECASE,
)
_LIST_NOTEBOOKS = re.compile(
    _POLITE + r"(?:list|show)\s+(?:me\s+)?(?:my\s+|all\s+)?(?:saved\s+)?notebooks[\s.!?]*$",
    re.IGNORECASE,
)
_REWRITE_DEPTH = re.compile(
    _POLITE + r"(?:(?:make|explain|rewrite|rephrase)\s+(?:it|that|this|the\s+answer)\s+(?:more\s+)?(?P<depth1>eli5|like\s+i'?m\s+5|simpler|simple|standard|detailed|in\s+detail|expert|at\s+expert\s+level)|"
    r"(?:explain\s+like\s+i'?m\s+5|eli5))[\s.!?]*$",
    re.IGNORECASE,
)
_POWER_SLEEP = re.compile(
    _POLITE + r"(?:go\s+to\s+sleep|sleep(?:\s+mode)?|stand\s*by)[\s.!?]*$",
    re.IGNORECASE,
)
_POWER_OFF = re.compile(
    _POLITE + r"(?:turn\s+off|shut\s*down(?:\s+lora)?|exit\s+lora|quit\s+lora)[\s.!?]*$",
    re.IGNORECASE,
)
_SAVE_LAST = re.compile(
    _POLITE + r"(?:save|put|store|write|export)\s+(?:that|this|it|the (?:answer|reply|response|above|last (?:answer|reply)))\s*"
    r"(?:to|in|into|as|to a|as a)?\s*(?:a\s+|an\s+)?(?P<kind>markdown|md|text|txt|note|file)?\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_NOTE_THAT = re.compile(
    _POLITE + r"(?:make|take|jot down)\s+(?:a\s+)?note\s+(?:of\s+)?(?:that|this|it)\b(?P<rest>.*)$", re.IGNORECASE
)
_OPEN_AND_ASK_AI = re.compile(
    _POLITE + r"(?:open|launch|start|go to|visit|use)\s+(?:up\s+)?(?:the\s+)?(?P<service>chat ?gpt|claude|gemini|perplexity)\s+"
    r"(?:and\s+)?(?:then\s+)?(?:please\s+)?(?:ask|tell|message|prompt|type|search|query)(?:\s+(?:it|them|for|about|to|that))?\s*(?:to|that|about|:|,)?\s+(?P<prompt>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_ASK_AI = re.compile(
    _POLITE + r"(?:ask|tell|message|prompt|type (?:this )?(?:into|in to|to))\s+(?P<service>chat ?gpt|claude|gemini|perplexity)\s*(?:to|that|about|:|,)?\s+(?P<prompt>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_GOOGLE = re.compile(
    _POLITE + r"(?:(?:open )?google|search google for|google search(?: for)?|search on google for|open google and search(?: for)?)\s+(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_LIST = re.compile(
    _POLITE + r"(?:list|show)\s+(?:me\s+)?(?:the\s+)?(?:files|contents|what'?s)\s+(?:in|inside|of|on)\s+(?:the\s+|my\s+)?(?P<target>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_SITE_NAMES = r"youtube|you ?tube|amazon|wikipedia|wikimedia|wiki ?pedia|wiki|arxiv|ar ?xive|github|git ?hub|reddit|maps|google maps|stack ?overflow|flipkart|imdb|spotify|perplexity|google"

# Matches: "open youtube and search for lofi beats", "open wikipedia and look up alan turing"
_SITE_SEARCH_A = re.compile(
    _POLITE + r"(?:open|launch|start|activate|go to|visit)\s+(?:the\s+)?(?P<site>[\w .-]{2,30}?)\s+and\s+(?:then\s+)?(?:search|look(?:\s+up)?|find|query)(?:\s+(?:for|about|on))?\s+(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
# Matches: "open wikipedia on <page>", "open wikipedia about <topic>", "open youtube for <query>", "open arxiv on <paper>"
_SITE_OPEN_ON = re.compile(
    _POLITE + r"(?:open|launch|start|activate|go to|visit|pull up|bring up|show me)\s+(?:up\s+)?(?:the\s+)?(?P<site>" + _SITE_NAMES + r")\s+"
    r"(?:(?:page|article|video|entry|repo|paper|results?)\s+)?(?:on|about|for|in|regarding|titled|called)\s+(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
# Matches: "open <page/topic> on wikipedia", "open <video> on youtube", "open <paper> on arxiv"
_SITE_OPEN_REVERSE = re.compile(
    _POLITE + r"(?:open|launch|start|activate|go to|visit|pull up|bring up|show me)\s+(?:up\s+)?(?:the\s+)?(?:(?:page|article|video|entry|paper)\s+(?:on|about|for)\s+)?(?P<q>.+?)\s+"
    r"(?:on|in)\s+(?P<site>" + _SITE_NAMES + r")(?:\s+(?:page|site|website|app))?[\s.!?]*$",
    re.IGNORECASE,
)
_SITE_SEARCH_B = re.compile(
    _POLITE + r"(?:search|look up|find)\s+(?:on\s+|in\s+)?(?P<site>" + _SITE_NAMES + r")"
    r"\s+(?:for\s+|about\s+|on\s+)?(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_SITE_SEARCH_C = re.compile(
    _POLITE + r"(?:search|look up|find)\s+(?:for\s+)?(?P<q>.+?)\s+(?:on|in)\s+(?P<site>" + _SITE_NAMES + r")[\s.!?]*$",
    re.IGNORECASE,
)
_PLAY_ON = re.compile(_POLITE + r"play\s+(?P<q>.+?)\s+on\s+(?P<site>youtube|spotify)[\s.!?]*$", re.IGNORECASE)
_READ_URL = re.compile(
    _POLITE + r"(?:summari[sz]e|read|tl;?dr|explain|check out|look at)\s+(?:this\s+|the\s+)?(?:page|article|link|url|site|website)?\s*(?P<url>https?://\S+)",
    re.IGNORECASE,
)
_DIAG = re.compile(
    r"^(?:please\s+)?(?:run\s+|do\s+)?(?:a\s+|the\s+)?(?:diagnos(?:e|tics?)|self[- ]?test|self[- ]?check|health ?check|system check|check (?:my|your|the) setup|what'?s wrong(?: with you)?)[\s.!?]*$",
    re.IGNORECASE,
)
_SWITCH_MODEL = re.compile(
    _POLITE + r"(?:switch|change|set|use)\s+(?:(?:the|your|my)\s+)?model\s+(?:to\s+)?(?P<model>[\w.:/-]+)[\s.!?]*$",
    re.IGNORECASE,
)
_SWITCH_VOICE = re.compile(
    _POLITE + r"(?:switch|change|set|use)\s+(?:(?:the|your|my)\s+)?voice\s+(?:to\s+)?(?P<voice>[\w_-]+)[\s.!?]*$",
    re.IGNORECASE,
)
_SWITCH_SKILL = re.compile(
    r"^(?:/skill\s+(?P<slash_skill>[\w-]+)(?:\s+(?P<slash_rest>.*))?|"
    + _POLITE
    + r"(?:use|activate|enable|switch to)\s+(?:the\s+)?(?P<skill>[\w-]+)\s+skill)[\s.!?]*$",
    re.IGNORECASE | re.DOTALL,
)
_TYPE_INTO = re.compile(
    _POLITE + r"(?:type|write|enter|dictate|put)\s+(?P<text>[\"“'].+?[\"”']|.+?)\s+(?:into|in|on)\s+(?:the\s+|my\s+)?"
    r"(?P<app>notepad|wordpad|word|onenote|vs ?code|visual studio code|text editor)\b[\s.!?]*$",
    re.IGNORECASE | re.DOTALL,
)
_DEEP = re.compile(
    r"\b(deep(?:ly)?|thorough(?:ly)?|in[- ]depth|comprehensive(?:ly)?|really dig|double[- ]check|fact[- ]?check|verify|8[- ]?way|outer[- ]?loop)\b", re.IGNORECASE
)
_DOC_EXT = re.compile(r"\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|png|jpe?g|gif|svg|zip|mp3|mp4|py|html|json|ipynb)$", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/?#]\S*)?$", re.IGNORECASE)
_PATHLIKE = re.compile(r"^(?:[a-z]:[\\/]|\\\\|~[\\/]|/)", re.IGNORECASE)
_TRAIL_WORDS = re.compile(r"\s+(?:app|application|program|website|site|page|folder|directory|please)$", re.IGNORECASE)

KNOWN_FOLDERS = ("downloads", "documents", "desktop", "pictures", "music", "videos")
FOLDER_WORDS = {
    "download": "downloads", "downloads": "downloads", "document": "documents", "documents": "documents",
    "desktop": "desktop", "picture": "pictures", "pictures": "pictures", "photo": "pictures", "photos": "pictures",
    "music": "music", "video": "videos", "videos": "videos",
}
_EXT = {
    "markdown": "md", "md": "md", "text": "txt", "txt": "txt",
    "python": "py", "py": "py", "script": "py", "html": "html",
    "csv": "csv", "json": "json", "svg": "svg",
}

_RESEARCH_STRONG = re.compile(
    r"\b(research|look\s?up|look into|search(?: the web| online| for| up)|search\b|find (?:out|information|info|sources)|"
    r"browse|on the web|online|sources? (?:on|for)|cite|citations?|with sources|fact[- ]?check|verify|8[- ]?way|outer[- ]?loop)\b",
    re.IGNORECASE,
)
_RESEARCH_WEAK = re.compile(
    r"\b(latest|newest|recent(?:ly)?|news|current(?:ly)?|today'?s|this (?:week|month|year)|who (?:is|was|won|are)|"
    r"what (?:happened|is happening)|how much (?:is|does|do)|price of|release date|compare|versus|statistics|"
    r"reviews? of)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(r"\b(you|your|yours|yourself|lora)\b", re.IGNORECASE)
_RESEARCH_EXPLICIT = re.compile(r"\b(research|look\s?up|look into|search for|search up|find out|find (?:information|info|sources) (?:on|about))\b", re.IGNORECASE)


_GREETING = re.compile(
    r"^(?:(?:hey|hi|hello|yo|hiya|sup|hola|good (?:morning|afternoon|evening))(?:\s+there)?(?:\s+lora)?|lora|hey lora|hi lora|hello lora|"
    r"you there|are you there|wake up)[\s,.!?]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(?:thanks|thank you|thx|cheers|ty|nice one|great job|awesome|perfect)(?:\s+lora)?[\s,.!?]*$", re.IGNORECASE)


@dataclass
class Command:
    kind: str  # open | list_dir | note | save_last | ask_ai | google | site_search | switch_model | switch_voice | switch_skill | generate_image | rewrite_depth | save_notebook | list_notebooks | sleep | turn_off
    target: str = ""
    args: dict = field(default_factory=dict)


def _clean_target(raw: str) -> str:
    t = raw.strip().strip("\"'`“”")
    while True:
        new = _TRAIL_WORDS.sub("", t).strip()
        if new == t:
            return t
        t = new


def _normalize_site_key(site: str) -> str:
    s = re.sub(r"\s+", "", _clean_target(site).lower())
    aliases = {
        "wiki": "wikipedia",
        "wikimedia": "wikipedia",
        "wikipedia": "wikipedia",
        "arxive": "arxiv",
        "googlemaps": "maps",
        "googlesearch": "google",
    }
    return aliases.get(s, _clean_target(site))


def normalize_speech(text: str) -> str:
    """Whisper adds commas and question marks ("Can You Open, VSCORE?"); the patterns expect plain words."""
    t = (text or "").strip()
    t = re.sub(r"[,;:!?]+", " ", t)
    t = re.sub(r"(?<!\w)\.+(?!\w)", " ", t)  # stray periods / ellipses; dots inside words, URLs and filenames stay
    t = re.sub(r"\.\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_command(text: str) -> Command | None:
    """Return a Command for an unambiguous imperative request, else None (also for spoken, punctuated text)."""
    raw = (text or "").strip()
    for variant in dict.fromkeys((raw, normalize_speech(raw))):
        cmd = _parse(variant)
        if cmd is not None:
            return cmd
    return None


def _parse(text: str) -> Command | None:
    q = (text or "").strip()
    if not q or len(q) > 600:
        return None

    if _DIAG.match(q):
        return Command("diagnose")

    if _POWER_SLEEP.match(q):
        return Command("sleep")

    if _POWER_OFF.match(q):
        return Command("turn_off")

    m = _REWRITE_DEPTH.match(q)
    if m:
        raw_d = (m.group("depth1") or "eli5").strip().lower()
        if "5" in raw_d or "eli5" in raw_d:
            d = "eli5"
        elif "simpl" in raw_d:
            d = "simple"
        elif "detail" in raw_d:
            d = "detailed"
        elif "expert" in raw_d:
            d = "expert"
        else:
            d = "standard"
        return Command("rewrite_depth", target=d)

    m = _SAVE_NOTEBOOK.match(q)
    if m:
        return Command("save_notebook", target=(m.group("title") or "").strip())

    if _LIST_NOTEBOOKS.match(q):
        return Command("list_notebooks")

    m = _SWITCH_MODEL.match(q)
    if m:
        return Command("switch_model", target=m.group("model").strip())

    m = _SWITCH_VOICE.match(q)
    if m:
        return Command("switch_voice", target=m.group("voice").strip().lower())

    m = _SWITCH_SKILL.match(q)
    if m:
        sk = (m.group("slash_skill") or m.group("skill") or "").strip().lower()
        rest = (m.group("slash_rest") or "").strip()
        if sk and not rest:
            return Command("switch_skill", target=sk)

    m = _TYPE_INTO.match(q)
    if m and not re.match(r"^(?:a|an|the)\s+(?:new\s+)?(?:note|file|document|doc|markdown)\b", m.group("text").strip(), re.IGNORECASE):
        text_val = m.group("text").strip().strip("\"'“”")
        return Command("type_into", target=re.sub(r"\s+", " ", m.group("app").lower()), args={"text": text_val})

    m = _READ_URL.match(q)
    if m:
        return Command("read_url", target=m.group("url").rstrip(".,)"))

    # Compound "open chatgpt and ask <query>" MUST be checked BEFORE generic _OPEN!
    m = _OPEN_AND_ASK_AI.match(q)
    if m:
        service = re.sub(r"\s+", "", m.group("service").lower())
        return Command("ask_ai", args={"service": service, "prompt": m.group("prompt").strip()})

    m = _ASK_AI.match(q)
    if m:
        service = re.sub(r"\s+", "", m.group("service").lower())
        return Command("ask_ai", args={"service": service, "prompt": m.group("prompt").strip()})

    # Site-specific topic navigation ("open wikipedia on <page>", "open <page> on wikipedia", etc.)
    for pat in (_SITE_SEARCH_A, _SITE_OPEN_ON, _SITE_OPEN_REVERSE, _SITE_SEARCH_B, _SITE_SEARCH_C, _PLAY_ON):
        m = pat.match(q)
        if m:
            site = _normalize_site_key(m.group("site"))
            query = m.group("q").strip().strip("\"'“”")
            if site and query:
                return Command("site_search", target=site, args={"query": query})

    m = _NOTE_THAT.match(q)
    if m:
        return Command("save_last", args={"ext": "md", "rest": (m.group("rest") or "").strip()})

    m = _SAVE_LAST.match(q)
    if m:
        return Command("save_last", args={"ext": _EXT.get((m.group("kind") or "md").lower(), "md"), "rest": (m.group("rest") or "").strip()})

    m = _IMAGE_GEN.match(q)
    if m:
        return Command("generate_image", target=m.group("prompt").strip(), args={"kind": m.group("kind").lower()})

    m = _NOTE.match(q)
    if m:
        kind = m.group("kind").lower()
        return Command("note", args={"ext": _EXT.get(kind, "md"), "instruction": q, "rest": (m.group("rest") or "").strip()})

    m = _GOOGLE.match(q)
    if m:
        return Command("google", target=m.group("q").strip())

    m = _LIST.match(q)
    if m:
        return Command("list_dir", target=_clean_target(m.group("target")))

    m = _OPEN.match(q)
    if m:
        target = _clean_target(m.group("target"))
        # Do not greedily swallow multi-clause agentic instructions as an app name
        if re.search(r"\b(?:and\s+(?:then\s+)?(?:ask|tell|write|summarize|read|create|make|save|explain|find|check))\b", target, re.IGNORECASE):
            return None
        if target and len(target.split()) <= 8:
            return Command("open", target=target, args={"verb": m.group("verb").lower()})
    return None


_ABOUT_LORA = re.compile(
    r"\b(?:you (?:mean|said|say|told|can'?t|cannot|couldn'?t|won'?t|aren'?t|are not|didn'?t|did not|just)|"
    r"your (?:answer|reply|response|abilities|capabilities|model|sources|mistake|last))\b",
    re.IGNORECASE,
)


def is_research(text: str) -> bool:
    """True when the question needs fresh information from the web."""
    q = normalize_speech(text)
    if len(q) < 8:
        return False
    ability = _CAN_YOU.match(q)
    if (ability and _GENERIC_OBJECT.match(ability.group("rest"))) or _ABOUT_LORA.search(q):
        return False  # "can you browse the internet", "you can't browse": about LORA, not a request to search
    if _RESEARCH_STRONG.search(q):
        return True
    return bool(_RESEARCH_WEAK.search(q)) and len(q.split()) >= 4


def quick_reply(text: str, now: datetime | None = None) -> str | None:
    """Instant canned answers for greetings/thanks so small talk never waits on a model."""
    q = normalize_speech(text)
    now = now or datetime.now()
    if _GREETING.match(q):
        hour = now.hour
        part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        return random.choice(
            [
                f"Good {part}! What are we working on?",
                "Hey! I'm here. What do you need?",
                f"Good {part}. Ready when you are.",
                "Hey there. What can I do for you?",
            ]
        )
    if _THANKS.match(q):
        return random.choice(["Anytime.", "Happy to help.", "You got it.", "Of course."])
    return None


def resolve_folder(target: str) -> Path | None:
    """Map 'downloads' / 'my documents folder' / a path to a concrete path (no validation)."""
    from lora.tools.files.known import known_folder

    t = target.strip().strip("\"'")
    low = t.lower()
    if low in FOLDER_WORDS:
        return known_folder(FOLDER_WORDS[low])
    if _PATHLIKE.match(t):
        return Path(t).expanduser()
    return None


def looks_like_url(target: str) -> bool:
    t = target.strip()
    return bool(_URL.match(t)) and not _DOC_EXT.search(t)


def looks_like_filename(target: str) -> bool:
    """'resume.pdf', 'budget 2026.xlsx': a bare file name (no folders) to search for."""
    t = target.strip().strip("\"'")
    return bool(_DOC_EXT.search(t)) and not _PATHLIKE.match(t) and "/" not in t and "\\" not in t


def note_title(instruction: str, explicit: str = "") -> str:
    """Pick a readable title for a note from 'called X' / 'named X' / 'about X'."""
    for pat in (r"(?:called|named|titled)\s+[\"“']?(?P<t>[^\"”'\n,.]{2,60})", r"\babout\s+(?P<t>[^\n,.]{2,60})", r"\bon\s+(?P<t>[^\n,.]{2,60})"):
        m = re.search(pat, instruction, re.IGNORECASE)
        if m:
            return m.group("t").strip()
    return explicit or "Note " + datetime.now().strftime("%Y-%m-%d %H%M")


def is_deep(text: str) -> bool:
    """'deep research on…', 'thoroughly check…': ask for the slow, double-checked mode."""
    return bool(_DEEP.search(normalize_speech(text)))


_CAN_YOU = re.compile(
    r"^(?:so |hey |ok |okay |lora )*(?:can|could|will|would|do|are) you (?:actually |really |also |even |ever )*(?:be able to |able to )?"
    r"(?P<verb>browse|search|surf|look up|open|launch|type|write|read|access|see|hear|listen|speak|talk|remember|create|save|control|use|do)\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_GENERIC_OBJECT = re.compile(
    r"^\s*(?:the |my |a |any )?(?:internet|web|net|online|files?|folders?|apps?|applications?|programs?|websites?|web pages?|pages?|"
    r"things|stuff|anything|that|this|it|properly|too|now|yet)?\s*$",
    re.IGNORECASE,
)
_WHAT_CAN = re.compile(
    r"^(?:what (?:can|do) you (?:do|know how to do)|what are you able to do|what are your (?:capabilities|abilities|skills|features)|"
    r"how can you help(?: me)?|what can lora do)$",
    re.IGNORECASE,
)
CAPABILITIES = (
    "Here's what I can do:\n"
    "- **Laya Decision Router & Multi-Agent Research:** ask me to research anything. Laya routes your intent on CPU, fetches web & scholarly sources, runs an 8-stage verification & self-correction loop (V1–V8), and displays the Verification Scorecard.\n"
    "- **Smart Navigation & External AI:** open apps, folders, files, specific site pages (\"open Wikipedia on Alan Turing\"), or delegate to external AIs (\"open ChatGPT and ask...\").\n"
    "- **Files & Visual Studio:** paste or attach images and files directly in chat, or ask me to generate code, notes, SVG diagrams, charts, and edited images.\n"
    "- **Depth Rewrite & Notebooks:** say \"explain like I'm 5\", \"make it more detailed\", or \"save this to my notebook\".\n"
    "- **Skills, Models & Voice (Kokoro TTS `af_sky`):** switch models manually or use Auto smart routing, activate skills from the top bar, and use voice or mobile LAN pairing."
)
_VERB_REPLY = {
    "browse": "Yes. Ask me to research something and I'll search the web, read the pages, check the facts and show you my sources. I can also open a site or topic for you, like \"open Wikipedia on quantum computing\".",
    "search": "Yes. Say \"research\" and what you want to know, and I'll search the web and read the pages. For a quick look in your browser, say \"open Wikipedia on…\" or \"search YouTube for…\".",
    "open": "Yes. I can open apps, folders, files and websites. Try \"open Chrome\", \"open Wikipedia on Alan Turing\", \"open my Downloads folder\" or \"open resume.pdf\".",
    "type": "Yes. I can create code, Markdown, text, or visual files, and type into Notepad, Word or VS Code. I'll show you the text and ask first.",
    "read": "Yes. You can paste/attach images and files directly into the chat bar, or ask me to read files in Documents, Downloads, Desktop, or your indexed sources.",
    "hear": "Yes. Hold Space, press the mic, or say \"Hey LORA\".",
}
for _alias, _target in (("surf", "browse"), ("look up", "search"), ("access", "browse"), ("launch", "open"), ("write", "type"),
                        ("listen", "hear"), ("speak", "hear"), ("talk", "hear")):
    _VERB_REPLY[_alias] = _VERB_REPLY[_target]


def capability_reply(text: str) -> str | None:
    """Instant, truthful answers to "can you browse the internet?" / "what can you do?"."""
    q = normalize_speech(text)
    if _WHAT_CAN.match(q):
        return CAPABILITIES
    m = _CAN_YOU.match(q)
    if m and _GENERIC_OBJECT.match(m.group("rest")):
        return _VERB_REPLY.get(m.group("verb").lower(), CAPABILITIES)
    return None
