"""Deterministic parser for imperative requests ("open Chrome", "create a note about ...").

A 4B model is unreliable at deciding when to call tools, and a regex is instant, so
unambiguous commands are recognised here. Everything else falls through to research or chat.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_POLITE = r"^(?:(?:(?:hey|hi|hello|ok|okay)[, ]+)?(?:lora|laura|lara|lola|lorra)[, ]+)?(?:please |kindly )?(?:(?:can|could|would|will) you (?:please )?|i (?:want|need|'d like) you to )?"

_OPEN = re.compile(
    _POLITE + r"(?P<verb>open|launch|start|run|fire up|bring up|pull up|show me|go to|visit|browse(?: to)?|navigate to|take me to)\s+(?:up\s+)?(?:the\s+|my\s+)?(?P<target>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_NOTE = re.compile(
    _POLITE + r"(?:create|make|write|save|draft|generate|compose|put together)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+|new\s+)*"
    r"(?P<kind>markdown|md|text|txt|note|notes|file|document|doc)\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_SAVE_LAST = re.compile(
    _POLITE + r"(?:save|put|store|write|export)\s+(?:that|this|it|the (?:answer|reply|response|above|last (?:answer|reply)))\s*"
    r"(?:to|in|into|as|to a|as a)?\s*(?:a\s+|an\s+)?(?P<kind>markdown|md|text|txt|note|file)?\b(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_NOTE_THAT = re.compile(
    _POLITE + r"(?:make|take|jot down)\s+(?:a\s+)?note\s+(?:of\s+)?(?:that|this|it)\b(?P<rest>.*)$", re.IGNORECASE
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
_SITE_SEARCH_A = re.compile(
    _POLITE + r"(?:open|launch|go to|visit)\s+(?:the\s+)?(?P<site>[\w .-]{2,30}?)\s+and\s+(?:then\s+)?(?:search|look)(?:\s+for)?\s+(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_SITE_SEARCH_B = re.compile(
    _POLITE + r"(?:search|look up|find)\s+(?:on\s+|in\s+)?(?P<site>youtube|amazon|wikipedia|github|reddit|maps|google maps|stack ?overflow|flipkart|imdb|spotify)"
    r"\s+(?:for\s+)?(?P<q>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
_SITE_SEARCH_C = re.compile(
    _POLITE + r"(?:search|look up|find)\s+(?P<q>.+?)\s+(?:on|in)\s+(?P<site>youtube|amazon|wikipedia|github|reddit|google maps|maps|stack ?overflow|flipkart|imdb)[\s.!?]*$",
    re.IGNORECASE,
)
_PLAY_ON = re.compile(_POLITE + r"play\s+(?P<q>.+?)\s+on\s+(?P<site>youtube)[\s.!?]*$", re.IGNORECASE)
_READ_URL = re.compile(
    _POLITE + r"(?:summari[sz]e|read|tl;?dr|explain|check out|look at)\s+(?:this\s+|the\s+)?(?:page|article|link|url|site|website)?\s*(?P<url>https?://\S+)",
    re.IGNORECASE,
)
_DIAG = re.compile(
    r"^(?:please\s+)?(?:run\s+|do\s+)?(?:a\s+|the\s+)?(?:diagnos(?:e|tics?)|self[- ]?test|self[- ]?check|health ?check|system check|check (?:my|your|the) setup|what'?s wrong(?: with you)?)[\s.!?]*$",
    re.IGNORECASE,
)
_TYPE_INTO = re.compile(
    _POLITE + r"(?:type|write|enter|dictate|put)\s+(?P<text>[\"“'].+?[\"”']|.+?)\s+(?:into|in|on)\s+(?:the\s+|my\s+)?"
    r"(?P<app>notepad|wordpad|word|onenote|vs ?code|visual studio code|text editor)\b[\s.!?]*$",
    re.IGNORECASE | re.DOTALL,
)
_DEEP = re.compile(
    r"\b(deep(?:ly)?|thorough(?:ly)?|in[- ]depth|comprehensive(?:ly)?|really dig|double[- ]check|fact[- ]check|verify)\b", re.IGNORECASE
)
_DOC_EXT = re.compile(r"\.(?:pdf|docx?|txt|md|xlsx?|pptx?|csv|png|jpe?g|gif|zip|mp3|mp4|py|ipynb)$", re.IGNORECASE)
_URL = re.compile(r"^(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/?#]\S*)?$", re.IGNORECASE)
_PATHLIKE = re.compile(r"^(?:[a-z]:[\\/]|\\\\|~[\\/]|/)", re.IGNORECASE)
_TRAIL_WORDS = re.compile(r"\s+(?:app|application|program|website|site|page|folder|directory|please)$", re.IGNORECASE)

KNOWN_FOLDERS = ("downloads", "documents", "desktop", "pictures", "music", "videos")
FOLDER_WORDS = {
    "download": "downloads", "downloads": "downloads", "document": "documents", "documents": "documents",
    "desktop": "desktop", "picture": "pictures", "pictures": "pictures", "photo": "pictures", "photos": "pictures",
    "music": "music", "video": "videos", "videos": "videos",
}
_EXT = {"markdown": "md", "md": "md", "text": "txt", "txt": "txt"}

_RESEARCH_STRONG = re.compile(
    r"\b(research|look\s?up|look into|search(?: the web| online| for| up)|search\b|find (?:out|information|info|sources)|"
    r"browse|on the web|online|sources? (?:on|for)|cite|citations?|with sources)\b",
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
    kind: str  # open | list_dir | note | save_last | ask_ai | google
    target: str = ""
    args: dict = field(default_factory=dict)


def _clean_target(raw: str) -> str:
    t = raw.strip().strip("\"'`“”")
    while True:
        new = _TRAIL_WORDS.sub("", t).strip()
        if new == t:
            return t
        t = new


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

    m = _TYPE_INTO.match(q)
    if m and not re.match(r"^(?:a|an|the)\s+(?:new\s+)?(?:note|file|document|doc|markdown)\b", m.group("text").strip(), re.IGNORECASE):
        text = m.group("text").strip().strip("\"'“”")
        return Command("type_into", target=re.sub(r"\s+", " ", m.group("app").lower()), args={"text": text})

    m = _READ_URL.match(q)
    if m:
        return Command("read_url", target=m.group("url").rstrip(".,)"))

    for pat in (_SITE_SEARCH_A, _SITE_SEARCH_B, _SITE_SEARCH_C, _PLAY_ON):
        m = pat.match(q)
        if m:
            return Command("site_search", target=_clean_target(m.group("site")), args={"query": m.group("q").strip()})

    m = _NOTE_THAT.match(q)
    if m:
        return Command("save_last", args={"ext": "md", "rest": (m.group("rest") or "").strip()})

    m = _SAVE_LAST.match(q)
    if m:
        return Command("save_last", args={"ext": _EXT.get((m.group("kind") or "md").lower(), "md"), "rest": (m.group("rest") or "").strip()})

    m = _ASK_AI.match(q)
    if m:
        service = re.sub(r"\s+", "", m.group("service").lower())
        return Command("ask_ai", args={"service": service, "prompt": m.group("prompt").strip()})

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
    # Time-sensitive words only count for real questions, not small talk.
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
    "- **Research:** ask me to research anything. I search the web, read the pages, check the facts and show my sources.\n"
    "- **Open things:** apps, folders, files and websites (\"open Chrome\", \"open my Downloads folder\", \"open resume.pdf\").\n"
    "- **Write:** create .md or .txt notes, and type into Notepad, Word or VS Code (I'll ask you first).\n"
    "- **Your files:** use anything you add under Add sources.\n"
    "- **Go deeper:** say \"deep research on…\" and I'll get a second opinion and improve the answer.\n"
    "- **Talk:** hold Space, press the mic, or say \"Hey LORA\"."
)
_VERB_REPLY = {
    "browse": "Yes. Ask me to research something and I'll search the web, read the pages, check the facts and show you my sources. I can also open a site for you, like \"open YouTube\".",
    "search": "Yes. Say \"research\" and what you want to know, and I'll search the web and read the pages. For a quick look in your browser, say \"search YouTube for…\" or \"google…\".",
    "open": "Yes. I can open apps, folders, files and websites. Try \"open Chrome\", \"open my Downloads folder\" or \"open resume.pdf\".",
    "type": "Yes. I can create .md or .txt notes, and type into Notepad, Word or VS Code. I'll show you the text and ask first.",
    "read": "Yes. I can read files in your Documents, Downloads and Desktop folders, plus anything you add under Add sources.",
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
