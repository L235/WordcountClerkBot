#!/usr/bin/env python3
"""
WordcountClerkBot – stable 2.1 (May 2025)
========================================
Polish & correctness tweaks:
* **Always show an ARC section** – even when no open requests.
* **Heading links** – each per‑request sub‑table heading now links directly to
  the request (`[[Page#Anchor|Heading]]`).
* **Placeholder filter** – rows from the `{other-editor}` template are now
  suppressed.
* **New status** – rows ≤10 % over the limit are flagged *within 10 %* and
  highlighted yellow (`#ffffcc`).
* **AE pseudo‑section regex** updated to match the explicit
  `[[WP:DIFF|Diffs]] of edits …` wording.
* **Closed requests** – if a level‑2 section is fully wrapped in `{{hat …
  |…}}…{{hab}}`, we mark the heading with *“(closed)”*.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import ClassVar, List

import mwclient
import mwparserfromhell as mwpfh
import requests

###############################################################################
# Configuration                                                               #
###############################################################################

SETTINGS_PATH = "settings.json"
DEFAULT_CFG = {
    "site": "en.wikipedia.org",
    "path": "/w/",
    "user": "BotUser@PasswordName",
    "bot_password": "",
    "ua": "WordcountClerkBot/2.1 (https://github.com/L235/WordcountClerkBot)",
    "cookie_path": "~/wordcountclerkbot/cookies.txt",
    "arca_page": "Wikipedia:Arbitration/Requests/Clarification and Amendment",
    "ae_page": "Wikipedia:Arbitration/Requests/Enforcement",
    "arc_page": "Wikipedia:Arbitration/Requests/Case",
    "target_page": "User:WordcountClerkBot/word counts",
    "default_limit": 500,
    "over_factor": 1.10,
    "run_interval": 600,
    "red_hex": "#ffcccc",
    "amber_hex": "#ffffcc",
    "placeholder_heading": "statement by {other-editor}",
}

CFG = DEFAULT_CFG.copy()

###############################################################################
# Logging                                                                     #
###############################################################################

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

###############################################################################
# Regex helpers & constants                                                   #
###############################################################################

WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
AWL_RE = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*?\bwords\s*=\s*(\d+)", re.I | re.S)
TEMPLATE_RE = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*}}", re.I | re.S)
PAREN_RE = re.compile(r"^(.*?)(\s*\([^()]+\)\s*)$")
HAT_OPEN_RE = re.compile(r"\{\{\s*hat", re.I)
HAT_CLOSE_RE = re.compile(r"\{\{\s*hab\s*}}", re.I)

slugify = lambda s: re.sub(r"\s+", "_", mwpfh.parse(s).strip_code()).strip("_")

###############################################################################
# Dataclasses                                                                 #
###############################################################################

@dataclass
class Statement:
    user: str
    anchor: str
    words: int
    limit: int

    @property
    def status(self) -> str:  # ok | within | over
        if self.words <= self.limit:
            return "ok"
        elif self.words <= self.limit * CFG["over_factor"]:
            return "within 10%"
        return "over"

    @property
    def colour(self) -> str | None:
        return {
            "within 10%": CFG["amber_hex"],
            "over": CFG["red_hex"],
        }.get(self.status)


@dataclass
class RequestTable:
    title: str
    anchor: str
    statements: List[Statement]
    closed: bool = False

    HEADER: ClassVar[str] = (
        "{| class=\"wikitable sortable\"\n! User !! Section !! Words !! Limit !! Status\n"
    )

    def to_wikitext(self, board_page: str) -> str:
        heading_link = f"[[{board_page}#{self.anchor}|{self.title}]]"
        heading_line = f"=== {heading_link} ==="
        if self.closed:
            heading_line += "\n''(closed)''"
        if not self.statements:
            return heading_line + "\n''No word‑limited statements.''"
        rows = []
        for s in self.statements:
            if not s.user:
                continue
            style = f" style=\"background:{s.colour}\"" if s.colour else ""
            rows.append(
                f"|-{style}\n| [[User:{s.user}|{s.user}]] || [[{board_page}#{s.anchor}|link]] "
                f"|| {s.words} || {s.limit} || {s.status}"
            )
        return heading_line + "\n" + self.HEADER + "\n".join(rows) + "\n|}"

###############################################################################
# Wiki utilities                                                              #
###############################################################################

_RENDER_CACHE: dict[str, int] = {}

def rendered_word_count(site: mwclient.Site, wikitext: str) -> int:
    if wikitext in _RENDER_CACHE:
        return _RENDER_CACHE[wikitext]
    try:
        html = site.api("parse", text=wikitext, prop="text", contentmodel="wikitext")["parse"]["text"]["*"]
        words = len(WORD_RE.findall(re.sub(r"<[^>]+>", "", html)))
    except Exception as exc:
        LOGGER.warning("parse API failed: %s", exc)
        words = 0
    _RENDER_CACHE[wikitext] = words
    return words

###############################################################################
# Username heuristics                                                         #
###############################################################################

def strip_parenthetical(username: str, body: str) -> str:
    m = PAREN_RE.match(username)
    if not m:
        return username
    base = m.group(1).rstrip()
    if re.search(rf"\[\[\s*(?:User(?: talk)?)\s*:\s*{re.escape(username)}\b", body, flags=re.I):
        return username
    return base


def user_links(body: str) -> List[str]:
    links: List[str] = []
    for wl in mwpfh.parse(body).filter_wikilinks():
        title = str(wl.title)
        ns, _, rest = title.partition(":")
        if ns.lower() in {"user", "user talk"} and rest:
            links.append(rest.split("/")[0].strip())
    return links


def fuzzy_username(header: str, body: str) -> str:
    simple = strip_parenthetical(header, body)
    cands = user_links(body)
    if not cands:
        return simple
    for c in cands:
        if c.lower() == simple.lower():
            return c
    best, score = simple, 0.0
    for c in cands:
        r = SequenceMatcher(None, simple.lower(), c.lower()).ratio()
        if r > score:
            best, score = c, r
    return best if score >= 0.6 else simple

###############################################################################
# Section scanner                                                             #
###############################################################################

@dataclass
class RawSection:
    level: int
    title: str
    start: int
    end: int

    def body(self, text: str) -> str:
        return text[self.start : self.end]


def scan_sections(text: str) -> List[RawSection]:
    heads = list(HEADING_RE.finditer(text))
    return [
        RawSection(len(h.group(1)), h.group(2).strip(), h.end(), heads[i + 1].start() if i + 1 < len(heads) else len(text))
        for i, h in enumerate(heads)
    ]

###############################################################################
# Parsers                                                                     #
###############################################################################

class BaseParser:
    board_page: str

    def __init__(self, site: mwclient.Site):
        self.site = site

    def _make_statement(self, raw_user: str, body: str, anchor: str, limit: int | None = None) -> Statement:
        return Statement(
            fuzzy_username(raw_user, body),
            anchor,
            rendered_word_count(self.site, TEMPLATE_RE.sub("", body)),
            limit or CFG["default_limit"],
        )

    def parse(self, text: str) -> List[RequestTable]:
        raise NotImplementedError


class SimpleBoardParser(BaseParser):
    """ARCA/ARC – mwparserfromhell, straightforward."""

    def parse(self, text: str) -> List[RequestTable]:
        code = mwpfh.parse(text)
        tables: List[RequestTable] = []
        for lvl2 in code.get_sections(levels=[2]):
            sec_title = mwpfh.parse(lvl2.filter_headings()[0].title).strip_code().strip()
            anchor = slugify(sec_title)
            sec_wikitext = str(lvl2)
            closed = HAT_OPEN_RE.match(sec_wikitext) and HAT_CLOSE_RE.search(sec_wikitext)
            statements: List[Statement] = []
            for st in lvl2.get_sections(levels=[3]):
                heading = mwpfh.parse(st.filter_headings()[0].title).strip_code().strip()
                if heading.lower() == CFG["placeholder_heading"]:
                    continue
                if not heading.lower().startswith("statement by"):
                    continue
                raw_user = re.sub(r"^Statement by\s+", "", heading, flags=re.I)
                body = str(st).split("\n", 1)[1]
                statements.append(self._make_statement(raw_user, body, slugify(heading)))
            tables.append(RequestTable(sec_title, anchor, statements, closed=bool(closed)))
        return tables


class AEParser(BaseParser):
    _STMT = re.compile(r"^Statement by\s+", re.I)
    _REQ_USER = re.compile(r";\s*User who is submitting.*?\{\{\s*userlinks\|(.*?)}}", re.I | re.S)
    _PSEUDO = re.compile(r";\s*(\[\[WP:DIFF\|Diffs\]\] of edits[^:]*|Diffs of previous[^:]*|Additional comments[^:]*):", re.I)

    def parse(self, text: str) -> List[RequestTable]:
        sections = scan_sections(text)
        tables: List[RequestTable] = []
        current: RequestTable | None = None
        for sec in sections:
            if sec.level == 2:
                closed = HAT_OPEN_RE.match(sec.body(text)) and HAT_CLOSE_RE.search(sec.body(text))
                current = RequestTable(sec.title, slugify(sec.title), [], bool(closed))
                tables.append(current)
                continue
            if current is None:
                continue
            if sec.level in {3, 4} and self._STMT.match(sec.title):
                body = sec.body(text)
                raw_user = self._STMT.sub("", sec.title)
                current.statements.append(self._make_statement(raw_user, body, slugify(sec.title)))
            elif sec.level == 3 and sec.title.lower().startswith("request concerning"):
                body = self._collect_request_body(sec, text)
                if body:
                    user = self._requesting_user(sec, text)
                    current.statements.append(self._make_statement(user, body, slugify(f"{sec.title}-{user}")))
        return tables

    def _requesting_user(self, sec: RawSection, full: str) -> str:
        m = self._REQ_USER.search(sec.body(full))
        return m.group(1).strip() if m else ""

    def _collect_request_body(self, sec: RawSection, full: str) -> str:
        seg = sec.body(full)
        parts = []
        for m in self._PSEUDO.finditer(seg):
            start = m.end()
            nxt = re.search(r"^(?:;|=){1,6}", seg[start:], flags=re.M)
            end = nxt.start() + start if nxt else len(seg)
            parts.append(seg[start:end])
        return "\n".join(parts)

###############################################################################
# Registry & report builder                                                   #
###############################################################################

BOARD_PARSERS = {
    "ARCA": (CFG["arca_page"], SimpleBoardParser),
    "AE": (CFG["ae_page"], AEParser),
    "ARC": (CFG["arc_page"], SimpleBoardParser),
}


def build_report(site: mwclient.Site) -> str:
    blocks = []
    for label, (page, ParserCls) in BOARD_PARSERS.items():
        raw = site.pages[page].text()
        parser = ParserCls(site)
        parser.board_page = page  # type: ignore
        tables = parser.parse(raw)
        if not tables:
            # Always output ARC section even if empty
            if label == "ARC":
                blocks.append("== ARC ==\n''No open requests.''")
            continue
        body = "\n\n".join(t.to_wikitext(page) for t in tables)
        blocks.append(f"== {label} ==\n{body}")
    return "\n\n".join(blocks)

###############################################################################
# Main orchestration                                                          #
###############################################################################

def load_settings(path: str = SETTINGS_PATH) -> None:
    if os.path.exists(path):
        with open(path) as f:
            CFG.update(json.load(f))


def connect() -> mwclient.Site:
    sess = requests.Session()
    site = mwclient.Site(CFG["site"], path=CFG["path"], clients_useragent=CFG["ua"], pool=sess)
    if not site.logged_in:
        site.login(CFG["user"], CFG["bot_password"])
    return site


def run_once(site: mwclient.Site) -> None:
    target = site.pages[CFG["target_page"]]
    new_text = build_report(site)
    if new_text != target.text():
        target.save(new_text, summary="Wordcount update – bot", minor=False, bot=False)
        LOGGER.info("Updated target page.")
    else:
        LOGGER.info("No changes detected.")


def main(loop: bool, debug: bool) -> None:
    if debug:
        LOGGER.setLevel(logging.DEBUG)
    load_settings()
    site = connect()
    if not loop:
        run_once(site)
        return
    interval = CFG["run_interval"]
    while True:
        try:
            run_once(site)
        except Exception:
            LOGGER.exception("Error; sleeping before retry")
        time.sleep(interval)

###############################################################################
# Entry‑point                                                                 #
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WordcountClerkBot 2.1")
    ap.add_argument("-1", "--once", action="store_true", help="run once and exit")
    ap.add_argument("--debug", action="store_true", help="verbose debug logging")
    args = ap.parse_args()
    main(loop=not args.once, debug=args.debug)
