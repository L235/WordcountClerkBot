#!/usr/bin/env python3
"""
WordcountClerkBot
========================================

**Version 2.4 – accurate "visible" counts**

The bot now mirrors the front‑end word‑counter even more closely:

* **Expanded words** – unchanged.
* **Visible words** – now computed with real HTML parsing (BeautifulSoup) so
  that content hidden by `mw-collapsible mw-collapsed`, `<span
  style="display:none">`, struck‑through text, etc., is excluded exactly as the
  on‑wiki JavaScript does.

Only the *visible* figure is compared with the word‑limit policy.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import ClassVar, List, Dict, Tuple, Type

# ---------------------------------------------------------------------------
# Third‑party dependencies
# ---------------------------------------------------------------------------
import mwclient
import mwparserfromhell as mwpfh
import requests
from bs4 import BeautifulSoup  # NEW – precise HTML filtering

###############################################################################
# Configuration                                                               #
###############################################################################

SETTINGS_PATH = "settings.json"

DEFAULT_CFG = {
    "site": "en.wikipedia.org",
    "path": "/w/",
    "user": "BotUser@PasswordName",
    "bot_password": "",
    "ua": "WordcountClerkBot/2.4 (https://github.com/L235/WordcountClerkBot)",
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
    "header_text": "",
    "placeholder_heading": "statement by {other-editor}",
}

CFG: Dict[str, object] = DEFAULT_CFG.copy()

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
_TS_RE = re.compile(r"\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")

slugify = lambda s: re.sub(r"\s+", "_", mwpfh.parse(s).strip_code()).strip("_")

###############################################################################
# Dataclasses                                                                 #
###############################################################################

@dataclass
class Statement:
    user: str
    anchor: str
    visible: int
    expanded: int
    limit: int

    @property
    def status(self) -> str:
        if self.visible <= self.limit:
            return "ok"
        elif self.visible <= self.limit * float(CFG["over_factor"]):
            return "within 10%"
        return "over"

    @property
    def colour(self) -> str | None:
        return {
            "within 10%": str(CFG["amber_hex"]),
            "over": str(CFG["red_hex"]),
        }.get(self.status)


@dataclass
class RequestTable:
    title: str
    anchor: str
    statements: List[Statement]
    closed: bool = False

    HEADER: ClassVar[str] = (
        "{| class=\"wikitable sortable\"\n"
        "! User !! Section !! Words !! Uncollapsed&nbsp;words !! Limit !! Status\n"
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
                f"|-{style}\n"
                f"| [[User:{s.user}|{s.user}]] "
                f"|| [[{board_page}#{s.anchor}|link]] "
                f"|| {s.visible} || {s.expanded} || {s.limit} || {s.status}"
            )
        return heading_line + "\n" + self.HEADER + "\n".join(rows) + "\n|}"

###############################################################################
# Word‑count helpers                                                          #
###############################################################################

_RENDER_CACHE: Dict[str, int] = {}
_VISIBLE_CACHE: Dict[str, int] = {}

def _api_render(site: mwclient.Site, wikitext: str) -> str:
    return site.api(
        "parse",
        text=wikitext,
        prop="text",
        contentmodel="wikitext",
    )["parse"]["text"]["*"]


def rendered_word_count(site: mwclient.Site, wikitext: str) -> int:
    if wikitext in _RENDER_CACHE:
        return _RENDER_CACHE[wikitext]
    try:
        html = _api_render(site, wikitext)
        words = len(WORD_RE.findall(re.sub(r"<[^>]+>", "", html)))
    except Exception as exc:
        LOGGER.warning("parse API failed: %s", exc)
        words = 0
    _RENDER_CACHE[wikitext] = words
    return words


def visible_word_count(site: mwclient.Site, wikitext: str) -> int:
    """
    Replicates front‑end wordcount.js:

    • remove display:none elements  
    • drop collapsible‑but‑collapsed (`mw-collapsed`, `mw-collapsible-content`)  
    • remove struck‑through text  
    • strip page furniture  
    • delete UTC timestamps  
    • count tokens with at least one alphanumeric
    """
    if wikitext in _VISIBLE_CACHE:
        return _VISIBLE_CACHE[wikitext]

    try:
        html = _api_render(site, wikitext)
        soup = BeautifulSoup(html, "html.parser")

        # 1 – elements hidden via inline style
        for tag in soup.select('[style*="display:none" i]'):
            tag.decompose()

        # 2 – collapsed content
        for tag in soup.select('.mw-collapsed, .mw-collapsible-content'):
            tag.decompose()

        # 3 – struck‑through content
        for tag in soup.select('[style*="text-decoration:line-through" i], s, strike, del'):
            tag.decompose()

        # 4 – page furniture
        for tag in soup.select('div#siteSub, div#contentSub, div#jump-to-nav'):
            tag.decompose()

        # 5 – plain text & cleanup
        text = _TS_RE.sub("", soup.get_text())
        tokens = [
            t for t in re.split(r"\s+", text)
            if t and re.search(r"[A-Za-z0-9]", t)
        ]
        words = len(tokens)
    except Exception as exc:
        LOGGER.warning("visible parse API failed: %s", exc)
        words = 0

    _VISIBLE_CACHE[wikitext] = words
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
# Section scanner (for AE)                                                    #
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
        RawSection(
            len(h.group(1)),
            h.group(2).strip(),
            h.end(),
            heads[i + 1].start() if i + 1 < len(heads) else len(text),
        )
        for i, h in enumerate(heads)
    ]

###############################################################################
# Parsers                                                                     #
###############################################################################

class BaseParser:
    board_page: str

    def __init__(self, site: mwclient.Site):
        self.site = site

    def _extract_limit(self, body: str, default: int | None = None) -> int:
        m = AWL_RE.search(body)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                LOGGER.debug("Invalid ApprovedWordLimit value: %s", m.group(1))
        return default or int(CFG["default_limit"])

    def _make_statement(
        self, raw_user: str, body: str, anchor: str, limit: int | None = None
    ) -> Statement:
        limit_val = self._extract_limit(body, limit)
        body_no_awl = TEMPLATE_RE.sub("", body)
        visible = visible_word_count(self.site, body_no_awl)
        expanded = rendered_word_count(self.site, body_no_awl)
        return Statement(
            fuzzy_username(raw_user, body),
            anchor,
            visible,
            expanded,
            limit_val,
        )

    def parse(self, text: str) -> List[RequestTable]:
        raise NotImplementedError


class SimpleBoardParser(BaseParser):
    def parse(self, text: str) -> List[RequestTable]:
        code = mwpfh.parse(text)
        tables: List[RequestTable] = []
        for lvl2 in code.get_sections(levels=[2]):
            sec_title = mwpfh.parse(lvl2.filter_headings()[0].title).strip_code().strip()
            anchor = slugify(sec_title)
            sec_wikitext = str(lvl2)
            closed = bool(
                HAT_OPEN_RE.match(sec_wikitext.lstrip()) and HAT_CLOSE_RE.search(sec_wikitext)
            )

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

            tables.append(RequestTable(sec_title, anchor, statements, closed))
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
                closed = bool(
                    HAT_OPEN_RE.match(sec.body(text).lstrip()) and HAT_CLOSE_RE.search(sec.body(text))
                )
                current = RequestTable(sec.title, slugify(sec.title), [], closed)
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
                    current.statements.append(
                        self._make_statement(user, body, slugify(f"{sec.title}"))
                    )
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
# Dynamic board registry                                                      #
###############################################################################

def get_board_parsers() -> Dict[str, Tuple[str, Type[BaseParser]]]:
    return {
        "ARCA": (str(CFG["arca_page"]), SimpleBoardParser),
        "AE": (str(CFG["ae_page"]), AEParser),
        "ARC": (str(CFG["arc_page"]), SimpleBoardParser),
    }

###############################################################################
# Report builder                                                             #
###############################################################################

def build_report(site: mwclient.Site) -> str:
    blocks: List[str] = []
    blocks.append(CFG["header_text"])
    for label, (page, ParserCls) in get_board_parsers().items():
        raw = site.pages[page].text()
        parser = ParserCls(site)
        parser.board_page = page  # type: ignore[attr-defined]
        tables = parser.parse(raw)

        if not tables:
            # if this board has no open requests, show a placeholder for every label
            blocks.append(f"== {label} ==\n''No open requests.''")
            continue

        body = "\n\n".join(t.to_wikitext(page) for t in tables)
        blocks.append(f"== {label} ==\n{body}")
    return "\n\n".join(blocks)

###############################################################################
# Runtime helpers                                                             #
###############################################################################

def load_settings(path: str = SETTINGS_PATH) -> None:
    if os.path.exists(path):
        with open(path) as f:
            CFG.update(json.load(f))


def connect() -> mwclient.Site:
    sess = requests.Session()
    site = mwclient.Site(
        str(CFG["site"]),
        path=str(CFG["path"]),
        clients_useragent=str(CFG["ua"]),
        pool=sess,
    )
    if not site.logged_in:
        site.login(str(CFG["user"]), str(CFG["bot_password"]))
    return site


def run_once(site: mwclient.Site) -> None:
    target = site.pages[str(CFG["target_page"])]
    new_text = build_report(site)
    if new_text != target.text():
        # recompute stats for the edit summary
        all_tables: List[RequestTable] = []
        for label, (page, Pcls) in get_board_parsers().items():
            raw = site.pages[page].text()
            parser = Pcls(site)
            parser.board_page = page  # type: ignore[attr-defined]
            all_tables.extend(parser.parse(raw))

        all_statements = [s for tbl in all_tables for s in tbl.statements]
        x = sum(1 for s in all_statements if s.status == "over")
        y = sum(1 for s in all_statements if s.status == "within 10%")
        z = len(all_statements)
        a = len(all_tables)

        summary = (
            f"updating table ({x} statements over, {y} statements within 10%, "
            f"{z} total statements, {a} total pending requests) "
            f"([[User:KevinClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"
        )
        target.save(new_text, summary=summary, minor=False, bot=False)
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
    interval = int(CFG["run_interval"])
    while True:
        try:
            run_once(site)
        except Exception:
            LOGGER.exception("Error; sleeping before retry")
        time.sleep(interval)

###############################################################################
# CLI entry‑point                                                             #
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WordcountClerkBot 2.4")
    ap.add_argument("-1", "--once", action="store_true", help="run once and exit")
    ap.add_argument("--debug", action="store_true", help="verbose debug logging")
    args = ap.parse_args()
    main(loop=not args.once, debug=args.debug)
