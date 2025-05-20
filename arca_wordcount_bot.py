#!/usr/bin/env python3
"""
WordcountClerkBot
========================================

A bot that monitors word counts in Wikipedia arbitration requests. It:

* Scans ARCA, AE, and ARC pages for word-limited statements
* Counts both visible and expanded word counts for each statement
* Compares counts against approved word limits
* Generates a report page with color-coded status indicators
* Updates automatically on a configurable interval

The bot uses HTML parsing to match the front-end word counter exactly,
excluding hidden content, collapsed sections, and struck-through text.
Only the visible word count is compared against the word limit policy.
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
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from functools import lru_cache
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
    "data_page": "User:WordcountClerkBot/word counts/data",
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

# ─── 1) CUSTOM LOGGING SETUP (INFO→stdout, WARNING+→stderr) ───────────────────
class MaxLevelFilter(logging.Filter):
    """Allow through only records <= a given level."""
    def __init__(self, level: int):
        super().__init__()
        self.max_level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

LOG = logging.getLogger(__name__)
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Handler for INFO and DEBUG → stdout
h_info = logging.StreamHandler(sys.stdout)
h_info.setLevel(logging.DEBUG)
h_info.addFilter(MaxLevelFilter(logging.INFO))

# Handler for WARNING and above → stderr
h_err = logging.StreamHandler(sys.stderr)
h_err.setLevel(logging.WARNING)

fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
h_info.setFormatter(fmt)
h_err.setFormatter(fmt)

LOG.addHandler(h_info)
LOG.addHandler(h_err)

###############################################################################
# Regex helpers & constants                                                   #
###############################################################################

WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
AWL_RE = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*?\bwords\s*=\s*(\d+)", re.I | re.S)
TEMPLATE_RE = re.compile(r"\{\{\s*(?:ApprovedWordLimit|ACWordStatus)[^}]*}}", re.I | re.S)
PAREN_RE = re.compile(r"^(.*?)(\s*\([^()]+\)\s*)$")
HAT_OPEN_RE = re.compile(r"\{\{\s*hat", re.I)
HAT_CLOSE_RE = re.compile(r"\{\{\s*hab\s*}}", re.I)
_TS_RE = re.compile(r"\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")

def slugify(s: str) -> str:
    """Convert a heading into a MediaWiki anchor (no spaces, punctuation stripped)."""
    text = mwpfh.parse(s).strip_code()
    return re.sub(r"\s+", "_", text).strip("_")

def strip_parenthetical(username: str, body: str) -> str:
    """Remove parenthetical content from username if not referenced in body."""
    m = PAREN_RE.match(username)
    if not m:
        return username
    base = m.group(1).rstrip()
    if re.search(rf"\[\[\s*(?:User(?: talk)?)\s*:\s*{re.escape(username)}\b", body, flags=re.I):
        return username
    return base

def user_links(body: str) -> List[str]:
    """Extract usernames from User: and User talk: links in wikitext."""
    links: List[str] = []
    for wl in mwpfh.parse(body).filter_wikilinks():
        title = str(wl.title)
        ns, _, rest = title.partition(":")
        if ns.lower() in {"user", "user talk"} and rest:
            links.append(rest.split("/")[0].strip())
    return links

def fuzzy_username(header: str, body: str) -> str:
    """
    Find best matching username from header and body links.
    
    Args:
        header: Raw username from section header
        body: Full wikitext of the section
        
    Returns:
        Best matching username, preferring exact matches from links
    """
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
    return best if score >= 0.3 else simple

###############################################################################
# Dataclasses                                                                 #
###############################################################################

class Status(Enum):
    """Possible states for a statement's word count."""
    OK = "ok"
    WITHIN = "within 10%"
    OVER = "over"

@dataclass
class Statement:
    user: str
    anchor: str
    visible: int
    expanded: int
    limit: int

    @property
    def status(self) -> Status:
        if self.visible <= self.limit:
            return Status.OK
        if self.visible <= self.limit * float(CFG["over_factor"]):
            return Status.WITHIN
        return Status.OVER

    @property
    def colour(self) -> str | None:
        mapping = {
            Status.WITHIN: str(CFG["amber_hex"]),
            Status.OVER: str(CFG["red_hex"]),
        }
        return mapping.get(self.status)


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
                f"|| {s.visible} || {s.expanded} || {s.limit} || {s.status.value}"
            )
        return heading_line + "\n" + self.HEADER + "\n".join(rows) + "\n|}"

###############################################################################
# Word‑count helpers                                                          #
###############################################################################

@lru_cache(maxsize=1024)
def _api_render(site_url: str, api_path: str, wikitext: str) -> str:
    """Call parse API once per unique wikitext."""
    site = connect()  # we could optimize by caching site too
    return site.api(
        "parse", text=wikitext, prop="text", contentmodel="wikitext"
    )["parse"]["text"]["*"]

def rendered_word_count(site: mwclient.Site, wikitext: str) -> int:
    """Count words in rendered HTML, including hidden content."""
    html = _api_render(str(CFG["site"]), str(CFG["path"]), wikitext)
    return len(WORD_RE.findall(re.sub(r"<[^>]+>", "", html)))

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
    html = _api_render(str(CFG["site"]), str(CFG["path"]), wikitext)
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
    return len(tokens)

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
                LOG.debug("Invalid ApprovedWordLimit value: %s", m.group(1))
        return default or int(CFG["default_limit"])

    def _make_statement(
        self, raw_user: str, body: str, anchor: str, limit: int | None = None
    ) -> Statement:
        limit_val = self._extract_limit(body, limit)
        body_no_templates = TEMPLATE_RE.sub("", body)
        visible = visible_word_count(self.site, body_no_templates)
        expanded = rendered_word_count(self.site, body_no_templates)
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
# Runtime helpers                                                             #
###############################################################################

def load_settings(path: str = SETTINGS_PATH) -> None:
    """Load JSON settings into CFG (overrides defaults)."""
    defaults = DEFAULT_CFG.copy()
    CFG.update(defaults)
    if os.path.exists(path):
        with open(path) as f:
            CFG.update(json.load(f))

def connect() -> mwclient.Site:
    """Login to MediaWiki and return a Site object."""
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

def fetch_page(site: mwclient.Site, title: str) -> str:
    """Fetch wikitext body of `title` from the wiki."""
    return site.pages[title].text()

def assemble_report(site: mwclient.Site) -> str:
    """Build the entire report wikitext by fetching each board and parsing it."""
    blocks: List[str] = [CFG["header_text"]]
    for label, (page, ParserCls) in get_board_parsers().items():
        raw = fetch_page(site, page)
        parser = ParserCls(site)
        parser.board_page = page  # type: ignore
        tables = parser.parse(raw)

        if not tables:
            blocks.append(f"== {label} ==\n''No open requests.''")
        else:
            body = "\n\n".join(t.to_wikitext(page) for t in tables)
            blocks.append(f"== {label} ==\n{body}")
    return "\n\n".join(blocks)

def assemble_data_template(site: mwclient.Site) -> str:
    """
    Build a nested #switch template containing full data for each statement.
    """
    # First, gather everything into a nested dict:
    data: dict[str, dict[str, dict[str, Statement]]] = {}
    for label, (page, ParserCls) in get_board_parsers().items():
        parser = ParserCls(site)
        parser.board_page = page  # type: ignore
        raw = fetch_page(site, page)
        for tbl in parser.parse(raw):
            # Use the title instead of anchor for the key to preserve spaces
            data.setdefault(label, {}).setdefault(tbl.title, {})
            for stmt in tbl.statements:
                data[label][tbl.title][stmt.user] = stmt

    # Now build the wikitext:
    parts: list[str] = []
    parts.append('{{#switch: {{{page}}}')  # outer switch on page
    for label, requests in data.items():
        parts.append(' | ' + label + ' = {{#switch: {{{section}}}')  # inner switch on section
        for req, users in requests.items():
            parts.append('     | ' + req + ' = {{#switch: {{{user}}}')  # inner switch on user
            for user, stmt in users.items():
                parts.append('         | ' + user + ' = {{#switch: {{{type}}}')  # inner switch on type
                parts.append(f'             | words       = {stmt.visible}')
                parts.append(f'             | uncollapsed = {stmt.expanded}')
                parts.append(f'             | limit       = {stmt.limit}')
                parts.append(f'             | status      = {stmt.status.value}')
                parts.append('           }}')  # close type switch
            parts.append('       }}')  # close user switch
        parts.append('   }}')  # close section switch
    parts.append('}}')  # close page switch

    return "\n".join(parts)

def run_once(site: mwclient.Site) -> None:
    """
    Build report, compare to target page, and save if changed.
    """
    target = site.pages[str(CFG["target_page"])]
    new_text = assemble_report(site)
    if new_text != target.text():
        # recompute stats for the edit summary
        all_tables: List[RequestTable] = []
        for label, (page, Pcls) in get_board_parsers().items():
            raw = fetch_page(site, page)
            parser = Pcls(site)
            parser.board_page = page  # type: ignore
            all_tables.extend(parser.parse(raw))

        all_statements = [s for tbl in all_tables for s in tbl.statements]
        x = sum(1 for s in all_statements if s.status == Status.OVER)
        y = sum(1 for s in all_statements if s.status == Status.WITHIN)
        z = len(all_statements)
        a = len(all_tables)

        summary = (
            f"updating table ({x} statements over, {y} statements within 10%, "
            f"{z} total statements, {a} total pending requests) "
            f"([[User:KevinClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"
        )
        target.save(new_text, summary=summary, minor=False, bot=False)
        LOG.info("Updated target page.")

        # now update the data‐template page
        data_title = str(CFG["data_page"])
        data_page = site.pages[data_title]
        new_data = assemble_data_template(site)
        if new_data != data_page.text():
            data_page.save(new_data,
                           summary="Updating data template",
                           minor=True,
                           bot=False)
            LOG.info("Updated data template page.")
    else:
        LOG.info("No changes detected.")

def main(loop: bool, debug: bool) -> None:
    """
    Entry point: load settings, connect, and either run once or loop.
    """
    if debug:
        LOG.setLevel(logging.DEBUG)
    load_settings()
    site = connect()
    if not loop:
        run_once(site)
    else:
        interval = int(CFG["run_interval"])
        while True:
            try:
                run_once(site)
            except Exception:
                LOG.exception("Error; sleeping before retry")
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
