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
from datetime import datetime, timezone
import pickle
import http.cookiejar

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

# Default configuration values
DEFAULT_CFG = {
    "SITE": "en.wikipedia.org",
    "API_PATH": "/w/",
    "BOT_USER": "BotUser@PasswordName",
    "BOT_PASSWORD": "",
    "USER_AGENT": "WordcountClerkBot/2.4 (https://github.com/L235/WordcountClerkBot)",
    "ARCA_PAGE": "Wikipedia:Arbitration/Requests/Clarification and Amendment",
    "AE_PAGE": "Wikipedia:Arbitration/Requests/Enforcement",
    "ARC_PAGE": "Wikipedia:Arbitration/Requests/Case",
    "OPEN_CASES_PAGE": "Template:ArbComOpenTasks/Cases",
    "TARGET_PAGE": "User:WordcountClerkBot/word counts",
    "DATA_PAGE": "User:WordcountClerkBot/word counts/data",
    "EXTENDED_PAGE": "User:WordcountClerkBot/word counts/extended",
    "DEFAULT_LIMIT": 500,
    "EVIDENCE_LIMIT_NAMED": 1000,
    "EVIDENCE_LIMIT_OTHER": 500,
    "OVER_FACTOR": 1.10,
    "RUN_INTERVAL": 600,
    "RED_HEX": "#ff000040",
    "AMBER_HEX": "#ffff0040",
    "HEADER_TEXT": "",
    "PLACEHOLDER_HEADING": "statement by {other-editor}",
    "SESSION_FILE": "~/wordcountclerkbot/cookies/cookies.txt",
}

# Global configuration dictionary
CFG: Dict[str, object] = {}

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

WORD_RE        = re.compile(r"\b\w+(?:['-]\w+)*\b")
HEADING_RE     = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)
AWL_RE         = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*?\bwords\s*=\s*(\d+)", re.I | re.S)
TEMPLATE_RE    = re.compile(r"\{\{\s*(?:ApprovedWordLimit|ACWordStatus)[^}]*}}", re.I | re.S)
ACWORDSTATUS_RE = re.compile(r"\{\{\s*ACWordStatus\b", re.I)
PAREN_RE = re.compile(r"^(.*?)(\s*\([^()]+\)\s*)$")
HAT_OPEN_RE = re.compile(r"\{\{\s*hat", re.I)
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
    def capitalize_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s
    
    simple = capitalize_first(strip_parenthetical(header, body))
    cands = [capitalize_first(c) for c in user_links(body)]
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
    has_status_template: bool = False

    @property
    def status(self) -> Status:
        if self.visible <= self.limit:
            return Status.OK
        if self.visible <= self.limit * float(CFG["OVER_FACTOR"]):
            return Status.WITHIN
        return Status.OVER

    @property
    def colour(self) -> str | None:
        mapping = {
            Status.WITHIN: str(CFG["AMBER_HEX"]),
            Status.OVER: str(CFG["RED_HEX"]),
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

    EXTENDED_HEADER: ClassVar[str] = (
        "{| class=\"wikitable sortable\"\n"
        "! User !! Section !! Words !! Uncollapsed&nbsp;words !! Limit !! Status !! Template\n"
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

    def to_extended_wikitext(self, board_page: str, label: str) -> str:
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
            template = f"{{{{ACWordStatus|page={label}|section={self.title}|user={s.user}}}}}"

            # yellow‑highlight the cell only when the template is **missing**
            if s.has_status_template:
                template_cell = f"|| <nowiki>{template}</nowiki>"
            else:
                template_cell = (
                    f'|| style="background:{CFG["AMBER_HEX"]}" | '
                    f'<nowiki>{template}</nowiki>'
                )

            rows.append(
                f"|-{style}\n"
                f"| [[User:{s.user}|{s.user}]] "
                f"|| [[{board_page}#{s.anchor}|link]] "
                f"|| {s.visible} || {s.expanded} || {s.limit} || {s.status.value} "
                f"{template_cell}"
            )
        return heading_line + "\n" + self.EXTENDED_HEADER + "\n".join(rows) + "\n|}"

###############################################################################
# Word‑count helpers                                                          #
###############################################################################

@lru_cache(maxsize=1024)
def _api_render_cached(site_url: str, api_path: str, wikitext: str) -> str:
    """Cache the raw API response - site must be passed separately."""
    # This function now only caches the API call, doesn't handle the site connection
    pass  # Implementation moved to _api_render

def _api_render(site: mwclient.Site, wikitext: str) -> str:
    """Call parse API once per unique wikitext using provided site."""
    # Create a cache key that doesn't include the site object
    cache_key = (str(site.host), str(site.path), wikitext)
    
    # Try to get from a simple in-memory cache first
    if not hasattr(_api_render, '_cache'):
        _api_render._cache = {}
    
    if cache_key in _api_render._cache:
        return _api_render._cache[cache_key]
    
    # Make the API call with the provided site
    result = site.api(
        "parse", text=wikitext, prop="text", contentmodel="wikitext"
    )["parse"]["text"]["*"]
    
    # Cache the result
    _api_render._cache[cache_key] = result
    
    # Limit cache size to prevent memory issues
    if len(_api_render._cache) > 1000:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_api_render._cache.keys())[:100]
        for key in keys_to_remove:
            del _api_render._cache[key]
    
    return result

def rendered_word_count(site: mwclient.Site, wikitext: str) -> int:
    """Count words in rendered HTML, including hidden content."""
    html = _api_render(site, wikitext)  # Pass site instead of reconnecting
    return len(WORD_RE.findall(re.sub(r"<[^>]+>", "", html)))

def visible_word_count(site: mwclient.Site, wikitext: str) -> int:
    """
    Approximate the front-end wordcount.js logic for "visible words".
    * Render the snippet through the MediaWiki API.
    * Remove anything that the canonical script ignores:
        – hidden / collapsed / struck-through elements  
        – page-furniture boxes  
        – reference-list content (text produced from <ref> tags)  
        – localcomments spans  
        – timestamps (hh:mm, d Month yyyy UTC)
    """
    html = _api_render(site, wikitext)  # Pass site instead of reconnecting
    soup = BeautifulSoup(html, "html.parser")

    # 0 – drop reference lists generated from <ref>…</ref>
    #     (ol.references / .reflist blocks are *not* part of the prose limit)
    for tag in soup.select("ol.references, div.references, div.reflist"):
        tag.decompose()

    # 1 – elements hidden outright
    for tag in soup.select('[style*="display:none" i]'):
        tag.decompose()

    # 1a – HTML5/ARIA hidden helpers (also matched by jQuery :hidden)
    for tag in soup.select("[hidden], [aria-hidden='true']"):
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

    # 4a – inline talk-page comments (ignored by frontend via .ignore())
    for tag in soup.select("span.localcomments"):
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
# Open case helpers                                                           #
###############################################################################

def extract_open_cases(site: mwclient.Site) -> List[str]:
    """Extract case names from Template:ArbComOpenTasks/Cases."""
    page_text = fetch_page(site, str(CFG["OPEN_CASES_PAGE"]))
    case_names = []
    
    # Parse the wikitext to find ArbComOpenTasks/line templates
    code = mwpfh.parse(page_text)
    for template in code.filter_templates():
        template_name = str(template.name).strip()
        if template_name == "ArbComOpenTasks/line":
            # Check if mode=case and extract name
            mode_param = template.get("mode").value
            name_param = template.get("name").value
            if mode_param and str(mode_param).strip() == "case" and name_param:
                case_names.append(str(name_param).strip())
    return case_names

def extract_involved_parties(site: mwclient.Site, case_name: str) -> List[str]:
    """Extract involved parties from a case page."""
    case_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}"
    try:
        page_text = fetch_page(site, case_page)
    except Exception:
        LOG.warning("Could not fetch case page: %s", case_page)
        return []
    
    parties = []
    sections = scan_sections(page_text)
    
    # Find "Involved parties" section (level 3)
    for sec in sections:
        if sec.level == 3 and sec.title.lower() == "involved parties":
            body = sec.body(page_text)
            # Look for {{admin|...}} and {{userlinks|...}} templates
            for match in re.finditer(r'\{\{\s*(?:admin|userlinks)\s*\|\s*(?:1\s*=\s*)?([^|}]+)', body, re.I):
                username = match.group(1).strip()
                if username:
                    parties.append(username)
            break
    
    return parties

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
        return default or int(CFG["DEFAULT_LIMIT"])

    def _make_statement(
        self, raw_user: str, body: str, anchor: str, limit: int | None = None
    ) -> Statement:
        limit_val = self._extract_limit(body, limit)
        # detect ACWordStatus before we strip all templates
        has_acwordstatus = bool(ACWORDSTATUS_RE.search(body))
        body_no_templates = TEMPLATE_RE.sub("", body)
        visible = visible_word_count(self.site, body_no_templates)
        expanded = rendered_word_count(self.site, body_no_templates)
        return Statement(
            fuzzy_username(raw_user, body),
            anchor,
            visible,
            expanded,
            limit_val,
            has_acwordstatus,
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
            closed = bool(HAT_OPEN_RE.match(sec_wikitext.lstrip()))

            statements: List[Statement] = []
            for st in lvl2.get_sections(levels=[3]):
                heading = mwpfh.parse(st.filter_headings()[0].title).strip_code().strip()
                if heading.lower() == CFG["PLACEHOLDER_HEADING"]:
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

        for i, sec in enumerate(sections):
            # When we hit a level-2 header, start a new RequestTable
            if sec.level == 2:
                # find the start of the *next* level-2 section (or EOF)
                next2_start = next(
                    (s.start for s in sections[i+1:] if s.level == 2),
                    len(text)
                )
                # grab everything from just after this header to just before the next lvl-2
                body_full = text[sec.start:next2_start]

                closed = bool(HAT_OPEN_RE.match(body_full.lstrip()))

                current = RequestTable(sec.title, slugify(sec.title), [], closed)
                tables.append(current)
                continue

            # everything below here belongs to the most recent level-2
            if current is None:
                continue

            # collect statements in level-3 and level-4 subsections
            if sec.level in {3, 4} and self._STMT.match(sec.title):
                body = sec.body(text)
                raw_user = self._STMT.sub("", sec.title)
                current.statements.append(
                    self._make_statement(raw_user, body, slugify(sec.title))
                )

            # special "Request concerning" blocks at level-3
            elif sec.level == 3 and sec.title.lower().startswith("request concerning"):
                body = self._collect_request_body(sec, text)
                if body:
                    user = self._requesting_user(sec, text)
                    current.statements.append(
                        self._make_statement(user, body, slugify(sec.title))
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


class EvidenceParser(BaseParser):
    """Parser for evidence pages in arbitration cases."""
    
    def __init__(self, site: mwclient.Site, case_name: str):
        super().__init__(site)
        self.case_name = case_name
        self.involved_parties = extract_involved_parties(site, case_name)
    
    def _get_word_limit(self, username: str) -> int:
        """Determine word limit based on whether user is a named party."""
        if username in self.involved_parties:
            return int(CFG["EVIDENCE_LIMIT_NAMED"])
        return int(CFG["EVIDENCE_LIMIT_OTHER"])
    
    def parse(self, text: str) -> List[RequestTable]:
        sections = scan_sections(text)
        statements: List[Statement] = []
        
        # Look for level-2 sections with "Evidence presented by" pattern
        for i, sec in enumerate(sections):
            if sec.level == 2 and sec.title.lower().startswith("evidence presented by"):
                raw_user = re.sub(r"^Evidence presented by\s+", "", sec.title, flags=re.I)
                
                # Skip placeholder sections (like "Evidence presented by {your user name}")
                if raw_user.lower() == "{your user name}":
                    continue
                
                # Find the start of the next level‑2 section (or EOF)
                next2_start = next(
                    (s.start for s in sections[i+1:] if s.level == 2),
                    len(text)
                )

                # ── NEW: include *everything* between the level‑2 heading
                # and the next level‑2 (or EOF).  This ensures that prose
                # between the heading and the first === subsection is counted.
                body = text[sec.start:next2_start]
                
                limit = self._get_word_limit(raw_user)
                statements.append(
                    self._make_statement(raw_user, body, slugify(sec.title), limit)
                )
        
        # Create a single table for the evidence page
        if statements:
            return [RequestTable(f"{self.case_name}/Evidence", slugify(f"{self.case_name}_Evidence"), statements)]
        return []

###############################################################################
# Dynamic board registry                                                      #
###############################################################################

def get_board_parsers() -> Dict[str, Tuple[str, Type[BaseParser]]]:
    return {
        "ARCA": (str(CFG["ARCA_PAGE"]), SimpleBoardParser),
        "AE": (str(CFG["AE_PAGE"]), AEParser),
        "ARC": (str(CFG["ARC_PAGE"]), SimpleBoardParser),
    }

###############################################################################
# Runtime helpers                                                             #
###############################################################################

def load_settings() -> None:
    """Load settings from environment variables, falling back to defaults."""
    # Start with defaults
    CFG.update(DEFAULT_CFG.copy())
    
    # Process all environment variables directly
    for env_var in DEFAULT_CFG.keys():
        value = os.getenv(env_var)
        if value is not None:
            # Handle numeric conversions
            if env_var in ['DEFAULT_LIMIT', 'EVIDENCE_LIMIT_NAMED', 'EVIDENCE_LIMIT_OTHER', 'RUN_INTERVAL']:
                try:
                    CFG[env_var] = int(value)
                except ValueError:
                    LOG.warning(f"Invalid {env_var} value: {value}, using default")
            elif env_var == 'OVER_FACTOR':
                try:
                    CFG[env_var] = float(value)
                except ValueError:
                    LOG.warning(f"Invalid {env_var} value: {value}, using default")
            else:
                # String values
                CFG[env_var] = value

def connect() -> mwclient.Site:
    """
    Login to MediaWiki, persisting cookies in
    a Mozilla‐format jar at CFG['SESSION_FILE'].
    """
    # Prepare cookie‐jar
    jar_path = os.path.expanduser(CFG["SESSION_FILE"])
    jar = http.cookiejar.MozillaCookieJar(jar_path)
    if os.path.exists(jar_path):
        try:
            jar.load(ignore_discard=True, ignore_expires=True)
            LOG.info("Loaded cookies from %s", jar_path)
        except Exception:
            LOG.warning("Could not load cookies, will start fresh")

    # Attach to requests
    sess = requests.Session()
    sess.cookies = jar

    site = mwclient.Site(
        CFG["SITE"],
        path=CFG["API_PATH"],
        clients_useragent=CFG["USER_AGENT"],
        pool=sess,
    )

    # If not logged in, do fresh login and save jar
    if not site.logged_in:
        LOG.info("Logging in fresh")
        site.login(CFG["BOT_USER"], CFG["BOT_PASSWORD"])
        os.makedirs(os.path.dirname(jar_path), exist_ok=True)
        try:
            jar.save(ignore_discard=True, ignore_expires=True)
            LOG.debug("Saved cookies to %s", jar_path)
        except Exception as e:
            LOG.warning("Failed to save cookies: %s", e)

    return site

def fetch_page(site: mwclient.Site, title: str) -> str:
    """Fetch wikitext body of `title` from the wiki."""
    return site.pages[title].text()

@dataclass
class ParsedData:
    """Container for parsed board and evidence data."""
    boards: Dict[str, List[RequestTable]]
    evidence: Dict[str, List[RequestTable]]

def collect_all_data(site: mwclient.Site) -> ParsedData:
    """Collect and parse all board and evidence page data once."""
    boards = {}
    evidence = {}
    
    # Process regular boards (ARCA, AE, ARC)
    for label, (page, ParserCls) in get_board_parsers().items():
        try:
            raw = fetch_page(site, page)
            parser = ParserCls(site)
            parser.board_page = page  # type: ignore
            boards[label] = parser.parse(raw)
        except Exception as e:
            LOG.warning("Could not process board %s: %s", label, e)
            boards[label] = []
    
    # Process evidence pages for open cases
    try:
        case_names = extract_open_cases(site)
        for case_name in case_names:
            evidence_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence"
            try:
                raw = fetch_page(site, evidence_page)
                parser = EvidenceParser(site, case_name)
                parser.board_page = evidence_page  # type: ignore
                evidence[case_name] = parser.parse(raw)
            except Exception as e:
                LOG.warning("Could not process evidence page for case %s: %s", case_name, e)
                evidence[case_name] = []
    except Exception as e:
        LOG.warning("Could not fetch open cases: %s", e)
    
    return ParsedData(boards, evidence)

def assemble_report_from_data(data: ParsedData) -> str:
    """Build the entire report wikitext from parsed data."""
    blocks: List[str] = [CFG["HEADER_TEXT"]]
    
    # Process regular boards (ARCA, AE, ARC)
    for label, (page, _) in get_board_parsers().items():
        tables = data.boards.get(label, [])
        if not tables:
            blocks.append(f"== {label} ==\n''No open requests.''")
        else:
            body = "\n\n".join(t.to_wikitext(page) for t in tables)
            blocks.append(f"== {label} ==\n{body}")
    
    # Process evidence pages for open cases
    if data.evidence:
        case_blocks = []
        for case_name, tables in data.evidence.items():
            evidence_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence"
            if tables:
                for table in tables:
                    case_blocks.append(table.to_wikitext(evidence_page))
            else:
                case_blocks.append(f"=== [[{evidence_page}|{case_name}/Evidence]] ===\n''No word‑limited statements.''")
        
        if case_blocks:
            blocks.append(f"== Case pages ==\n{chr(10).join(case_blocks)}")
        else:
            blocks.append("== Case pages ==\n''No open cases.''")
    else:
        blocks.append("== Case pages ==\n''No open cases.''")
    
    return "\n\n".join(blocks)

def assemble_report(site: mwclient.Site) -> str:
    """Build the entire report wikitext by fetching each board and parsing it."""
    return assemble_report_from_data(collect_all_data(site))

def assemble_data_template_from_data(parsed_data: ParsedData) -> str:
    """
    Build a nested #switch template containing full data for each statement.
    """
    # First, gather everything into a nested dict:
    data: dict[str, dict[str, dict[str, Statement]]] = {}
    
    # Process regular boards (ARCA, AE, ARC)
    for label, tables in parsed_data.boards.items():
        for tbl in tables:
            # Use the title instead of anchor for the key to preserve spaces
            data.setdefault(label, {}).setdefault(tbl.title, {})
            for stmt in tbl.statements:
                data[label][tbl.title][stmt.user] = stmt
    
    # Process evidence pages for open cases
    for case_name, tables in parsed_data.evidence.items():
        for tbl in tables:
            data.setdefault("Case pages", {}).setdefault(tbl.title, {})
            for stmt in tbl.statements:
                data["Case pages"][tbl.title][stmt.user] = stmt

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

def assemble_data_template(site: mwclient.Site) -> str:
    """
    Build a nested #switch template containing full data for each statement.
    """
    return assemble_data_template_from_data(collect_all_data(site))

def assemble_extended_report_from_data(data: ParsedData) -> str:
    """Build the extended report wikitext with template column from parsed data."""
    blocks: List[str] = [CFG["HEADER_TEXT"]]
    
    # Process regular boards (ARCA, AE, ARC)
    for label, (page, _) in get_board_parsers().items():
        tables = data.boards.get(label, [])
        if not tables:
            blocks.append(f"== {label} ==\n''No open requests.''")
        else:
            body = "\n\n".join(t.to_extended_wikitext(page, label) for t in tables)
            blocks.append(f"== {label} ==\n{body}")
    
    # Process evidence pages for open cases
    if data.evidence:
        case_blocks = []
        for case_name, tables in data.evidence.items():
            evidence_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence"
            if tables:
                for table in tables:
                    case_blocks.append(table.to_extended_wikitext(evidence_page, "Case pages"))
            else:
                case_blocks.append(f"=== [[{evidence_page}|{case_name}/Evidence]] ===\n''No word‑limited statements.''")
        
        if case_blocks:
            blocks.append(f"== Case pages ==\n{chr(10).join(case_blocks)}")
        else:
            blocks.append("== Case pages ==\n''No open cases.''")
    else:
        blocks.append("== Case pages ==\n''No open cases.''")
    
    return "\n\n".join(blocks)

def assemble_extended_report(site: mwclient.Site) -> str:
    """Build the extended report wikitext with template column."""
    return assemble_extended_report_from_data(collect_all_data(site))

def run_once(site: mwclient.Site) -> None:
    """
    Build report, compare to target page, and save if changed.
    """
    # ---- early-exit check ----
    target_title = str(CFG["TARGET_PAGE"])
    target = site.pages[target_title]
    # get the latest revision of the target page
    revs = target.revisions(dir="older", api_chunk_size=1)
    try:
        ts_target = _parse_ts(next(revs)["timestamp"])
    except StopIteration:
        # target has no revisions (brand new?), proceed with full run
        ts_target = datetime.min.replace(tzinfo=timezone.utc)

    # fetch the last edit time of each source page
    source_titles = [
        str(CFG["AE_PAGE"]),
        str(CFG["ARC_PAGE"]),
        str(CFG["ARCA_PAGE"]),
        str(CFG["OPEN_CASES_PAGE"]),
    ]
    
    # Add evidence pages for open cases
    try:
        case_names = extract_open_cases(site)
        for case_name in case_names:
            source_titles.append(f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence")
    except Exception as e:
        LOG.warning("Could not fetch open cases for early-exit check: %s", e)
    ts_sources = []
    for title in source_titles:
        rev_iter = site.pages[title].revisions(dir="older", api_chunk_size=1)
        try:
            ts_sources.append(_parse_ts(next(rev_iter)["timestamp"]))
        except StopIteration:
            # missing page or no revs → force a refresh
            ts_sources.append(datetime.min.replace(tzinfo=timezone.utc))

    # if our report is newer than *all* source pages, nothing to do
    if ts_target > max(ts_sources):
        LOG.info("No new edits on AE/ARC/ARCA since last report; exiting early.")
        return
    # ---- end early-exit check ----

    # Collect all data once
    data = collect_all_data(site)
    
    # Generate all three outputs using the shared data
    new_text = assemble_report_from_data(data)
    new_data = assemble_data_template_from_data(data)
    new_extended = assemble_extended_report_from_data(data)
    
    if new_text != target.text():
        # compute stats for the edit summary using the shared data
        all_tables: List[RequestTable] = []
        for tables in data.boards.values():
            all_tables.extend(tables)
        for tables in data.evidence.values():
            all_tables.extend(tables)

        # Only count open requests and their statements
        open_tables = [tbl for tbl in all_tables if not tbl.closed]
        all_statements = [s for tbl in open_tables for s in tbl.statements]
        x = sum(1 for s in all_statements if s.status == Status.OVER)
        y = sum(1 for s in all_statements if s.status == Status.WITHIN)
        z = len(all_statements)
        a = len(open_tables)

        summary = (
            f"updating table ({x} statements over, {y} statements within 10%, "
            f"{z} total statements, {a} pending requests) "
            f"([[User:KevinClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"
        )
        target.save(new_text, summary=summary, minor=False, bot=False)
        LOG.info("Updated target page.")

        # now update the data‐template page
        data_title = str(CFG["DATA_PAGE"])
        data_page = site.pages[data_title]
        if new_data != data_page.text():
            data_page.save(new_data,
                           summary= (f"Updating data template ({a} open requests, {z} statements)"
                                     f"([[User:KevinClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"),
                           minor=True,
                           bot=False)
            LOG.info("Updated data template page.")

        # now update the extended page
        extended_title = str(CFG["EXTENDED_PAGE"])
        extended_page = site.pages[extended_title]
        if new_extended != extended_page.text():
            extended_page.save(new_extended,
                           summary= (f"Updating extended report ({a} open requests, {z} statements)"
                                     f"([[User:KevinClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"),
                           minor=True,
                           bot=False)
            LOG.info("Updated extended report page.")
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
        interval = int(CFG["RUN_INTERVAL"])
        while True:
            try:
                run_once(site)
            except Exception:
                LOG.exception("Error; sleeping before retry")
            time.sleep(interval)

def _parse_ts(ts) -> datetime:
    """
    Parse a timestamp from mwclient.revisions():
    - if it's a struct_time, pull the tm_* fields
    - if it's an ISO string ("YYYY-MM-DDTHH:MM:SSZ"), parse it
    """
    if isinstance(ts, time.struct_time):
        # struct_time is already in UTC
        return datetime(
            ts.tm_year, ts.tm_mon, ts.tm_mday,
            ts.tm_hour, ts.tm_min, ts.tm_sec,
            tzinfo=timezone.utc
        )
    # otherwise assume string like "2025-05-20T18:34:56Z"
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

###############################################################################
# CLI entry‑point                                                             #
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WordcountClerkBot 2.4")
    ap.add_argument("-1", "--once", action="store_true", help="run once and exit")
    ap.add_argument("--debug", action="store_true", help="verbose debug logging")
    args = ap.parse_args()
    main(loop=not args.once, debug=args.debug)

