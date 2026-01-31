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
excluding hidden content, collapsed sections, struck-through text, and
template error messages. Only the visible word count is compared against
the word limit policy.
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
from typing import ClassVar, Type
from datetime import datetime, timezone
import pickle
import hashlib

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
import pywikibot
from pywikibot.data import api as pwb_api
from pywikibot.site import APISite
import mwparserfromhell as mwpfh
from bs4 import BeautifulSoup

###############################################################################
# Configuration                                                               #
###############################################################################

@dataclass
class BotConfig:
    """Type-safe configuration container."""
    site: str = "en.wikipedia.org"
    api_path: str = "/w/"
    user_agent: str = "WordcountClerkBot/2.4 (https://github.com/L235/WordcountClerkBot)"
    
    arca_page: str = "Wikipedia:Arbitration/Requests/Clarification and Amendment"
    ae_page: str = "Wikipedia:Arbitration/Requests/Enforcement"
    arc_page: str = "Wikipedia:Arbitration/Requests/Case"
    open_cases_page: str = "Template:ArbComOpenTasks/Cases"
    
    target_page: str = "User:ClerkBot/word counts"
    data_page: str = "User:ClerkBot/word counts/data"
    extended_page: str = "User:ClerkBot/word counts/extended"
    
    default_limit: int = 500
    evidence_limit_named: int = 1000
    evidence_limit_other: int = 500
    over_factor: float = 1.10
    
    run_interval: int = 600
    
    red_hex: str = "#ff000040"
    amber_hex: str = "#ffff0040"
    
    header_text: str = ""
    placeholder_heading: str = "statement by {other-editor}"
    state_dir: str = "."
    edit_summary_suffix: str = "([[User:ClerkBot#t1|task 1]], [[WP:EXEMPTBOT|exempt]])"

    @classmethod
    def from_env(cls) -> BotConfig:
        """Load configuration from environment variables.

        For each environment variable, if it exists and can be converted to the
        expected type, it will override the default value. If conversion fails
        (e.g., DEFAULT_LIMIT="abc"), a warning is logged and the default value
        is used instead. This allows the bot to continue running with sensible
        defaults even if some configuration values are invalid.

        Returns:
            BotConfig: Configuration instance with values from environment or defaults.
        """
        # Mapping from env var name to field name and type
        # (env_var, field_name, type_converter)
        env_map = {
            "SITE": ("site", str),
            "API_PATH": ("api_path", str),
            "USER_AGENT": ("user_agent", str),
            "ARCA_PAGE": ("arca_page", str),
            "AE_PAGE": ("ae_page", str),
            "ARC_PAGE": ("arc_page", str),
            "OPEN_CASES_PAGE": ("open_cases_page", str),
            "TARGET_PAGE": ("target_page", str),
            "DATA_PAGE": ("data_page", str),
            "EXTENDED_PAGE": ("extended_page", str),
            "DEFAULT_LIMIT": ("default_limit", int),
            "EVIDENCE_LIMIT_NAMED": ("evidence_limit_named", int),
            "EVIDENCE_LIMIT_OTHER": ("evidence_limit_other", int),
            "OVER_FACTOR": ("over_factor", float),
            "RUN_INTERVAL": ("run_interval", int),
            "RED_HEX": ("red_hex", str),
            "AMBER_HEX": ("amber_hex", str),
            "HEADER_TEXT": ("header_text", str),
            "PLACEHOLDER_HEADING": ("placeholder_heading", str),
            "STATE_DIR": ("state_dir", str),
            "EDIT_SUMMARY_SUFFIX": ("edit_summary_suffix", str),
        }
        
        kwargs = {}
        for env_key, (field_name, converter) in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                try:
                    kwargs[field_name] = converter(val)
                except ValueError:
                    logging.getLogger(__name__).warning(
                        f"Invalid value for {env_key}: {val}. Using default."
                    )
        
        return cls(**kwargs)

# Global configuration instance (populated in main)
CFG: BotConfig = BotConfig()

###############################################################################
# Logging                                                                     #
###############################################################################

LOG = logging.getLogger(__name__)

def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    if not debug:
        # Allow override via env var if not explicitly debugging
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    LOG.setLevel(level)

    # Custom filtering: INFO -> stdout, WARNING+ -> stderr
    class MaxLevelFilter(logging.Filter):
        def __init__(self, max_level: int):
            super().__init__()
            self.max_level = max_level
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno <= self.max_level

    h_info = logging.StreamHandler(sys.stdout)
    h_info.setLevel(logging.DEBUG)  # Filter will restrict this
    h_info.addFilter(MaxLevelFilter(logging.INFO))
    
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h_info.setFormatter(fmt)
    h_err.setFormatter(fmt)

    # Clear existing handlers to avoid duplicates if called multiple times
    if LOG.hasHandlers():
        LOG.handlers.clear()

    LOG.addHandler(h_info)
    LOG.addHandler(h_err)


###############################################################################
# Word Count Cache                                                            #
###############################################################################

class WordCountCache:
    """Manages persistent caching of word counts for rendered wikitext.

    The cache has two levels:
    1. Persistent disk cache for (visible_words, rendered_words) tuples
    2. In-memory cache for rendered HTML (not persisted to disk)

    Cache pruning uses FIFO insertion-order, not LRU. When the cache exceeds
    100k entries, the oldest 10k by insertion time are removed. Note that
    updating an existing key does not change its position in the insertion order.
    """

    def __init__(self, state_dir: str):
        self.path = os.path.join(os.path.expanduser(state_dir), "wordcount_cache.pkl")
        self._cache: dict[str, tuple[int, int]] = {}
        self._writes = 0
        self._loaded = False
        self._mem_cache_html: dict[tuple[str, str, str, str], str] = {}  # In-memory HTML cache

    def load(self) -> None:
        """Load the cache from disk."""
        if self._loaded:
            return
        dir_path = os.path.dirname(self.path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            with open(self.path, "rb") as fh:
                self._cache = pickle.load(fh)
        except Exception:
            self._cache = {}
        self._writes = 0
        self._loaded = True

    def flush(self) -> None:
        """Persist the cache to disk."""
        if not self._loaded:
            return
        try:
            with open(self.path, "wb") as fh:
                pickle.dump(self._cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            LOG.debug("Could not write wordcount cache: %s", e)

    def get(self, key: str) -> tuple[int, int] | None:
        return self._cache.get(key)

    def put(self, key: str, value: tuple[int, int]) -> None:
        self._cache[key] = value

        # Prune cache (~100k entries max, drop 10k oldest by insertion order)
        if len(self._cache) > 100000:
            for k in list(self._cache.keys())[:10000]:
                del self._cache[k]

        self._writes += 1
        if self._writes % 50 == 0:  # amortize disk I/O
            self.flush()

    def get_rendered_html(self, key: tuple[str, str, str, str]) -> str | None:
        """Get rendered HTML from in-memory cache."""
        return self._mem_cache_html.get(key)

    def cache_rendered_html(self, key: tuple[str, str, str, str], html: str) -> None:
        """Cache rendered HTML in memory with FIFO pruning at 256 entries."""
        self._mem_cache_html[key] = html
        if len(self._mem_cache_html) > 256:
            # Drop oldest 64 by insertion order
            for k in list(self._mem_cache_html.keys())[:64]:
                del self._mem_cache_html[k]

# Global cache instance
CACHE: WordCountCache | None = None

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
    text = mwpfh.parse(s, skip_style_tags=True).strip_code()
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

def user_links(body: str) -> list[str]:
    """Extract usernames from User: and User talk: links in wikitext."""
    links: list[str] = []
    for wl in mwpfh.parse(body, skip_style_tags=True).filter_wikilinks():
        title = str(wl.title)
        ns, _, rest = title.partition(":")
        if ns.lower() in {"user", "user talk"} and rest:
            links.append(rest.split("/")[0].strip())
    return links

def fuzzy_username(header: str, body: str) -> str:
    """
    Find best matching username from header and body links.
    """
    def capitalize_first(s: str) -> str:
        return s[0].upper() + s[1:] if s else s

    # --- Template override (top priority) ---
    try:
        code = mwpfh.parse(body)
        def _norm(name: str) -> str:
            return re.sub(r"\s+", " ", name.replace("_", " ")).strip().lower()
        for tmpl in code.filter_templates():
            tname = _norm(str(tmpl.name))
            if tname in {"acwordstatus", "arbitration committee word status"} and tmpl.has("user"):
                raw = str(tmpl.get("user").value).strip()
                if raw:
                    return capitalize_first(raw)
    except Exception:
        pass

    # ---- Existing heuristics (header + body links) ----
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
        if self.visible <= self.limit * CFG.over_factor:
            return Status.WITHIN
        return Status.OVER

    @property
    def colour(self) -> str | None:
        mapping = {
            Status.WITHIN: CFG.amber_hex,
            Status.OVER: CFG.red_hex,
        }
        return mapping.get(self.status)


@dataclass
class RequestTable:
    title: str
    anchor: str
    statements: list[Statement]
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
            # Construct {{ACWordStatus|...}} template.
            # Brace counting explanation:
            # 1. First 4 braces '{{{{': Output literal '{{' (f-string escaping: {{ -> {).
            # 2. Last 5 braces '}}}}}':
            #    - The first '}' is the closing brace for the f-string variable '{s.user}'.
            #    - The remaining 4 '}}}}' output literal '}}' (f-string escaping: }} -> }).
            template = f"{{{{ACWordStatus|page={label}|section={self.title}|user={s.user}}}}}"

            # Highlight the cell only when the template is missing
            if s.has_status_template:
                template_cell = f"|| <nowiki>{template}</nowiki>"
            else:
                template_cell = (
                    f'|| style="background:{CFG.amber_hex}" | '
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
# Word-count helpers
###############################################################################

def _api_render(site: APISite, wikitext: str, title: str | None = None) -> str:
    """
    Render a wikitext snippet to HTML via the MediaWiki parse API.
    Uses in-memory cache for rendered HTML.

    Args:
        site: The pywikibot site to use for the API call.
        wikitext: The wikitext to render.
        title: Optional page title for context. Templates like {{tq}} behave
               differently based on whether they're on a talk/project page.
    """
    if CACHE is None:
        raise RuntimeError("Cache not initialized")

    try:
        host = site.hostname()
        path = site.apipath()
    except Exception:
        host = str(site)
        path = ""

    key = (host, path, title or "", wikitext)

    # Check in-memory HTML cache
    cached_html = CACHE.get_rendered_html(key)
    if cached_html is not None:
        return cached_html

    params = {
        "action": "parse",
        "text": wikitext,
        "prop": "text",
        "contentmodel": "wikitext",
    }
    if title:
        params["title"] = title
    req = pwb_api.Request(site=site, parameters=params)
    LOG.info("Making render API call for wikitext (length: %d)", len(wikitext))
    try:
        data = req.submit()
    except Exception as e:
        LOG.warning("parse API failed: %s", e)
        html = wikitext  # fallback
    else:
        html = data.get("parse", {}).get("text", {}).get("*", "") or wikitext

    # Cache the rendered HTML
    CACHE.cache_rendered_html(key, html)
    return html

def _count_rendered_visible(
    site: APISite, wikitext: str, title: str | None = None
) -> tuple[int, int]:
    """
    Return (visible_words, rendered_words) for the given snippet.

    Args:
        site: The pywikibot site to use for the API call.
        wikitext: The wikitext to count words in.
        title: Optional page title for context. Templates like {{tq}} behave
               differently based on whether they're on a talk/project page.
    """
    if CACHE is None:
        raise RuntimeError("Cache not initialized")
    CACHE.load() # Ensure loaded

    try:
        host = site.hostname()
        path = site.apipath()
    except Exception:
        host = str(site)
        path = ""
    raw_key = f"{host}|{path}|{title or ''}|{wikitext}".encode("utf-8")
    cache_key = hashlib.sha1(raw_key).hexdigest()

    cached_val = CACHE.get(cache_key)
    if cached_val is not None:
        return cached_val

    # Cache miss: render HTML, compute both counts once
    html = _api_render(site, wikitext, title)

    # Rendered words (all text, including hidden)
    rendered = len(WORD_RE.findall(re.sub(r"<[^>]+>", "", html)))

    # Visible words (filtered HTML)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("ol.references, div.references, div.reflist"):
        tag.decompose()
    for tag in soup.select('[style*="display:none" i]'):
        tag.decompose()
    for tag in soup.select("[hidden], [aria-hidden='true']"):
        tag.decompose()
    for tag in soup.select(".mw-collapsed, .mw-collapsible-content"):
        tag.decompose()
    for tag in soup.select('[style*="text-decoration:line-through" i], s, strike, del'):
        tag.decompose()
    for tag in soup.select("div#siteSub, div#contentSub, div#jump-to-nav"):
        tag.decompose()
    for tag in soup.select("span.localcomments"):
        tag.decompose()
    for tag in soup.select(".error"):
        tag.decompose()
    # Exclude quoted text from {{tq}} template (talk page quotes)
    for tag in soup.select(".inline-quote-talk, .inline-quote-talk-italic"):
        tag.decompose()
    text = _TS_RE.sub("", soup.get_text())
    tokens = [t for t in re.split(r"\s+", text) if t and re.search(r"[A-Za-z0-9]", t)]
    visible = len(tokens)

    CACHE.put(cache_key, (visible, rendered))
    return visible, rendered

def rendered_word_count(site: APISite, wikitext: str, title: str | None = None) -> int:
    """Return the rendered (uncollapsed) word count for the given snippet."""
    _v, r = _count_rendered_visible(site, wikitext, title)
    return r


def visible_word_count(site: APISite, wikitext: str, title: str | None = None) -> int:
    """Return the visible word count for the given snippet."""
    v, _r = _count_rendered_visible(site, wikitext, title)
    return v

###############################################################################
# Section scanner (for AE)
###############################################################################

@dataclass
class RawSection:
    level: int
    title: str
    start: int
    end: int

    def body(self, text: str) -> str:
        return text[self.start : self.end]


def scan_sections(text: str) -> list[RawSection]:
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

def extract_open_cases(site: APISite) -> list[str]:
    """Extract case names from Template:ArbComOpenTasks/Cases."""
    page_text = fetch_page(site, CFG.open_cases_page)
    case_names = []
    
    code = mwpfh.parse(page_text, skip_style_tags=True)
    for template in code.filter_templates():
        template_name = str(template.name).strip()
        if template_name == "ArbComOpenTasks/line":
            mode_param = template.get("mode").value
            name_param = template.get("name").value
            if mode_param and str(mode_param).strip() == "case" and name_param:
                case_names.append(str(name_param).strip())
    return case_names

def extract_involved_parties(site: APISite, case_name: str) -> list[str]:
    """Extract involved parties from a case page."""
    case_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}"
    try:
        page_text = fetch_page(site, case_page)
    except Exception:
        LOG.warning("Could not fetch case page: %s", case_page)
        return []
    
    parties = []
    sections = scan_sections(page_text)
    
    for sec in sections:
        if sec.level == 3 and sec.title.lower() == "involved parties":
            body = sec.body(page_text)
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

    def __init__(self, site: APISite):
        self.site = site

    def _extract_limit(self, body: str, default: int | None = None) -> int:
        m = AWL_RE.search(body)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                LOG.debug("Invalid ApprovedWordLimit value: %s", m.group(1))
        return default or CFG.default_limit

    def _make_statement(
        self, raw_user: str, body: str, anchor: str, limit: int | None = None
    ) -> Statement:
        limit_val = self._extract_limit(body, limit)
        has_acwordstatus = bool(ACWORDSTATUS_RE.search(body))
        body_no_templates = TEMPLATE_RE.sub("", body)
        # Pass board_page as title for proper template rendering context.
        # Templates like {{tq}} only work on talk/project pages.
        visible = visible_word_count(self.site, body_no_templates, self.board_page)
        expanded = rendered_word_count(self.site, body_no_templates, self.board_page)
        return Statement(
            fuzzy_username(raw_user, body),
            anchor,
            visible,
            expanded,
            limit_val,
            has_acwordstatus,
        )

    def parse(self, text: str) -> list[RequestTable]:
        raise NotImplementedError


class SimpleBoardParser(BaseParser):
    def parse(self, text: str) -> list[RequestTable]:
        code = mwpfh.parse(text, skip_style_tags=True)
        tables: list[RequestTable] = []
        for lvl2 in code.get_sections(levels=[2]):
            sec_title = mwpfh.parse(lvl2.filter_headings()[0].title, skip_style_tags=True).strip_code().strip()
            anchor = slugify(sec_title)
            sec_wikitext = str(lvl2)
            closed = bool(HAT_OPEN_RE.match(sec_wikitext.lstrip()))

            statements: list[Statement] = []
            for st in lvl2.get_sections(levels=[3]):
                heading = mwpfh.parse(st.filter_headings()[0].title, skip_style_tags=True).strip_code().strip()
                if heading.lower() == CFG.placeholder_heading:
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

    def parse(self, text: str) -> list[RequestTable]:
        sections = scan_sections(text)
        tables: list[RequestTable] = []
        current: RequestTable | None = None

        for i, sec in enumerate(sections):
            if sec.level == 2:
                next2_start = next(
                    (s.start for s in sections[i+1:] if s.level == 2),
                    len(text)
                )
                body_full = text[sec.start:next2_start]
                closed = bool(HAT_OPEN_RE.match(body_full.lstrip()))
                current = RequestTable(sec.title, slugify(sec.title), [], closed)
                tables.append(current)
                continue

            if current is None:
                continue

            if sec.level in {3, 4} and self._STMT.match(sec.title):
                body = sec.body(text)
                raw_user = self._STMT.sub("", sec.title)
                current.statements.append(
                    self._make_statement(raw_user, body, slugify(sec.title))
                )

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
    def __init__(self, site: APISite, case_name: str):
        super().__init__(site)
        self.case_name = case_name
        self.involved_parties = extract_involved_parties(site, case_name)
    
    def _get_word_limit(self, username: str) -> int:
        if username in self.involved_parties:
            return CFG.evidence_limit_named
        return CFG.evidence_limit_other
    
    def parse(self, text: str) -> list[RequestTable]:
        sections = scan_sections(text)
        statements: list[Statement] = []
        
        for i, sec in enumerate(sections):
            if sec.level == 2 and sec.title.lower().startswith("evidence presented by"):
                raw_user = re.sub(r"^Evidence presented by\s+", "", sec.title, flags=re.I)
                
                if raw_user.lower() == "{your user name}":
                    continue
                
                next2_start = next(
                    (s.start for s in sections[i+1:] if s.level == 2),
                    len(text)
                )
                body = text[sec.start:next2_start]
                
                limit = self._get_word_limit(raw_user)
                statements.append(
                    self._make_statement(raw_user, body, slugify(sec.title), limit)
                )
        
        if statements:
            return [RequestTable(f"{self.case_name}/Evidence", slugify(f"{self.case_name}_Evidence"), statements)]
        return []

###############################################################################
# Dynamic board registry                                                      #
###############################################################################

def get_board_parsers() -> dict[str, tuple[str, Type[BaseParser]]]:
    return {
        "ARCA": (CFG.arca_page, SimpleBoardParser),
        "AE": (CFG.ae_page, AEParser),
        "ARC": (CFG.arc_page, SimpleBoardParser),
    }

###############################################################################
# Runtime helpers                                                             #
###############################################################################

def connect() -> APISite:
    """
    Connect/login via pywikibot.
    """
    host = CFG.site.lower()
    if host.endswith(".org"):
        parts = host.split(".")
        code = parts[0]
        family = parts[1] if len(parts) > 1 else "wikipedia"
    else:
        code, family = "en", "wikipedia"

    site = pywikibot.Site(code=code, fam=family)
    site.login()
    
    if CFG.user_agent:
        try:
            import pywikibot.config as pwb_config
            pwb_config.user_agent = CFG.user_agent
        except Exception:
            LOG.debug("Could not set custom user-agent via pywikibot.config; ignoring.")

    return site

def fetch_page(site: APISite, title: str) -> str:
    return pywikibot.Page(site, title).text

@dataclass
class ParsedData:
    """Container for parsed board and evidence data."""
    boards: dict[str, list[RequestTable]]
    evidence: dict[str, list[RequestTable]]

def collect_all_data(site: APISite) -> ParsedData:
    boards = {}
    evidence = {}
    
    for label, (page, ParserCls) in get_board_parsers().items():
        try:
            raw = fetch_page(site, page)
            parser = ParserCls(site)
            parser.board_page = page  # type: ignore
            boards[label] = parser.parse(raw)
        except Exception as e:
            LOG.warning("Could not process board %s: %s", label, e)
            boards[label] = []
    
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
    blocks: list[str] = [CFG.header_text]
    
    for label, (page, _) in get_board_parsers().items():
        tables = data.boards.get(label, [])
        if not tables:
            blocks.append(f"== {label} ==\n''No open requests.''")
        else:
            body = "\n\n".join(t.to_wikitext(page) for t in tables)
            blocks.append(f"== {label} ==\n{body}")
    
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

def assemble_data_template_from_data(parsed_data: ParsedData) -> str:
    data: dict[str, dict[str, dict[str, Statement]]] = {}
    
    for label, tables in parsed_data.boards.items():
        for tbl in tables:
            data.setdefault(label, {}).setdefault(tbl.title, {})
            for stmt in tbl.statements:
                data[label][tbl.title][stmt.user] = stmt
    
    for case_name, tables in parsed_data.evidence.items():
        for tbl in tables:
            data.setdefault("Case pages", {}).setdefault(tbl.title, {})
            for stmt in tbl.statements:
                data["Case pages"][tbl.title][stmt.user] = stmt

    parts: list[str] = []
    parts.append('{{#switch: {{{page}}}')
    for label, requests in data.items():
        parts.append(' | ' + label + ' = {{#switch: {{{section}}}')
        for req, users in requests.items():
            parts.append('     | ' + req + ' = {{#switch: {{{user}}}')
            for user, stmt in users.items():
                parts.append('         | ' + user + ' = {{#switch: {{{type}}}')
                parts.append(f'             | words       = {stmt.visible}')
                parts.append(f'             | uncollapsed = {stmt.expanded}')
                parts.append(f'             | limit       = {stmt.limit}')
                parts.append(f'             | status      = {stmt.status.value}')
                parts.append('           }}')
            parts.append('       }}')
        parts.append('   }}')
    parts.append('}}')

    return "\n".join(parts)

def assemble_extended_report_from_data(data: ParsedData) -> str:
    blocks: list[str] = [CFG.header_text]
    
    for label, (page, _) in get_board_parsers().items():
        tables = data.boards.get(label, [])
        if not tables:
            blocks.append(f"== {label} ==\n''No open requests.''")
        else:
            body = "\n\n".join(t.to_extended_wikitext(page, label) for t in tables)
            blocks.append(f"== {label} ==\n{body}")
    
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

# ---------------------------------------------------------------------------
# Logic Refactoring: Decomposed Steps
# ---------------------------------------------------------------------------

def should_run(site: APISite) -> bool:
    """Check timestamps to see if a run is needed."""
    target_page = pywikibot.Page(site, CFG.target_page)
    if not target_page.exists():
        return True # Always run if target doesn't exist
    
    ts_target = target_page.latest_revision.timestamp

    # Check source pages
    source_titles = [CFG.ae_page, CFG.arc_page, CFG.arca_page, CFG.open_cases_page]
    try:
        case_names = extract_open_cases(site)
        for case_name in case_names:
            source_titles.append(f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence")
    except Exception as e:
        LOG.warning("Could not fetch open cases for early-exit check: %s", e)

    ts_sources = []
    for title in source_titles:
        p = pywikibot.Page(site, title)
        if p.exists():
            ts_sources.append(p.latest_revision.timestamp)
        else:
            ts_sources.append(datetime.min.replace(tzinfo=timezone.utc))
    
    if not ts_sources:
        return True  # Run if we can't determine source timestamps
    return max(ts_sources) > ts_target

@dataclass
class ReportContent:
    text: str
    data: str
    extended: str
    stats_summary: str

def generate_reports(data: ParsedData) -> ReportContent:
    """Generate all report variations from parsed data."""
    text = assemble_report_from_data(data)
    data_tmpl = assemble_data_template_from_data(data)
    extended = assemble_extended_report_from_data(data)
    
    # Compute stats for edit summary
    all_tables: list[RequestTable] = []
    for tables in data.boards.values():
        all_tables.extend(tables)
    for tables in data.evidence.values():
        all_tables.extend(tables)

    open_tables = [tbl for tbl in all_tables if not tbl.closed]
    all_statements = [s for tbl in open_tables for s in tbl.statements]
    
    over = sum(1 for s in all_statements if s.status == Status.OVER)
    within = sum(1 for s in all_statements if s.status == Status.WITHIN)
    total = len(all_statements)
    pending = len(open_tables)

    stats = (
        f"{over} statements over, {within} statements within 10%, "
        f"{total} total statements, {pending} pending requests"
    )
    return ReportContent(text, data_tmpl, extended, stats)

def publish_changes(site: APISite, content: ReportContent) -> None:
    """Compare generated content with wiki and save if different."""
    
    # 1. Main Report
    target = pywikibot.Page(site, CFG.target_page)
    current_text = target.text if target.exists() else ""
    
    if content.text != current_text:
        summary = f"updating table ({content.stats_summary}) {CFG.edit_summary_suffix}"
        target.text = content.text
        target.save(summary=summary, minor=False, botflag=False)
        LOG.info("Updated target page.")
    else:
        LOG.info("No changes to target page.")

    # 2. Data Template
    data_page = pywikibot.Page(site, CFG.data_page)
    current_data = data_page.text if data_page.exists() else ""
    
    if content.data != current_data:
        summary = f"Updating data template ({content.stats_summary}) {CFG.edit_summary_suffix}"
        data_page.text = content.data
        data_page.save(summary=summary, minor=False, botflag=False)
        LOG.info("Updated data template page.")

    # 3. Extended Report
    ext_page = pywikibot.Page(site, CFG.extended_page)
    current_ext = ext_page.text if ext_page.exists() else ""
    
    if content.extended != current_ext:
        summary = f"Updating extended report ({content.stats_summary}) {CFG.edit_summary_suffix}"
        ext_page.text = content.extended
        ext_page.save(summary=summary, minor=False, botflag=False)
        LOG.info("Updated extended report page.")

def run_logic(site: APISite) -> None:
    """Orchestrate a single run."""
    if not should_run(site):
        LOG.info("No new edits on source pages; exiting early.")
        return

    LOG.info("Changes detected. Collecting data...")
    data = collect_all_data(site)
    reports = generate_reports(data)
    publish_changes(site, reports)

def main(loop: bool, debug: bool) -> None:
    """
    Entry point: load settings, connect, and either run once or loop.
    """
    global CFG, CACHE
    
    setup_logging(debug)
    CFG = BotConfig.from_env()
    CACHE = WordCountCache(CFG.state_dir)
    
    site = connect()
    
    if not loop:
        run_logic(site)
        CACHE.flush()
    else:
        interval = CFG.run_interval
        while True:
            try:
                run_logic(site)
            except Exception:
                LOG.exception("Error during run; sleeping before retry")
            
            CACHE.flush()
            LOG.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)

###############################################################################
# CLI entry-point
###############################################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WordcountClerkBot 2.4")
    ap.add_argument("-1", "--once", action="store_true", help="run once and exit")
    ap.add_argument("--debug", action="store_true", help="verbose debug logging")
    args = ap.parse_args()
    main(loop=not args.once, debug=args.debug)