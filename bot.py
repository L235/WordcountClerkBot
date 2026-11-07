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
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import ClassVar, Type
from pathlib import Path
import json

from wiki_state import read_revisions, read_contents, load_checkpoint, save_checkpoint
import pickle
import hashlib

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
import pywikibot
from pywikibot.data import api as pwb_api
from pywikibot.site import APISite
from pywikibot.tools import normalize_username
import mwparserfromhell as mwpfh
from bs4 import BeautifulSoup
import tinycss2
from tinycss2.color4 import parse_color

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
    open_cases_page: str = "Template:Arbitration Committee open tasks/Cases"
    
    target_page: str = "User:ClerkBot/word counts"
    data_page: str = "User:ClerkBot/word counts/data"
    extended_page: str = "User:ClerkBot/word counts/extended"
    
    default_limit: int = 500
    evidence_limit_named: int = 1000
    evidence_limit_other: int = 500
    over_factor: float = 1.10
    
    run_interval: int = 600
    render_refresh_interval: int = 3600
    
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
            "RENDER_REFRESH_INTERVAL": ("render_refresh_interval", int),
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

    def flush(self, *, required: bool = False) -> None:
        """Persist the cache to disk."""
        if not self._loaded:
            return
        try:
            with open(self.path + ".tmp", "wb") as fh:
                pickle.dump(self._cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(self.path + ".tmp", self.path)
        except Exception as e:
            if required:
                raise
            LOG.debug("Could not write wordcount cache: %s", e)

    def clear(self) -> None:
        """Invalidate counts and HTML together when dependencies may have changed."""
        self.load()
        self._cache.clear()
        self._mem_cache_html.clear()
        # Persist invalidation before a refresh can be published/checkpointed.
        # Restarting before refreshed counts are flushed must only cause misses.
        self.flush(required=True)

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
# Invalidate counts produced by older algorithms, including raw-text fallbacks.
COUNT_CACHE_VERSION = 2

###############################################################################
# Regex helpers & constants                                                   #
###############################################################################

WORD_RE        = re.compile(r"\b\w+(?:['-]\w+)*\b")
HEADING_RE     = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$", re.M)
PAREN_RE = re.compile(r"^(.*?)(\s*\([^()]+\)\s*)$")
HAT_OPEN_RE = re.compile(r"\{\{\s*hat\s*(?=[|}])", re.I)
_TS_RE = re.compile(r"\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)")

def slugify(s: str) -> str:
    """Convert a heading into a MediaWiki anchor (no spaces, punctuation stripped)."""
    text = mwpfh.parse(str(s)).strip_code()
    return re.sub(r"\s+", "_", text).strip("_")


def heading_anchors(titles):
    """Allocate unique section IDs across the whole page, including non-authors."""
    used = set()
    for title in titles:
        base = slugify(title)
        anchor = base
        suffix = 2
        while anchor in used:
            anchor = f"{base}_{suffix}"
            suffix += 1
        used.add(anchor)
        yield anchor

def strip_parenthetical(username: str, body: str) -> str:
    """Remove parenthetical content from username if not referenced in body."""
    m = PAREN_RE.match(username)
    if not m:
        return username
    base = m.group(1).rstrip()
    if normalize_username(username) in {normalize_username(user) for user in user_links(body)}:
        return username
    return base

def user_links(body: str) -> list[str]:
    """Extract usernames from User: and User talk: links in wikitext."""
    links: list[str] = []
    for wl in mwpfh.parse(body, skip_style_tags=True).filter_wikilinks():
        title = str(wl.title).strip().lstrip(":")
        ns, _, rest = title.partition(":")
        if ns.strip().lower() in {"user", "user talk"} and rest:
            links.append(rest.split("/")[0].split("#")[0].strip())
    return links

STATUS_TEMPLATE_NAMES = {"acwordstatus", "arbitration committee word status"}


def normalize_template_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.replace("_", " ")).strip().lower()
    # A leading colon selects mainspace unless a namespace is explicit.
    if name.startswith(":"):
        target = name[1:].strip()
        if not target.startswith("template:"):
            return name
        name = target
    if name.startswith("template:"):
        name = name.partition(":")[2].strip()
    return name


@dataclass
class StatementControls:
    body: str
    user: str | None = None
    limit: int | None = None
    has_status: bool = False


def parse_statement_controls(body: str) -> StatementControls:
    """Interpret and exclude status/limit templates using the same name rules."""
    code = mwpfh.parse(body)
    result = StatementControls(body)
    remove = []
    for template in code.filter_templates():
        name = normalize_template_name(str(template.name))
        if name in STATUS_TEMPLATE_NAMES:
            result.has_status = True
            if result.user is None and template.has("user"):
                result.user = str(template.get("user").value).strip() or None
            remove.append(template)
        elif name == "approvedwordlimit":
            if result.limit is None and template.has("words"):
                raw = str(template.get("words").value).strip()
                if re.fullmatch(r"[0-9]+", raw) and int(raw) > 0:
                    result.limit = int(raw)
            remove.append(template)
    # Children first: deleting an outer template also deletes any nested nodes.
    for template in reversed(remove):
        code.remove(template, recursive=True)
    result.body = str(code)
    return result


def resolve_username(header: str, body: str, controls: StatementControls | None = None) -> str:
    """Use explicit identity evidence; unrelated user links never replace a heading."""
    controls = controls if controls is not None else parse_statement_controls(body)
    if controls.user:
        return normalize_username(controls.user) or ""
    plain = mwpfh.parse(header).strip_code().strip()
    simple = normalize_username(strip_parenthetical(plain, body)) or ""
    # A link identifies the author only when its label is the author portion,
    # not when it merely names someone in a trailing contextual qualifier.
    heading_code = mwpfh.parse(header)
    links = heading_code.filter_wikilinks()
    markers = []
    for index, link in enumerate(links):
        marker = f"CLERKBOTLINK{index}TOKEN"
        while marker in header:
            marker += "X"
        markers.append(marker)
        heading_code.replace(link, marker, recursive=True)
    author_markup = heading_code.strip_code().strip()
    if simple != normalize_username(plain):
        # Masking links preserves their positions while keeping parentheses
        # inside link targets/labels out of the qualifier boundary calculation.
        qualifier = PAREN_RE.match(author_markup)
        if qualifier:
            author_markup = qualifier.group(1).rstrip()
    heading_users = set()
    for link, marker in zip(links, markers):
        if marker not in author_markup:
            continue
        targets = user_links(str(link))
        label = normalize_username(mwpfh.parse(str(link)).strip_code().strip())
        if targets and label in {simple, normalize_username(plain)}:
            heading_users.add(normalize_username(targets[0]))
    if len(heading_users) == 1:
        return heading_users.pop() or ""
    # A signature may use a display alias (e.g. Vanamonde -> Vanamonde93).
    # Require the visible link label to match exactly, and a unique target.
    candidates = set()
    for link in mwpfh.parse(body).filter_wikilinks():
        targets = user_links(str(link))
        if not targets:
            continue
        target = normalize_username(targets[0])
        if target == simple:
            return simple
        if link.text is not None:
            label = normalize_username(mwpfh.parse(str(link.text)).strip_code().strip())
            if label == simple:
                candidates.add(target)
    return candidates.pop() if len(candidates) == 1 else simple

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
    anchors: tuple[str, ...] = ()

    def section_links(self, board_page: str) -> str:
        anchors = self.anchors or (self.anchor,)
        return " / ".join(
            f"[[{board_page}#{anchor}|{'link' if len(anchors) == 1 else f'link {index}'}]]"
            for index, anchor in enumerate(anchors, 1)
        )

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

    def author_totals(self) -> list[Statement]:
        """One total per author/request, retaining links to every source section."""
        totals: dict[str, Statement] = {}
        for statement in self.statements:
            user = normalize_username(statement.user)
            if not user:
                continue
            anchors = statement.anchors or (statement.anchor,)
            if user not in totals:
                totals[user] = replace(statement, user=user, anchors=anchors)
                continue
            total = totals[user]
            total.visible += statement.visible
            total.expanded += statement.expanded
            # An extension is a total allowance, not another additive allowance.
            total.limit = max(total.limit, statement.limit)
            total.has_status_template |= statement.has_status_template
            total.anchors = tuple(dict.fromkeys(total.anchors + anchors))
        return list(totals.values())

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
        for s in self.author_totals():
            if not s.user:
                continue
            style = f" style=\"background:{s.colour}\"" if s.colour else ""
            rows.append(
                f"|-{style}\n"
                f"| [[User:{s.user}|{s.user}]] "
                f"|| {s.section_links(board_page)} "
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
        for s in self.author_totals():
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
                f"|| {s.section_links(board_page)} "
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
    data = req.submit()  # Fail the pass rather than publish/cache an estimate.
    try:
        html = data["parse"]["text"]["*"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Parse API response is missing rendered HTML") from exc
    if not isinstance(html, str):
        raise RuntimeError("Parse API returned non-text HTML")

    # Cache the rendered HTML
    CACHE.cache_rendered_html(key, html)
    return html

def literal_style_exclusion(name: str, values: list) -> bool | None:
    """Validate literal property values; None means invalid/unsupported.

    This is not computed style: variables, math, and inherited styles require
    browser context. Only validated literal values participate in the cascade.
    """
    words = [token.lower_value for token in values if token.type == "ident"]
    if len(values) == 1 and words and words[0] in {
        "initial", "inherit", "unset", "revert", "revert-layer",
    }:
        return False
    if name == "display":
        if len(words) != len(values) or len(set(words)) != len(words):
            return None
        outside = {"block", "inline", "run-in"}
        inside = {"flow", "flow-root", "table", "flex", "grid", "ruby"}
        single = {
            "none", "contents", "inline-block", "inline-table", "inline-flex", "inline-grid",
            "table-row-group", "table-header-group", "table-footer-group", "table-row",
            "table-cell", "table-column-group", "table-column", "table-caption",
            "ruby-base", "ruby-text", "ruby-base-container", "ruby-text-container",
        }
        if len(words) == 1 and words[0] in single:
            return words[0] == "none"
        parts = set(words)
        allowed_inside = {"flow", "flow-root"} if "list-item" in parts else inside
        allowed = outside | allowed_inside | {"list-item"}
        if (parts <= allowed and len(parts & outside) <= 1
                and len(parts & allowed_inside) <= 1):
            return False
        return None

    lines = {"underline", "overline", "line-through", "blink"}
    exclusive_lines = {"none", "spelling-error", "grammar-error"}
    seen_lines = set()
    seen = set()
    length_units = {
        "px", "cm", "mm", "q", "in", "pc", "pt", "em", "rem", "ex", "rex", "ch", "rch",
        "cap", "rcap", "ic", "ric", "lh", "rlh", "vw", "vh", "vi", "vb", "vmin", "vmax",
        "svw", "svh", "svi", "svb", "svmin", "svmax", "lvw", "lvh", "lvi", "lvb", "lvmin", "lvmax",
        "dvw", "dvh", "dvi", "dvb", "dvmin", "dvmax", "cqw", "cqh", "cqi", "cqb", "cqmin", "cqmax",
    }
    for token in values:
        word = token.lower_value if token.type == "ident" else None
        if word in lines | exclusive_lines:
            if (word in seen_lines or (seen_lines and
                    (word in exclusive_lines or seen_lines & exclusive_lines))):
                return None
            seen_lines.add(word)
            continue
        if name == "text-decoration-line":
            return None
        if word in {"solid", "double", "dotted", "dashed", "wavy"}:
            category = "style"
        elif (word in {"auto", "from-font"} or token.type == "percentage"
              or (token.type == "number" and token.value == 0)
              or (token.type == "dimension" and token.lower_unit in length_units)):
            category = "thickness"
        elif parse_color(token) is not None:
            category = "color"
        else:
            return None
        if category in seen:
            return None
        seen.add(category)
    return "line-through" in seen_lines


def excluded_inline_style(style: str) -> bool:
    """Recognize literal hiding/strike declarations, respecting order/importance."""
    effective = {}
    for declaration in tinycss2.parse_declaration_list(style, skip_comments=True, skip_whitespace=True):
        if declaration.type != "declaration":
            continue
        name = declaration.lower_name
        if name not in {"display", "text-decoration", "text-decoration-line"}:
            continue
        values = [token for token in declaration.value if token.type not in {"whitespace", "comment"}]
        if not values or any(token.type == "error" for token in values):
            continue
        excluded = literal_style_exclusion(name, values)
        if excluded is None:
            continue
        key = "display" if name == "display" else "decoration"
        previous = effective.get(key)
        if previous and previous[0] and not declaration.important:
            continue
        effective[key] = (declaration.important, excluded)
    return any(excluded for _, excluded in effective.values())


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
    raw_key = f"{COUNT_CACHE_VERSION}|{host}|{path}|{title or ''}|{wikitext}".encode("utf-8")
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
    for tag in soup.select("[style]"):
        if tag.attrs is not None and excluded_inline_style(tag.get("style", "")):
            tag.decompose()
    for tag in soup.select("[hidden], [aria-hidden='true']"):
        tag.decompose()
    for tag in soup.select(".mw-collapsed, .mw-collapsible-content"):
        tag.decompose()
    for tag in soup.select("s, strike, del"):
        tag.decompose()
    for tag in soup.select("div#siteSub, div#contentSub, div#jump-to-nav"):
        tag.decompose()
    for tag in soup.select("span.localcomments"):
        tag.decompose()
    for tag in soup.select(".error"):
        tag.decompose()
    # Exclude quoted text from talk page quote templates ({{tq}}, {{tqb}}, etc.)
    for tag in soup.select(".inline-quote-talk, .talkquote"):
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
    heading_start: int
    subtree_end: int
    anchor: str

    def body(self, text: str) -> str:
        return text[self.start : self.end]

    def subtree_body(self, text: str) -> str:
        """Include child sections, stopping before the next peer/parent heading."""
        return text[self.start : self.subtree_end]


def scan_sections(text: str) -> list[RawSection]:
    # Only parsed headings reserve IDs. Equal-length markers retain source
    # offsets while excluding examples in comments, nowiki, and other literals.
    code = mwpfh.parse(text)
    for heading in code.filter_headings():
        code.replace(heading, mwpfh.nodes.Text("\0" * len(str(heading))), recursive=True)
    markers = str(code)
    heads = [h for h in HEADING_RE.finditer(text) if markers[h.start()] == "\0"]
    anchors = list(heading_anchors(h.group(2) for h in heads))
    return [
        RawSection(
            len(h.group(1)),
            h.group(2).strip(),
            h.end(),
            heads[i + 1].start() if i + 1 < len(heads) else len(text),
            h.start(),
            next((other.start() for other in heads[i + 1:]
                  if len(other.group(1)) <= len(h.group(1))), len(text)),
            anchors[i],
        )
        for i, h in enumerate(heads)
    ]

###############################################################################
# Open case helpers                                                           #
###############################################################################

def extract_open_cases(site: APISite) -> list[str]:
    """Extract case names from Template:ArbComOpenTasks/Cases."""
    page_text = fetch_page(site, CFG.open_cases_page)
    return parse_open_cases(page_text)

def parse_open_cases(page_text: str) -> list[str]:
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

def extract_involved_parties(site: APISite, case_name: str, page_text: str | None = None) -> list[str]:
    """Extract involved parties from a case page."""
    case_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}"
    try:
        if page_text is None:
            page_text = fetch_page(site, case_page)
    except Exception as exc:
        raise RuntimeError(f"Could not fetch party list: {case_page}") from exc
    
    parties = []
    sections = scan_sections(page_text)
    
    for sec in sections:
        if sec.level == 3 and sec.title.lower() == "involved parties":
            body = sec.body(page_text)
            for match in re.finditer(r'\{\{\s*(?:admin|adminlinks|userlinks)\s*\|\s*(?:1\s*=\s*)?([^|}]+)', body, re.I):
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

    def _default_limit(self, username: str) -> int:
        return CFG.default_limit

    def _extract_limit(self, body: str, default: int | None = None) -> int:
        return parse_statement_controls(body).limit or default or CFG.default_limit

    def _make_statement(
        self, raw_user: str, body: str, anchor: str, limit: int | None = None
    ) -> Statement:
        controls = parse_statement_controls(body)
        username = resolve_username(raw_user, body, controls)
        limit_val = controls.limit or limit or self._default_limit(username)
        has_acwordstatus = controls.has_status
        body_no_templates = controls.body
        # Pass board_page as title for proper template rendering context.
        # Templates like {{tq}} only work on talk/project pages.
        visible = visible_word_count(self.site, body_no_templates, self.board_page)
        expanded = rendered_word_count(self.site, body_no_templates, self.board_page)
        return Statement(
            username,
            anchor,
            visible,
            expanded,
            limit_val,
            has_acwordstatus,
        )

    def parse(self, text: str) -> list[RequestTable]:
        raise NotImplementedError


def is_placeholder_user(user: str) -> bool:
    user = user.strip().lower()
    return (user in {"", "(username)", "{other-editor}", "{your user name}"}
            or bool(re.fullmatch(r"\((?:un)?involved editor(?: \d+)?\)", user)))


class SimpleBoardParser(BaseParser):
    def parse(self, text: str) -> list[RequestTable]:
        code = mwpfh.parse(text, skip_style_tags=True)
        headings = code.filter_headings()
        anchors = dict(zip((id(h) for h in headings), heading_anchors(h.title for h in headings)))
        tables: list[RequestTable] = []
        for lvl2 in code.get_sections(levels=[2]):
            sec_title = mwpfh.parse(lvl2.filter_headings()[0].title, skip_style_tags=True).strip_code().strip()
            anchor = anchors[id(lvl2.filter_headings()[0])]
            sec_wikitext = str(lvl2)
            closed = bool(HAT_OPEN_RE.match(sec_wikitext.partition("\n")[2].lstrip()))

            statements: list[Statement] = []
            for st in lvl2.get_sections(levels=[3]):
                heading = mwpfh.parse(str(st.filter_headings()[0].title)).strip_code().strip()
                if heading.lower() == CFG.placeholder_heading:
                    continue
                if not heading.lower().startswith("statement by"):
                    continue
                heading_code = mwpfh.parse(str(st.filter_headings()[0].title))
                # Unwrap formatting before removing the prefix, retaining
                # wikilink targets in the author suffix (including piped links).
                for tag in reversed(heading_code.filter_tags()):
                    heading_code.replace(tag, tag.contents or "", recursive=True)
                for comment in heading_code.filter_comments():
                    heading_code.remove(comment, recursive=True)
                raw_heading = str(heading_code).strip()
                raw_user = re.sub(r"^Statement by\s+", "", raw_heading, flags=re.I)
                if is_placeholder_user(raw_user):
                    continue
                body = str(st).split("\n", 1)[1]
                statements.append(self._make_statement(raw_user, body, anchors[id(st.filter_headings()[0])]))

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
                body_full = sec.subtree_body(text)
                closed = bool(HAT_OPEN_RE.match(body_full.lstrip()))
                current = RequestTable(sec.title, sec.anchor, [], closed)
                tables.append(current)
                continue

            if current is None:
                continue

            if sec.level in {3, 4} and self._STMT.match(sec.title):
                raw_user = self._STMT.sub("", sec.title)
                if is_placeholder_user(raw_user):
                    continue
                body = self._without_nested_replies(sec, text)
                current.statements.append(
                    self._make_statement(raw_user, body, sec.anchor)
                )

            elif sec.level == 3 and sec.title.lower().startswith("request concerning"):
                body = self._collect_request_body(sec, text)
                if body:
                    user = self._requesting_user(sec, text)
                    current.statements.append(
                        self._make_statement(user, body, sec.anchor)
                    )

        return tables

    def _requesting_user(self, sec: RawSection, full: str) -> str:
        m = self._REQ_USER.search(sec.body(full))
        return m.group(1).strip() if m else ""

    def _without_nested_replies(self, sec: RawSection, full: str) -> str:
        seg = sec.subtree_body(full)
        # Replies have their own rows and controls. Remove their complete
        # subtrees, retaining ordinary evidence subsections before and after.
        for child in reversed(scan_sections(seg)):
            if child.level == 4 and self._STMT.match(child.title):
                seg = seg[:child.heading_start] + seg[child.subtree_end:]
        return seg

    def _collect_request_body(self, sec: RawSection, full: str) -> str:
        seg = self._without_nested_replies(sec, full)
        parts = []
        for m in self._PSEUDO.finditer(seg):
            start = m.end()
            nxt = re.search(r"^;", seg[start:], flags=re.M)
            end = nxt.start() + start if nxt else len(seg)
            parts.append(seg[start:end])
        return "\n".join(parts)


class EvidenceParser(BaseParser):
    def __init__(self, site: APISite, case_name: str, parties: list[str] | None = None):
        super().__init__(site)
        self.case_name = case_name
        self.involved_parties = {
            normalize_username(party)
            for party in (parties if parties is not None else extract_involved_parties(site, case_name))
        }
    
    def _default_limit(self, username: str) -> int:
        # Normalize both inputs without folding case after the first character.
        if normalize_username(username) in self.involved_parties:
            return CFG.evidence_limit_named
        return CFG.evidence_limit_other
    
    def parse(self, text: str) -> list[RequestTable]:
        sections = scan_sections(text)
        statements: list[Statement] = []
        
        for i, sec in enumerate(sections):
            if sec.level == 2 and sec.title.lower().startswith("evidence presented by"):
                raw_user = re.sub(r"^Evidence presented by\s+", "", sec.title, flags=re.I)
                
                if is_placeholder_user(raw_user):
                    continue
                
                body = sec.subtree_body(text)
                
                statements.append(
                    self._make_statement(raw_user, body, sec.anchor)
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

def collect_all_data(site: APISite, pages: dict[str, str] | None = None) -> ParsedData:
    read = lambda title: pages[title] if pages is not None else fetch_page(site, title)
    boards = {}
    evidence = {}
    
    for label, (page, ParserCls) in get_board_parsers().items():
        try:
            raw = read(page)
            parser = ParserCls(site)
            parser.board_page = page  # type: ignore
            boards[label] = parser.parse(raw)
        except Exception as exc:
            raise RuntimeError(f"Could not process board {label}") from exc
    
    try:
        case_names = parse_open_cases(read(CFG.open_cases_page))
        for case_name in case_names:
            evidence_page = f"Wikipedia:Arbitration/Requests/Case/{case_name}/Evidence"
            try:
                raw = read(evidence_page)
                parties = extract_involved_parties(
                    site, case_name, read(f"Wikipedia:Arbitration/Requests/Case/{case_name}")
                )
                parser = EvidenceParser(site, case_name, parties)
                parser.board_page = evidence_page  # type: ignore
                evidence[case_name] = parser.parse(raw)
            except Exception as exc:
                raise RuntimeError(f"Could not process evidence for {case_name}") from exc
    except Exception as exc:
        raise RuntimeError("Could not collect complete case data") from exc
    
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
            for stmt in tbl.author_totals():
                data[label][tbl.title][stmt.user] = stmt
    
    for case_name, tables in parsed_data.evidence.items():
        for tbl in tables:
            data.setdefault("Case pages", {}).setdefault(tbl.title, {})
            for stmt in tbl.author_totals():
                data["Case pages"][tbl.title][stmt.user] = stmt

    parts: list[str] = []
    parts.append('{{#switch: {{{page}}}')
    for label, requests in data.items():
        if label in get_board_parsers():
            # MediaWiki #switch fall-through: full title and short label agree.
            parts.append(' | ' + get_board_parsers()[label][0])
        parts.append(' | ' + label + ' = {{#switch: {{{section}}}')
        for req, users in requests.items():
            parts.append('     | ' + req + ' = {{#switch: {{{user}}}')
            for user, stmt in users.items():
                if " " in user:
                    parts.append("         | " + user.replace(" ", "_"))
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

@dataclass
class RunPlan:
    sources: dict[str, int]
    outputs: dict[str, int]
    texts: dict[str, str]
    cases: list[str]
    fingerprint: str
    refreshed_at: float
    refresh: bool


def checkpoint_path() -> Path:
    return Path(CFG.state_dir).expanduser() / "publication_checkpoint.json"


def source_titles(cases: list[str]) -> list[str]:
    titles = [CFG.ae_page, CFG.arc_page, CFG.arca_page, CFG.open_cases_page]
    for case in cases:
        root = f"Wikipedia:Arbitration/Requests/Case/{case}"
        titles.extend([root, root + "/Evidence"])
    return titles


def run_fingerprint(site: APISite) -> str:
    # Reconcile on configuration or algorithm changes, including new deployments.
    digest = hashlib.sha256(json.dumps(asdict(CFG), sort_keys=True).encode())
    digest.update(site.hostname().encode())
    for name in ("bot.py", "wiki_state.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def prepare_run(site: APISite) -> RunPlan | None:
    state = load_checkpoint(checkpoint_path())
    fingerprint = run_fingerprint(site)
    if state.get("fingerprint") != fingerprint:
        state = {}
    outputs = [CFG.target_page, CFG.data_page, CFG.extended_page]
    cases = state.get("cases", [])
    revisions = read_revisions(site, source_titles(cases) + outputs)
    texts = {}
    open_revision = revisions[CFG.open_cases_page]
    if not open_revision:
        raise RuntimeError("Open-case list is missing")
    if open_revision != state.get("sources", {}).get(CFG.open_cases_page):
        texts.update(read_contents(site, {CFG.open_cases_page: open_revision}))
        cases = parse_open_cases(texts[CFG.open_cases_page])
        new_titles = [title for title in source_titles(cases) if title not in revisions]
        revisions.update(read_revisions(site, new_titles))
    sources = {title: revisions[title] for title in source_titles(cases)}
    if any(not revision for revision in sources.values()):
        raise RuntimeError("A required source page is missing")
    output_revisions = {title: revisions[title] for title in outputs}
    now = time.time()
    previous_refresh = state.get("refreshed_at", 0)
    refresh = (not state or now < previous_refresh
               or now - previous_refresh >= CFG.render_refresh_interval)
    if (not refresh and sources == state.get("sources")
            and output_revisions == state.get("outputs")):
        return None
    texts.update(read_contents(site, {
        title: revision for title, revision in {**sources, **output_revisions}.items()
        if title not in texts
    }))
    return RunPlan(sources, output_revisions, texts, cases, fingerprint,
                   now if refresh else previous_refresh, refresh)

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
    all_statements = [s for tbl in open_tables for s in tbl.author_totals()]
    
    over = sum(1 for s in all_statements if s.status == Status.OVER)
    within = sum(1 for s in all_statements if s.status == Status.WITHIN)
    total = len(all_statements)
    pending = len(open_tables)

    stats = (
        f"{over} statements over, {within} statements within 10%, "
        f"{total} total statements, {pending} pending requests"
    )
    return ReportContent(text, data_tmpl, extended, stats)

def publish_changes(
    site: APISite, content: ReportContent, plan: RunPlan
) -> dict[str, int]:
    """Save all outputs; a failure leaves the previous checkpoint untouched."""
    revisions = {}
    for title, text, summary in (
        (CFG.target_page, content.text, "updating table"),
        (CFG.data_page, content.data, "Updating data template"),
        (CFG.extended_page, content.extended, "Updating extended report"),
    ):
        if text == plan.texts[title] and plan.outputs[title]:
            revisions[title] = plan.outputs[title]
            continue
        page = pywikibot.Page(site, title)
        page.text = text
        saved = page.save(
            summary=f"{summary} ({content.stats_summary}) {CFG.edit_summary_suffix}",
            minor=False, botflag=False,
        )
        if saved is False:
            raise RuntimeError(f"Save did not succeed: {title}")
        revisions[title] = page.latest_revision_id
        LOG.info("Updated %s", title)
    return revisions


def run_logic(site: APISite) -> None:
    """Commit the exact input revisions only after complete publication."""
    plan = prepare_run(site)
    if plan is None:
        LOG.info("No source/output changes; exiting early.")
        return
    if plan.refresh and CACHE is not None:
        CACHE.clear()
    data = collect_all_data(site, plan.texts)
    reports = generate_reports(data)
    outputs = publish_changes(site, reports, plan)
    save_checkpoint(checkpoint_path(), {
        "sources": plan.sources, "outputs": outputs, "cases": plan.cases,
        "fingerprint": plan.fingerprint, "refreshed_at": plan.refreshed_at,
    })


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
