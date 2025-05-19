#!/usr/bin/env python3
"""
arca_wordcount_bot.py – revision 5
===================================
A MediaWiki bot that counts rendered words in each "Statement by ..." on
[[Wikipedia:Arbitration/Requests/Clarification and Amendment]] (ARCA) and
publishes one wikitable per request to the bot's userspace.

Configuration lives in **settings.json** (see sample below). Most runtime
constants can now be overridden there instead of editing code.

```jsonc
{
  /* —–– Wiki connection —–– */
  "site": "en.wikipedia.org",
  "path": "/w/",
  "user": "BotUser@PasswordName",
  "bot_password": "s3cret",
  "ua": "KevinClerkBot/0.5 (https://github.com/example)",
  "cookie_path": "~/kevinclerkbot/cookies.txt",

  /* —–– Bot behaviour —–– */
  "arca_page": "Wikipedia:Arbitration/Requests/Clarification and Amendment",
  "target_page": "User:KevinClerkBot/ARCA word counts",
  "default_limit": 500,
  "run_interval": 600,          /* seconds between updates */
  "over_factor": 1.10,          /* >10 % above limit → over */
  "red_hex": "#ffcccc",        /* highlight colour */
  "placeholder_heading": "statement by {other-editor}"
}
```
"""

import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from http.cookiejar import MozillaCookieJar
from typing import List, Tuple
import argparse

import mwclient
import mwparserfromhell as parser
import requests

# -------------------------------------------------------------------------
#  Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("arca_wordcount_bot.log"),
        logging.StreamHandler()  # This will log to stderr
    ]
)

# -------------------------------------------------------------------------
#  Settings loader
# -------------------------------------------------------------------------

def load_settings(path: str = "settings.json") -> dict:
    """Load JSON settings file. If missing, raise an explicit error."""
    logging.info("Loading settings from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file '{path}' not found.")
    
    with open(path) as f:
        settings = json.load(f)
    
    # Debug output
    logging.info("Loaded settings:")
    for key in ["site", "path", "user", "ua", "cookie_path", "arca_page", "target_page"]:
        if key in settings:
            logging.info("  %s: %s", key, settings[key])
    logging.info("  bot_password: %s", "set" if settings.get("bot_password") else "not set")
    
    return settings


SETTINGS = load_settings()

# Convenience helpers pulling from SETTINGS with sensible defaults
SITE_DOMAIN = SETTINGS["site"]
API_PATH = SETTINGS.get("path", "/w/")
USER_AGENT = SETTINGS.get("ua", "KevinClerkBot/unknown")
COOKIE_FILE = os.path.expanduser(SETTINGS["cookie_path"])

ARCA_PAGE = SETTINGS.get(
    "arca_page",
    "Wikipedia:Arbitration/Requests/Clarification and Amendment",
)
TARGET_PAGE = SETTINGS.get("target_page", "User:KevinClerkBot/ARCA word counts")
DEFAULT_LIMIT = int(SETTINGS.get("default_limit", 500))
RUN_INTERVAL = int(SETTINGS.get("run_interval", 600))
OVER_FACTOR = float(SETTINGS.get("over_factor", 1.10))
RED_HEX = SETTINGS.get("red_hex", "#ffcccc")
PLACEHOLDER_HEADING = SETTINGS.get(
    "placeholder_heading", "statement by {other-editor}"
).lower()

# -------------------------------------------------------------------------
#  MediaWiki connection
# -------------------------------------------------------------------------

def connect() -> mwclient.Site:
    """Return a logged‑in mwclient.Site using BotPassword credentials."""
    cj = MozillaCookieJar(COOKIE_FILE)
    if os.path.exists(COOKIE_FILE):
        cj.load(ignore_discard=True, ignore_expires=True)

    sess = requests.Session()
    sess.cookies = cj

    site = mwclient.Site(
        SITE_DOMAIN,
        path=API_PATH,
        clients_useragent=USER_AGENT,
        pool=sess,
    )
    
    # Debug logging
    logging.info("Attempting login with user: %s", SETTINGS["user"])
    logging.info("Site: %s, API path: %s", SITE_DOMAIN, API_PATH)
    
    if not site.logged_in:
        try:
            site.login(SETTINGS["user"], SETTINGS["bot_password"])
        except Exception as e:
            logging.error("Login failed: %s", str(e))
            raise

    cj.save(ignore_discard=True, ignore_expires=True)
    logging.info("Logged in as %s", SETTINGS["user"])
    return site

# -------------------------------------------------------------------------
#  Word counting helpers
# -------------------------------------------------------------------------
WORD_RE = re.compile(r"\b\w+(?:['-]\w+)*\b")
AWL_RE = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*?\bwords\s*=\s*(\d+)", flags=re.I | re.S)
TEMPLATE_RE = re.compile(r"\{\{\s*ApprovedWordLimit[^}]*}}", flags=re.I | re.S)


def strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def rendered_word_count(site: mwclient.Site, wikitext: str) -> int:
    resp = site.api("parse", text=wikitext, prop="text", contentmodel="wikitext", format="json")
    html = resp["parse"]["text"]["*"]
    return len(WORD_RE.findall(strip_html(html)))

# -------------------------------------------------------------------------
#  Username handling
# -------------------------------------------------------------------------
PAREN_RE = re.compile(r"^(.*?)(\s*\([^()]+\)\s*)$")


def strip_parenthetical(username: str, body: str) -> str:
    m = PAREN_RE.match(username)
    if not m:
        return username
    base = m.group(1).rstrip()
    pattern = rf"\[\[\s*(?:User(?: talk)?):\s*{re.escape(username)}\b"
    return username if re.search(pattern, body, flags=re.I) else base


def user_links(body: str) -> List[str]:
    links = []
    for wl in parser.parse(body).filter_wikilinks():
        title = str(wl.title).strip()
        ns, _, rest = title.partition(":")
        if ns.lower() in {"user", "user talk"} and rest:
            links.append(rest.split("/")[0].strip())
    return links


def fuzzy_username(header_name: str, body: str) -> str:
    simple = strip_parenthetical(header_name, body)
    candidates = user_links(body)
    if not candidates:
        return simple
    for cand in candidates:
        if cand.lower() == simple.lower():
            return cand
    best, score = simple, 0.0
    for cand in candidates:
        r = SequenceMatcher(None, simple.lower(), cand.lower()).ratio()
        if r > score:
            best, score = cand, r
    return best if score >= 0.6 else simple

# -------------------------------------------------------------------------
#  Extraction + table building
# -------------------------------------------------------------------------
Statement = Tuple[str, str, int, int]  # username, anchor, words, limit


def slugify(text: str) -> str:
    return re.sub(r"\s+", "_", parser.parse(text).strip_code()).strip("_")


def extract_statements(site: mwclient.Site, sec: parser.wikicode.Wikicode) -> List[Statement]:
    out: List[Statement] = []
    for sub in sec.get_sections(levels=[3]):
        head_node = sub.filter_headings()[0]
        head_raw = head_node.title.strip()
        head_plain = parser.parse(head_raw).strip_code().strip()
        if head_plain.lower() == PLACEHOLDER_HEADING:
            continue
        m = re.match(r"Statement by\s+(.+)", head_plain, re.I)
        if not m:
            continue
        raw_user = m.group(1).strip()
        body = sub[len(str(head_node)) :].strip()
        username = fuzzy_username(raw_user, body)
        anchor = slugify(head_raw)
        limit = int(AWL_RE.search(body).group(1)) if AWL_RE.search(body) else DEFAULT_LIMIT
        words = rendered_word_count(site, TEMPLATE_RE.sub("", body))
        out.append((username, anchor, words, limit))
    return out

HEADER = (
    "{| class=\"wikitable sortable\"\n"
    "! User !! Section !! Words !! Limit !! Over limit?\n"
)


def build_tables(site: mwclient.Site, src_text: str) -> str:
    blocks = []
    for lvl2 in parser.parse(src_text).get_sections(levels=[2]):
        sec_title = parser.parse(lvl2.filter_headings()[0].title).strip_code().strip()
        rows = []
        for username, anchor, words, limit in extract_statements(site, lvl2):
            over = words > limit * OVER_FACTOR
            style = f" style=\"background:{RED_HEX};\"" if over else ""
            link = f"[[{ARCA_PAGE}#{anchor}|link]]"
            rows.append(
                f"|-{style}\n| [[User:{username}|{username}]] || {link} || {words} || {limit} || {'yes' if over else 'no'}"
            )
        if rows:
            table = HEADER + "\n".join(rows) + "\n|}"
            blocks.append(f"=== {sec_title} ===\n{table}")
    return "\n\n".join(blocks)

# -------------------------------------------------------------------------
#  Main loop
# -------------------------------------------------------------------------

def run_once(site: mwclient.Site):
    src = site.pages[ARCA_PAGE]
    new_text = build_tables(site, src.text())
    dest = site.pages[TARGET_PAGE]
    if dest.text() != new_text:
        dest.save(new_text, summary=f"Update word‑count tables from [[{ARCA_PAGE}]] (bot)", minor=False, bot=False)
        logging.info("Updated %s", TARGET_PAGE)
    else:
        logging.info("No changes – up to date.")


def main(loop: bool = True):
    site = connect()
    # single-run mode?
    if not loop:
        try:
            run_once(site)
        except Exception:
            logging.exception("Exception during single run")
        return

    # continuous mode
    while True:
        try:
            run_once(site)
        except Exception:
            logging.exception("Exception in main loop; retrying.")
        time.sleep(RUN_INTERVAL)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ARCA word-count bot; runs forever by default."
    )
    p.add_argument(
        "-1", "--once",
        action="store_true",
        help="run exactly one update and exit"
    )
    args = p.parse_args()
    main(loop=not args.once)
