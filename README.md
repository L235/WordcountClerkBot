# WordcountClerkBot

A MediaWiki bot that monitors word-limited statements on Wikipedia arbitration request pages (ARCA, AE, ARC), counts both visible and expanded words, compares them against approved limits, and publishes a color-coded report and data template. It can run continuously or in single-run mode, and is fully configurable via environment variables.

---

## Configuration

The bot is configured via environment variables. All configuration keys use ALL_CAPS naming to match environment variables directly.

### Environment Variables for run.sh

These variables are **required** when using `run.sh`:

| Variable         | Description                                      |
| ---------------- | ------------------------------------------------ |
| `BOT_USER`       | **(required)** Bot username (format: `User@BotPasswordName`) |
| `BOT_PASSWORD`   | **(required)** Bot password                      |
| `PYWIKIBOT_DIR`  | **(required)** Directory for pywikibot config files (e.g., `.`) |

### Optional Bot Configuration Variables

| Variable               | Default                                                      | Description                         |
| ---------------------- | ------------------------------------------------------------ | ----------------------------------- |
| `SITE`                 | `en.wikipedia.org`                                           | MediaWiki domain                    |
| `API_PATH`             | `/w/`                                                        | API path                            |
| `USER_AGENT`           | `WordcountClerkBot/2.4 (<repo URL>)`                         | User-agent string                   |
| `STATE_DIR`            | `.` (current directory)                                      | Directory for state files (cookies) |
| `ARCA_PAGE`            | `Wikipedia:Arbitration/Requests/Clarification and Amendment` | ARCA requests page title            |
| `AE_PAGE`              | `Wikipedia:Arbitration/Requests/Enforcement`                 | AE requests page title              |
| `ARC_PAGE`             | `Wikipedia:Arbitration/Requests/Case`                        | ARC requests page title             |
| `TARGET_PAGE`          | `User:WordcountClerkBot/word counts`                         | Report page                         |
| `DATA_PAGE`            | `User:WordcountClerkBot/word counts/data`                    | Data template page                  |
| `EXTENDED_PAGE`        | `User:WordcountClerkBot/word counts/extended`                | Extended report page                |
| `DEFAULT_LIMIT`        | `500`                                                        | Default word limit                  |
| `EVIDENCE_LIMIT_NAMED` | `1000`                                                       | Word limit for named parties        |
| `EVIDENCE_LIMIT_OTHER` | `500`                                                        | Word limit for other users          |
| `OVER_FACTOR`          | `1.10`                                                       | "within" multiplier                 |
| `RUN_INTERVAL`         | `600`                                                        | Seconds between runs                |
| `RENDER_REFRESH_INTERVAL` | `3600`                                                    | Maximum interval between full rendered-count refreshes |
| `RED_HEX`              | `#ffcccc`                                                    | Over-limit highlight color          |
| `AMBER_HEX`            | `#ffffcc`                                                    | Within-limit highlight color        |
| `PLACEHOLDER_HEADING`  | `statement by {other-editor}`                                | Heading template to ignore          |
| `HEADER_TEXT`          | `""`                                                         | Prepend this wikitext at report top |

---

## Quick Start

### Using run.sh (Recommended)

The provided `run.sh` script handles Pywikibot configuration automatically using environment variables.

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export BOT_USER="YourBot@PasswordName"
export BOT_PASSWORD="your_password"
export PYWIKIBOT_DIR="."

# Run continuously
./run.sh

# Run once and exit
./run.sh --once
```

### Direct Execution

If you have a `user-config.py` set up manually:

```bash
python bot.py
```

---

## Development

When an author has multiple statement sections in the same request, the report
and status-template data use their combined word count. The highest recognized
word-limit allowance applies to that total; allowances are not added together.
The report retains links to each contributing section. Different requests and
usernames that differ in capitalization after the first character stay separate.

The bot batches revision checks and records the exact source revisions processed
in `STATE_DIR/publication_checkpoint.json` only after all three outputs succeed.
No-op source edits advance this checkpoint too. Failed reads, rendering, or saves
leave the pass pending for retry; case party pages and output-page changes are
included in the checks. Source content is fetched by revision ID so edits made
during a pass are picked up by the next poll.

Code/configuration changes trigger reconciliation. Rendered counts and their HTML
cache are refreshed together every `RENDER_REFRESH_INTERVAL` seconds (one hour by
default), even without source edits, to pick up changed transclusions. Source
edits do not postpone this refresh. Keep `STATE_DIR` on persistent storage and
run one publisher per state directory.

- Code is written for **Python 3.11**.
- Dependencies managed in `requirements.txt`.

---

## License

This project is licensed under the **MIT License**.
