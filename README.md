# WordcountClerkBot

A MediaWiki bot that monitors word-limited statements on Wikipedia arbitration request pages (ARCA, AE, ARC), counts both visible and expanded words, compares them against approved limits, and publishes a color-coded report and data template. It can run continuously or in single-run mode, and is fully configurable via environment variables.

---

## Configuration

The bot is configured via environment variables. All configuration keys use ALL_CAPS naming to match environment variables directly.

### Environment Variables

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
| `OVER_FACTOR`          | `1.10`                                                       | “within” multiplier                 |
| `RUN_INTERVAL`         | `600`                                                        | Seconds between runs                |
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

- Code is written for **Python 3.11**.
- Dependencies managed in `requirements.txt`.

---

## License

This project is licensed under the **MIT License**.
