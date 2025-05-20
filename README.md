# WordcountClerkBot

A MediaWiki bot that monitors word-limited statements on Wikipedia arbitration request pages (ARCA, AE, ARC), counts both visible and expanded words, compares them against approved limits, and publishes a color-coded report and data template. It can run continuously or in single-run mode, and is fully configurable via a JSON file or environment variables.

---

## Configuration

### 1. Via `settings.json`

Copy `settings.json.example` to `settings.json` and fill in your values:

```jsonc
{
  "site": "en.wikipedia.org",
  "path": "/w/",
  "user": "BotUser@PasswordName",
  "bot_password": "s3cret",
  "ua": "WordcountClerkBot/2.4 (https://github.com/YourRepo/WordcountClerkBot)",
  "cookie_path": "/app/cookies/cookies.txt",
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
  "placeholder_heading": "statement by {other-editor}",
  "header_text": ""
}
```

| Key                   | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `site`                | MediaWiki domain (e.g. `en.wikipedia.org`)                      |
| `path`                | API path (default `/w/`)                                        |
| `user`                | Bot username (`User@BotPassword` format)                        |
| `bot_password`        | Bot password                                                    |
| `ua`                  | User-agent string                                               |
| `cookie_path`         | File path to store cookies                                      |
| `arca_page`           | ARCA requests page title                                        |
| `ae_page`             | AE requests page title                                          |
| `arc_page`            | ARC requests page title                                         |
| `target_page`         | Output report page in your userspace                            |
| `data_page`           | Output data template page                                       |
| `default_limit`       | Default word limit (if no template present)                     |
| `over_factor`         | Multiplier for “within” threshold (e.g. `1.10` = 10% over)      |
| `run_interval`        | Seconds between automatic updates                               |
| `red_hex`             | Highlight color for **over**                                    |
| `amber_hex`           | Highlight color for **within 10%**                              |
| `placeholder_heading` | Template heading to skip (e.g. `"statement by {other-editor}"`) |
| `header_text`         | Optional wikitext inserted at top of report                     |

### 2. Via Environment Variables

When running in Docker or on Railway, you can omit `settings.json` and set:

| Variable              | Default                              | Description                         |
| --------------------- | ------------------------------------ | ----------------------------------- |
| `SITE`                | `en.wikipedia.org`                   | MediaWiki domain                    |
| `API_PATH`            | `/w/`                                | API path                            |
| `BOT_USER`            | **(required)**                       | Bot username                        |
| `BOT_PASSWORD`        | **(required)**                       | Bot password                        |
| `USER_AGENT`          | `WordcountClerkBot/2.4 (<repo URL>)` | User-agent string                   |
| `COOKIE_PATH`         | `/app/cookies/cookies.txt`           | Path for cookies                    |
| `ARCA_PAGE`           | *same as `arca_page` above*          | ARCA requests page title            |
| `AE_PAGE`             | *same as `ae_page` above*            | AE requests page title              |
| `ARC_PAGE`            | *same as `arc_page` above*           | ARC requests page title             |
| `TARGET_PAGE`         | *same as `target_page` above*        | Report page                         |
| `DATA_PAGE`           | *same as `data_page` above*          | Data template page                  |
| `DEFAULT_LIMIT`       | `500`                                | Default word limit                  |
| `OVER_FACTOR`         | `1.10`                               | “within” multiplier                 |
| `RUN_INTERVAL`        | `600`                                | Seconds between runs                |
| `RED_HEX`             | `#ffcccc`                            | Over-limit highlight color          |
| `AMBER_HEX`           | `#ffffcc`                            | Within-limit highlight color        |
| `PLACEHOLDER_HEADING` | `statement by {other-editor}`        | Heading template to ignore          |
| `HEADER_TEXT`         | `""`                                 | Prepend this wikitext at report top |

---

## Quick Start

### Local (direct)

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare your config
cp settings.json.example settings.json
# Edit settings.json...

# Run continuously
./start.sh && python arca_wordcount_bot.py

# Run once and exit
./start.sh && python arca_wordcount_bot.py --once
```

> **Note:** `start.sh` will generate `settings.json` if you supply the required `BOT_USER` and `BOT_PASSWORD` env vars.

### Docker

```bash
# Build
docker build -t wordcount-bot .

# Run once
docker run --rm \
  -e BOT_USER=YourBot \
  -e BOT_PASSWORD=secret \
  wordcount-bot

# Run continuously
docker run --rm \
  -e BOT_USER=YourBot \
  -e BOT_PASSWORD=secret \
  wordcount-bot --  # pass no args for looping
```

### Railway.app

1. Create a new Railway project and connect your repo.
2. Railway will detect `Dockerfile` and use `railway.json` (cron: \*/15 \* \* \* \*).
3. Set **Environment Variables**:

   * `BOT_USER`, `BOT_PASSWORD` (required)
   * (Optional) customize via the vars listed above
4. Deploy – the bot will run every 15 minutes as per `railway.json`.

---

## Development

* Code is written for **Python 3.11**.
* Dependencies managed in `requirements.txt`.

---

## License

This project is licensed under the **MIT License**.
