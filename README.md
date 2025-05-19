# ARCA Word Count Bot

A MediaWiki bot that counts rendered words in each "Statement by ..." on
[[Wikipedia:Arbitration/Requests/Clarification and Amendment]] (ARCA) and
publishes one wikitable per request to the bot's userspace.

## Configuration

The bot can be configured in two ways:

### 1. Local Development

Create a `settings.json` file with your configuration:

```json
{
  "site": "en.wikipedia.org",
  "path": "/w/",
  "user": "BotUser@PasswordName",
  "bot_password": "s3cret",
  "ua": "KevinClerkBot/0.5 (https://github.com/L235/WordcountClerkBot)",
  "cookie_path": "cookies.txt",
  "arca_page": "Wikipedia:Arbitration/Requests/Clarification and Amendment",
  "target_page": "User:KevinClerkBot/ARCA word counts",
  "default_limit": 500,
  "run_interval": 600,
  "over_factor": 1.10,
  "red_hex": "#ffcccc",
  "placeholder_heading": "statement by {other-editor}"
}
```

### 2. Environment Variables

When running in a container (e.g., Railway.app), configure via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SITE` | MediaWiki site | en.wikipedia.org |
| `API_PATH` | API path | /w/ |
| `BOT_USER` | Bot username | (required) |
| `BOT_PASSWORD` | Bot password | (required) |
| `USER_AGENT` | Bot user agent | KevinClerkBot/0.5 |
| `COOKIE_PATH` | Cookie storage path | /app/cookies/cookies.txt |
| `ARCA_PAGE` | Page to monitor | Wikipedia:Arbitration/Requests/Clarification and Amendment |
| `TARGET_PAGE` | Output page | User:KevinClerkBot/ARCA word counts |
| `DEFAULT_LIMIT` | Default word limit | 500 |
| `RUN_INTERVAL` | Update interval (seconds) | 600 |
| `OVER_FACTOR` | Over-limit factor | 1.10 |
| `RED_HEX` | Highlight color | #ffcccc |
| `PLACEHOLDER_HEADING` | Statement heading template | statement by {other-editor} |

## Running the Bot

### Local Development

```bash
# Run continuously
python arca_wordcount_bot.py

# Run once and exit
python arca_wordcount_bot.py -1
```

### Docker

```bash
# Build the image
docker build -t wordcount-bot .

# Run once
docker run -e BOT_USER=YourBot -e BOT_PASSWORD=secret wordcount-bot

# Run continuously
docker run -e BOT_USER=YourBot -e BOT_PASSWORD=secret wordcount-bot --
```

### Railway.app

1. Create a new project
2. Connect your GitHub repository
3. Set required environment variables:
   - `BOT_USER`
   - `BOT_PASSWORD`
4. (Optional) Set other environment variables to customize behavior
5. Deploy

The bot will run every 10 minutes via Railway's cron service.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (if any)
python -m pytest
```

## License

MIT License
