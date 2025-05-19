#!/bin/sh

set -e  # Exit on any error
set -x  # Print commands and their arguments as they are executed

# Validate required environment variables
if [ -z "$BOT_USER" ]; then
    echo "Error: BOT_USER environment variable is not set" >&2
    exit 1
fi

if [ -z "$BOT_PASSWORD" ]; then
    echo "Error: BOT_PASSWORD environment variable is not set" >&2
    exit 1
fi

# Generate settings.json from environment variables
cat > settings.json << EOF
{
  "site": "${SITE:-en.wikipedia.org}",
  "path": "${API_PATH:-/w/}",
  "user": "${BOT_USER}",
  "bot_password": "${BOT_PASSWORD}",
  "ua": "${USER_AGENT:-KevinClerkBot/0.5 (https://github.com/L235/WordcountClerkBot)}",
  "cookie_path": "${COOKIE_PATH:-/app/cookies/cookies.txt}",
  "arca_page": "${ARCA_PAGE:-Wikipedia:Arbitration/Requests/Clarification and Amendment}",
  "target_page": "${TARGET_PAGE:-User:KevinClerkBot/ARCA word counts}",
  "default_limit": ${DEFAULT_LIMIT:-500},
  "run_interval": ${RUN_INTERVAL:-600},
  "over_factor": ${OVER_FACTOR:-1.10},
  "red_hex": "${RED_HEX:-#ffcccc}",
  "placeholder_heading": "${PLACEHOLDER_HEADING:-statement by {other-editor}}"
}
EOF

# Debug output
echo "=== Environment Variables ===" >&2
echo "BOT_USER: ${BOT_USER}" >&2
echo "BOT_PASSWORD: ${BOT_PASSWORD:+set}${BOT_PASSWORD:-not set}" >&2
echo "SITE: ${SITE:-en.wikipedia.org}" >&2
echo "API_PATH: ${API_PATH:-/w/}" >&2
echo "=== Settings File Contents ===" >&2
cat settings.json >&2
echo "=== End Settings File ===" >&2

# Run the bot
exec python arca_wordcount_bot.py "$@" 