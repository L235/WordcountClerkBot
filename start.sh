#!/bin/sh

echo "=== Starting container ===" >&2
echo "Current directory: $(pwd)" >&2
echo "Directory contents:" >&2
ls -la >&2

echo "=== Environment variables ===" >&2
env | sort >&2

# Validate required environment variables
if [ -z "$BOT_USER" ]; then
    echo "ERROR: BOT_USER environment variable is not set" >&2
    exit 1
fi

if [ -z "$BOT_PASSWORD" ]; then
    echo "ERROR: BOT_PASSWORD environment variable is not set" >&2
    exit 1
fi

echo "=== Required variables are set ===" >&2
echo "BOT_USER is set to: ${BOT_USER}" >&2
echo "BOT_PASSWORD is set (length: ${#BOT_PASSWORD})" >&2

# Generate settings.json from environment variables
echo "=== Generating settings.json ===" >&2
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

echo "=== Settings file contents ===" >&2
cat settings.json >&2
echo "=== End settings file ===" >&2

echo "=== Starting bot ===" >&2
exec python arca_wordcount_bot.py "$@" 