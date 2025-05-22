#!/bin/sh

# Validate required environment variables
if [ -z "$BOT_USER" ]; then
    echo "ERROR: BOT_USER environment variable is not set" >&2
    exit 1
fi

if [ -z "$BOT_PASSWORD" ]; then
    echo "ERROR: BOT_PASSWORD environment variable is not set" >&2
    exit 1
fi

# Set default paths
COOKIE_PATH="${COOKIE_PATH:-/app/cookies/cookies.txt}"
SESSION_FILE="${SESSION_FILE:-/data/cookies/session.pkl}"

# Create required directories
mkdir -p "$(dirname "$COOKIE_PATH")"
mkdir -p "$(dirname "$SESSION_FILE")"

# Generate settings.json from environment variables
cat > settings.json << EOF
{
  "site": "${SITE:-en.wikipedia.org}",
  "path": "${API_PATH:-/w/}",
  "user": "${BOT_USER}",
  "bot_password": "${BOT_PASSWORD}",
  "ua": "${USER_AGENT:-KevinClerkBot/0.5 (https://github.com/L235/WordcountClerkBot)}",
  "cookie_path": "${COOKIE_PATH}",
  "arca_page": "${ARCA_PAGE:-Wikipedia:Arbitration/Requests/Clarification and Amendment}",
  "target_page": "${TARGET_PAGE:-User:KevinClerkBot/ArbCom word counts}",
  "data_page": "${DATA_PAGE:-User:KevinClerkBot/ArbCom word counts/data}",
  "default_limit": ${DEFAULT_LIMIT:-500},
  "run_interval": ${RUN_INTERVAL:-600},
  "over_factor": ${OVER_FACTOR:-1.10},
  "red_hex": "${RED_HEX:-#ffcccc}",
  "placeholder_heading": "${PLACEHOLDER_HEADING:-statement by {other-editor}}",
  "header_text": "${HEADER_TEXT:-}",
  "session_file": "${SESSION_FILE}",
  "session_max_age": ${SESSION_MAX_AGE:-86400}
}
EOF

# Run the bot
exec python arca_wordcount_bot.py "$@" 