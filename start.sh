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

# Set default cookie-jar location
SESSION_FILE="${SESSION_FILE:-/app/cookies/cookies.txt}"

# Create required directory
mkdir -p "$(dirname "$SESSION_FILE")"

# Generate settings.json from environment variables
cat > settings.json << EOF
{
  "site": "${SITE:-en.wikipedia.org}",
  "path": "${API_PATH:-/w/}",
  "user": "${BOT_USER}",
  "bot_password": "${BOT_PASSWORD}",
  "ua": "${USER_AGENT:-KevinClerkBot/0.5 (https://github.com/L235/WordcountClerkBot)}",
  "arca_page": "${ARCA_PAGE:-Wikipedia:Arbitration/Requests/Clarification and Amendment}",
  "target_page": "${TARGET_PAGE:-User:KevinClerkBot/ArbCom word counts}",
  "data_page": "${DATA_PAGE:-User:KevinClerkBot/ArbCom word counts/data}",
  "default_limit": ${DEFAULT_LIMIT:-500},
  "run_interval": ${RUN_INTERVAL:-600},
  "over_factor": ${OVER_FACTOR:-1.10},
  "red_hex": "${RED_HEX:-#ffcccc}",
  "placeholder_heading": "${PLACEHOLDER_HEADING:-statement by {other-editor}}",
  "header_text": "${HEADER_TEXT:-}",
  "session_file": "${SESSION_FILE}"
}
EOF

# Run the bot with all arguments
python arca_wordcount_bot.py "$@" 