#!/usr/bin/env bash
# Usage: ./run-bot.sh [bot.py args...]
# Requires env vars: BOT_USER="text1@text2", BOT_PASSWORD, PYWIKIBOT_DIR.

set -euo pipefail

# Ensure required env vars are present
: "${BOT_USER:?Environment variable BOT_USER must be set (format text1@text2).}"
: "${BOT_PASSWORD:?Environment variable BOT_PASSWORD must be set.}"
: "${PYWIKIBOT_DIR:?Environment variable PYWIKIBOT_DIR must be set.}"

# Parse BOT_USER into USERNAME and SUFFIX
USERNAME="${BOT_USER%@*}"
SUFFIX="${BOT_USER#*@}"

# Create the Pywikibot configuration directory if needed
mkdir -p "$PYWIKIBOT_DIR"

# Write user-config.py
cat >"$PYWIKIBOT_DIR/user-config.py" <<EOF
family = 'wikipedia'
mylang = 'en'
usernames['wikipedia']['en'] = '$USERNAME'
password_file = "user-password.py"
EOF

# Write user-password.py
cat >"$PYWIKIBOT_DIR/user-password.py" <<EOF
('$USERNAME', BotPassword('$SUFFIX', '$BOT_PASSWORD'))
EOF

# Run the bot, forwarding any arguments passed to this script
exec python bot.py "$@"
