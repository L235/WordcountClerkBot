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

# Create cookie directory if needed
SESSION_FILE="${SESSION_FILE:-/app/cookies/cookies.txt}"
mkdir -p "$(dirname "$SESSION_FILE")"

# Python script now handles environment variables directly
python arca_wordcount_bot.py "$@" 