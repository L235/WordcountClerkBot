# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WordcountClerkBot is a MediaWiki bot that monitors Wikipedia arbitration request pages (ARCA, AE, ARC), counts words in statements, and publishes color-coded reports. It's designed to help enforce word limits on arbitration statements.

## Key Components

- **arca_wordcount_bot.py**: Main bot script containing all logic
- **settings.json**: Configuration file (created from settings.json.example)
- **start.sh**: Entry script that generates settings.json from environment variables
- **Dockerfile**: Container configuration for deployment
- **railway.json**: Railway.app deployment configuration with cron schedule

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp settings.json.example settings.json
# Edit settings.json with your bot credentials

# Run once (single execution)
./start.sh -1

# Run continuously (loop mode)
./start.sh

# Debug mode
./start.sh --debug
```

### Docker Development
```bash
# Build container
docker build -t wordcount-bot .

# Run once with environment variables
docker run --rm -e BOT_USER=YourBot -e BOT_PASSWORD=secret wordcount-bot

# Run continuously
docker run --rm -e BOT_USER=YourBot -e BOT_PASSWORD=secret wordcount-bot --
```

## Architecture

### Core Parser Classes
- **BaseParser**: Abstract base for all board parsers
- **SimpleBoardParser**: Handles ARCA/ARC pages with standard "Statement by" sections
- **AEParser**: Handles AE pages with more complex request structures and "Request concerning" blocks

### Data Flow
1. **Configuration**: Loads from settings.json or environment variables via start.sh
2. **Authentication**: Uses MozillaCookieJar for persistent login sessions
3. **Parsing**: Each board type has specialized parser for extracting statements
4. **Word Counting**: Two methods - visible (frontend JS replica) and expanded (full HTML)
5. **Report Generation**: Creates wikitable with color-coded status indicators
6. **Output**: Updates both main report page and data template page

### Word Counting Logic
- **Visible count**: Replicates frontend wordcount.js, excluding hidden/collapsed/struck content
- **Expanded count**: Counts all words in rendered HTML
- Uses MediaWiki parse API with caching for efficiency

### Status Classification
- **OK**: Within word limit
- **WITHIN**: Over limit but within 10% threshold (amber highlight)
- **OVER**: Exceeds 10% threshold (red highlight)

## Configuration Sources
1. **Defaults**: Hardcoded in DEFAULT_CFG dictionary (lowest priority)
2. **settings.json**: JSON configuration file (overrides defaults)
3. **Environment variables**: Direct override (highest priority - overrides both defaults and settings.json)

## Deployment Modes
- **Local**: Direct Python execution with settings.json
- **Docker**: Container with environment variable configuration
- **Railway.app**: Automated cron deployment (every 5 minutes per railway.json)