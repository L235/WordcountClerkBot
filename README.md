# KevinClerkBot – ARCA Word‑Count Monitor

This bot watches the page
[`Wikipedia:Arbitration/Requests/Clarification and Amendment`](https://en.wikipedia.org/wiki/Wikipedia:Arbitration/Requests/Clarification_and_Amendment)
and maintains per‑request word‑count tables at
`User:KevinClerkBot/ARCA word counts`.

## Features

* Counts **rendered** words for each “Statement by …” subsection.
* Honors custom limits specified with `{{ApprovedWordLimit|words=n}}`.
* Ignores boiler‑plate “Statement by {other‑editor}” placeholders.
* Handles parenthetical descriptors and fuzzy‑matches shortened headings
  (e.g. *Vanamonde* → *Vanamonde93*).
* Flags and highlights submissions more than **10 % over** the limit.

## Requirements

* Python 3.9+
* `mwclient`, `mwparserfromhell`, `requests`

```bash
pip install mwclient mwparserfromhell requests
```

## Setup

1. **Clone/Copy** the script `arca_wordcount_bot.py`.
2. **Create `settings.json`** (see sample in the script header) and fill
   in:

   * BotPassword creds (`user`, `bot_password`).
   * `cookie_path` – where session cookies should be saved.
   * Any overrides (e.g. a different target page or run interval).
3. **Run once** to confirm it logs in and edits the target sandbox page
   correctly.

```bash
python arca_wordcount_bot.py
```

If everything looks good the bot will sleep for the configured
`run_interval` (default 10 minutes) and repeat.

## Constants 

Most tunable constants are now in `settings.json`; edit that instead of
changing code. If you *do* modify the script, restart the process.
