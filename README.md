# Applicazione Fantacalcio

Streamlit assistant for a 10-manager Serie A fantasy-football auction.
The app ranks players by role, groups them into fixed clusters of 10, and
shows the rule-of-thumb maximum bid for each cluster.

## Quick Start

Install the dependencies with Python 3.13:

```bash
python3.13 -m pip install -r requirements.txt
```

Start the application (web UI — HTML/CSS/JavaScript client, Python HTTP backend):

```bash
python3.13 web_app.py
```

Open <http://127.0.0.1:7860/>. Set `PORT` to use a different port. The
browser client talks to the JSON API in `web_app.py`; the data pipeline,
auction engine and scrapers are shared with the Streamlit version.

A legacy Streamlit UI is still available for reference:

```bash
streamlit run app_asta.py
```

See `AVVIO_APP.md` for background startup, logging, and troubleshooting.

## Application Tabs

- `Setup`: refresh data, configure the auction budget and fair-value table,
  and choose ranking weights/method.
- `Asta live`: personal on-the-clock assistant with a dynamic bid recommendation,
  an inviolable cluster cap, next-best alternatives, watchlist, and saved
  budget/roster tracking. Other participants are recorded only to keep the
  available-player pool and market context current.
- `Giocatori`: browse players by role and cluster, see the computed score, and
  mark unavailable players. Players recorded in the live auction disappear
  automatically from this list.
- `Formazioni`: one-click refresh of probable starters, set pieces,
  ballottaggi, injuries, suspensions, doubts, and team-level warnings.

## Ranking Model

The ranking is computed separately for each role using available historical
fantamedia, Gazzetta FVM, FCP ALG, injury resistance, probable starting
status, set pieces, and player attributes. If advanced historical statistics
are imported, it also uses minutes, starts and xG+xA per 90. The current
implementation supports manual, predictive, and blended ranking methods.

`Affidabilità d'impiego` combines probable lineups with historical usage when
available. `Valore stagione`, `Upside` and `Confidenza dati` are displayed as
separate, transparent indicators: a player is never assigned invented minutes
or xG when no source has supplied them.

The rank is then divided into fixed groups of 10 players. This rule is part of
the auction strategy: with 10 participants, each cluster represents one
player per participant. Fair values remain a separate rule-of-thumb price
table, indexed by role and cluster.

## Personal auction advice

The live assistant never raises a player above the fair value of their fixed
cluster. Its lower, personal recommendation adapts to the player's position
inside that cluster, open slots and chosen role priority in your squad, and
the number of comparable available alternatives. It also reserves one credit
for every other unfilled roster slot, so the personal maximum bid is always
affordable through the end of the auction. The verdict is deliberately simple:
bid, bid only if it is a priority, or walk away.

The model is intended to estimate relative player quality. It does not yet
learn auction prices because historical auction results are not available.
Record final prices during future auctions to calibrate the fair-value tables.

## Refreshing Data

The checked-in CSV files under `data/` are offline fixtures used by the app and
tests. Refresh them from the configured external sources with:

```bash
python3.13 scraper.py --quotes
python3.13 scraper.py --players
python3.13 scraper.py --lineups --setpieces
python3.13 scraper.py --panchinari
```

Use `python3.13 scraper.py --all` to run every scraper. Scraping requires a
network connection and overwrites the corresponding CSV files. The panchinari
source (sosfanta) is a seasonal "formazione-tipo" article: when the new season
is published, update `PANCHINARI_URL` in `scraper.py` and re-run
`--panchinari`. The `Formazioni` tab then shows, under every probable starter,
the most likely bench player for the same spot.

### Advanced historical statistics

For the best ranking, export the Serie A player-statistics table from FBref or
FotMob shortly before the auction and upload the CSV in
`Setup → Statistiche storiche avanzate`. The importer recognizes common English
and Italian headings for player, team, appearances, starts, minutes, xG, xA,
goals, assists, cards, injury days and matches missed, then saves a normalized local file at
`data/advanced_stats.csv`.

This deliberate CSV workflow is more reliable than an undocumented live API on
auction day. You can also run it from the terminal:

```bash
python3.13 scraper.py --advanced /path/to/export.csv
```

## Tests

Run the smoke test with the interpreter that has Streamlit installed:

```bash
python3.13 tests/smoke_test.py
```

The test checks the data pipeline, normalized features, FM estimation,
ranking methods, fixed-size clusters, model diagnostics, lineup matching,
runtime state round-trips, and Streamlit startup.

## Project Structure

```text
web_app.py        Python HTTP backend (static files + JSON API)
web/              HTML/CSS/JavaScript frontend (index.html, style.css, app.js, api-client.js)
app_asta.py       Legacy Streamlit UI
data_loader.py    Data loading, feature engineering, ranking, and clusters
asta_core.py      Auction sessions and cluster fair-value lookup
scraper.py        External data download and parsing
data/             CSV fixtures and runtime auction state
tests/            Offline smoke test
```

The web UI offers the same four areas as the Streamlit version — Setup,
Asta live, Giocatori, Formazioni — with a dynamic client: live search with
suggestions, personal bid advice that refreshes as the price changes, a
background data refresh with progress, dark/light themes, and toasts for
every saved action.

Runtime sessions and ranking preferences are local state. Do not commit
`data/sessions/` or `data/ranking_weights.json`.
