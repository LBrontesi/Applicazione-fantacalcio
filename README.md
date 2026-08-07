# Applicazione Fantacalcio

Streamlit assistant for a 10-manager Serie A fantasy-football auction.
The app ranks players by role, groups them into fixed clusters of 10, and
shows the rule-of-thumb maximum bid for each cluster.

## Quick Start

Install the dependencies with Python 3.13:

```bash
python3.13 -m pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app_asta.py
```

See `AVVIO_APP.md` for background startup, logging, and troubleshooting.

## Application Tabs

- `Setup`: refresh data, configure the auction budget and fair-value table,
  and choose ranking weights/method.
- `Giocatori`: browse players by role and cluster, see the computed score, and
  mark players already bought.
- `Formazioni`: inspect probable starters, substitutes, and set-piece duties.

## Ranking Model

The ranking is computed separately for each role using available historical
fantamedia, Gazzetta FVM, FCP ALG, injury resistance, probable starting
status, set pieces, and player attributes. The current implementation supports
manual, predictive, and blended ranking methods.

The rank is then divided into fixed groups of 10 players. This rule is part of
the auction strategy: with 10 participants, each cluster represents one
player per participant. Fair values remain a separate rule-of-thumb price
table, indexed by role and cluster.

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
```

Use `python3.13 scraper.py --all` to run every scraper. Scraping requires a
network connection and overwrites the corresponding CSV files.

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
app_asta.py       Streamlit UI
data_loader.py    Data loading, feature engineering, ranking, and clusters
asta_core.py      Auction sessions and cluster fair-value lookup
scraper.py        External data download and parsing
data/             CSV fixtures and runtime auction state
tests/             Offline smoke test
```

Runtime sessions and ranking preferences are local state. Do not commit
`data/sessions/` or `data/ranking_weights.json`.
