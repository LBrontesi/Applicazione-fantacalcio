"""Asta Coach — web UI backend.

Small Python HTTP server that replaces the Streamlit frontend with a plain
HTML/CSS/JavaScript client (see ``web/``). It serves the static assets and a
JSON API that reuses the same data pipeline, auction engine and scrapers as
``app_asta.py``.

Run with:

    python3 web_app.py

then open http://127.0.0.1:7860. Set ``PORT`` to use a different port.
"""

import json
import logging
import math
import mimetypes
import os
import tempfile
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

import asta_core
import data_loader
import scraper
from data_loader import (
    ROLE_ORDER, build_lineups, build_players, fair_values_scaled,
    load_ranking_weights, save_ranking_weights, backtest_predictor,
    DEFAULT_METHOD,
)

PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"
PORT = int(os.getenv("PORT", "7860"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(10 * 1024 * 1024)))

WEIGHT_LABELS = {
    "FM": "Fantamedia (FCP)",
    "FVM": "Fantamedia (Gazzetta)",
    "ALG": "Algoritmo (FCP)",
    "Starter": "Probabile titolare",
    "SetPieces": "Set-piece (rigorista/angoli/punizioni)",
    "Tags": "Attributi (Fuoriclasse, Goleador, …)",
    "Injury": "Robustezza infortuni (premia chi non si infortuna)",
    "Availability": "Affidabilità d'impiego (formazione + robustezza)",
    "ExpectedOutput": "xG + xA per 90 (CSV storico)",
    "GolSubiti": "Gol subiti/90 (portieri, CSV storico)",
}

METHOD_LABELS = {
    "blend": "Blend modello + pesi (consigliato)",
    "model": "Modello predittivo (FM attesa)",
    "manual": "Pesi manuali",
}

ROLE_PLAN_CHOICES = {"Risparmia": 0.8, "Equilibrio": 1.0, "Spingi": 1.2}

WATCHLIST_EXPLANATIONS = {
    "A": "Obiettivo: puoi spingere fino al prezzo consigliato.",
    "B": "Alternativa: segui il prezzo consigliato senza inseguire.",
    "C": "Occasione: punta solo se il prezzo resta conveniente.",
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("asta_coach.web")

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_CACHE = {"players": None, "lineups": None, "rho": None}

ACTIVE_SESSION = None
SESSION_LOCK = threading.RLock()

SCRAPE_JOB = {
    "running": False,
    "label": "",
    "progress": 0.0,
    "done": False,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
SCRAPE_LOCK = threading.RLock()


def get_players(force=False):
    with _LOCK:
        if force or _CACHE["players"] is None:
            LOGGER.info("Building players table (force=%s)", force)
            _CACHE["players"] = build_players()
        return _CACHE["players"]


def get_lineups(force=False):
    with _LOCK:
        if force or _CACHE["lineups"] is None:
            LOGGER.info("Building lineups table (force=%s)", force)
            _CACHE["lineups"] = build_lineups(get_players())
        return _CACHE["lineups"]


def invalidate_caches():
    with _LOCK:
        _CACHE["players"] = None
        _CACHE["lineups"] = None
        _CACHE["rho"] = None


def get_rho(players):
    with _LOCK:
        if _CACHE["rho"] is None:
            _CACHE["rho"] = backtest_predictor(players)
        return _CACHE["rho"]


def active_session():
    """Return the active auction session, auto-loading the most recent one."""
    global ACTIVE_SESSION
    with SESSION_LOCK:
        if ACTIVE_SESSION is None:
            sessions = asta_core.list_sessions()
            if sessions:
                ACTIVE_SESSION = asta_core.load_session(str(sessions[0]))
    return ACTIVE_SESSION


def set_active_session(session):
    global ACTIVE_SESSION
    with SESSION_LOCK:
        ACTIVE_SESSION = session


def load_excluded():
    path = scraper.DATA_DIR / "esclusi.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["Nome"].astype(str).tolist())


def save_excluded(names):
    path = scraper.DATA_DIR / "esclusi.csv"
    scraper.DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"Nome": sorted(names)}).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _clean(value):
    """Make a value JSON-safe: numpy scalars to Python, NaN/NaT to None."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, (float, int)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    return str(value)


def records(df):
    """Convert a DataFrame to JSON-safe list-of-dicts (NaN -> None)."""
    return _clean(df.to_dict("records"))


def session_info(session):
    meta = session["meta"]
    return {
        "name": Path(session.get("_path", "")).stem
        if session.get("_path") else None,
        "created": meta["created"],
        "budget": int(meta["budget"]),
        "slots": {role: int(meta["slots"].get(role, 0)) for role in ROLE_ORDER},
        "fair": {role: [int(v) for v in meta["fair"].get(role, [])] for role in ROLE_ORDER},
        "teams": list(meta["teams"]),
        "my_team": meta["my_team"],
        "role_priorities": {
            role: float(meta["role_priorities"].get(role, 1.0))
            for role in ROLE_ORDER
        },
        "watchlist": _clean(session.get("watchlist", [])),
        "purchases": _clean(session.get("purchases", [])),
    }


def own_summary(session):
    if not session:
        return None
    return _clean(asta_core.team_summary(session, session["meta"]["my_team"]))


def source_freshness():
    sources = [
        ("Quotazioni", scraper.QUOTAZIONI, 24),
        ("Giocatori/FCP", scraper.PLAYERS_FCP, 24 * 7),
        ("Formazioni", scraper.FORMAZIONI, 8),
        ("Tiratori", scraper.SET_PIECES, 24 * 7),
        ("Panchinari", scraper.PANCHINARI, 24 * 7),
        ("Statistiche stagioni", _latest_statistiche_path(), 24 * 7),
    ]
    now = datetime.now().timestamp()
    result = []
    for label, path, fresh_hours in sources:
        if not path.exists():
            result.append({"label": label, "value": "Manca", "state": "red"})
            continue
        age_hours = max(0, (now - path.stat().st_mtime) / 3600)
        if age_hours < 1:
            value = f"{max(1, round(age_hours * 60))} min fa"
        elif age_hours < 24:
            value = f"{age_hours:.0f} ore fa"
        else:
            value = f"{age_hours / 24:.0f} giorni fa"
        state = "green" if age_hours <= fresh_hours else "amber"
        result.append({"label": label, "value": value, "state": state})
    return result


def _latest_statistiche_path():
    files = sorted(scraper.DATA_DIR.glob("statistiche_*.csv"))
    return files[-1] if files else scraper.DATA_DIR / "statistiche_nessuna.csv"


def roster_alerts(session, players):
    """Roster risks for the active session, mirroring app_asta.py logic."""
    if not session:
        return []
    own = asta_core.team_summary(session, session["meta"]["my_team"])
    purchases = own.get("purchases", [])
    slots = session["meta"]["slots"]
    if not purchases:
        return []
    alerts = []
    clubs = pd.Series([
        str(item.get("club", "")).strip() for item in purchases
        if str(item.get("club", "")).strip()
    ]).value_counts()
    crowded = [f"{club} ({count})" for club, count in clubs.items() if count >= 3]
    if crowded:
        alerts.append("Troppi giocatori della stessa squadra: " + ", ".join(crowded) + ".")

    total_spent = max(1, int(own.get("spent", 0)))
    for role in ROLE_ORDER:
        role_spent = sum(int(item.get("price", 0)) for item in purchases
                         if item.get("role") == role)
        if len(purchases) >= 5 and role_spent / total_spent >= 0.45:
            alerts.append(f"{role} assorbe il {role_spent / total_spent:.0%} dei crediti spesi.")

    if len(purchases) >= 10:
        by_role = own.get("by_role", {})
        for role in ROLE_ORDER:
            planned = max(1, int(slots.get(role, 0)))
            bought = int(by_role.get(role, 0))
            if bought == 0 or (planned >= 4 and bought / planned < 0.25):
                alerts.append(f"Reparto {role} molto indietro: {bought}/{planned} giocatori.")

    names = {str(item.get("name", "")) for item in purchases}
    owned = players[players["Nome"].isin(names)]
    if len(purchases) >= 8 and not owned.empty:
        starters = int(owned["Starter"].fillna(0).sum())
        if starters < max(2, round(len(purchases) * 0.40)):
            alerts.append(f"Pochi titolari probabili in rosa: {starters}/{len(purchases)}.")
        attacking = owned[owned["Ruolo"].isin(["C", "A"])]
        if len(attacking) >= 3 and not (attacking["SetPieces"].fillna(0) >= 0.3).any():
            alerts.append("Nessun centrocampista/attaccante con piazzati rilevati.")
    return alerts


def closest_role_plan(value):
    return min(ROLE_PLAN_CHOICES, key=lambda label: abs(
        ROLE_PLAN_CHOICES[label] - float(value)
    ))


# ---------------------------------------------------------------------------
# Scraping in background
# ---------------------------------------------------------------------------


def _run_scrape_job():
    jobs = [
        ("1/6 Scaricando quotazioni Gazzetta...",
         lambda: scraper.scrape_quotazioni(progress_cb=None)),
        ("2/6 Scaricando lista giocatori FCP (può richiedere alcuni minuti)...",
         lambda: scraper.scrape_fantacalciopedia(progress_cb=None)),
        ("3/6 Aggiornando formazioni, ballottaggi e indisponibili...",
         lambda: scraper.scrape_lineups(progress_cb=None)),
        ("4/6 Aggiornando rigoristi e tiratori...",
         lambda: scraper.scrape_set_pieces(progress_cb=None)),
        ("5/6 Aggiornando probabili panchinari (sosfanta)...",
         lambda: scraper.scrape_panchinari(progress_cb=None)),
        ("6/6 Aggiornando statistiche stagionali (gol subiti portieri)...",
         lambda: scraper.scrape_statistiche(progress_cb=None)),
    ]
    total = float(len(jobs))
    try:
        for index, (label, job) in enumerate(jobs):
            with SCRAPE_LOCK:
                SCRAPE_JOB["label"] = label
                SCRAPE_JOB["progress"] = index / total
                SCRAPE_JOB["error"] = None
            job()
            with SCRAPE_LOCK:
                SCRAPE_JOB["progress"] = (index + 1) / total
        with SCRAPE_LOCK:
            SCRAPE_JOB["done"] = True
            SCRAPE_JOB["finished_at"] = datetime.now().isoformat()
    except scraper.ScrapeError as exc:
        with SCRAPE_LOCK:
            SCRAPE_JOB["error"] = str(exc)
            SCRAPE_JOB["done"] = True
            SCRAPE_JOB["finished_at"] = datetime.now().isoformat()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Scrape job failed")
        with SCRAPE_LOCK:
            SCRAPE_JOB["error"] = f"Errore inatteso: {exc}"
            SCRAPE_JOB["done"] = True
            SCRAPE_JOB["finished_at"] = datetime.now().isoformat()
    finally:
        invalidate_caches()
        with SCRAPE_LOCK:
            SCRAPE_JOB["running"] = False


def start_scrape():
    with SCRAPE_LOCK:
        if SCRAPE_JOB["running"]:
            return False
        SCRAPE_JOB.update({
            "running": True,
            "label": "Preparazione...",
            "progress": 0.0,
            "done": False,
            "error": None,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        })
    threading.Thread(target=_run_scrape_job, daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# Multipart upload
# ---------------------------------------------------------------------------


def parse_multipart(content_type, body):
    """Extract the first uploaded file as (filename, bytes) from a form body."""
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip('"')
    if not boundary:
        return None, None
    delimiter = f"--{boundary}".encode()
    parts = body.split(delimiter)
    for chunk in parts:
        if not chunk or chunk == b"--\r\n":
            continue
        header_end = chunk.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_raw = chunk[:header_end].decode("utf-8", "replace")
        payload = chunk[header_end + 4:]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        filename = None
        for line in headers_raw.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for token in line.split(";"):
                    token = token.strip()
                    if token.lower().startswith("filename="):
                        filename = token[len("filename="):].strip('"')
        if filename:
            return filename, payload
    return None, None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def build_state_payload():
    players = get_players()
    session = active_session()
    weights = load_ranking_weights()
    method = weights.get("_method", DEFAULT_METHOD)
    if method not in METHOD_LABELS:
        method = "blend"
    summary = {
        "players": int(len(players)),
        "with_quote": int(players["QA"].notna().sum()),
        "with_fm": int(players["FM"].notna().sum()),
        "starters": int(players["Starter"].sum()),
        "with_minutes": int(players["Minuti"].notna().sum()),
    }
    return {
        "session": session_info(session) if session else None,
        "weights": _clean(weights),
        "method": method,
        "method_labels": METHOD_LABELS,
        "weight_labels": WEIGHT_LABELS,
        "role_order": ROLE_ORDER,
        "role_plan_choices": ROLE_PLAN_CHOICES,
        "watchlist_explanations": WATCHLIST_EXPLANATIONS,
        "default_slots": _clean(data_loader.DEFAULT_SLOTS),
        "default_budget": 500,
        "excluded": sorted(load_excluded()),
        "freshness": source_freshness(),
        "summary": summary,
        "advanced_present": scraper.ADVANCED_STATS.exists(),
        "sessions": [
            {
                "name": path.stem,
                "label": path.stem.replace("config_", ""),
                "budget": _session_budget(path),
            }
            for path in asta_core.list_sessions()
        ],
        "rho": _clean(get_rho(players)),
        "own": own_summary(session),
        "alerts": roster_alerts(session, players),
        "market": _clean(asta_core.market_snapshot(session)) if session else None,
        "purchased": sorted(asta_core.purchased_names(session)) if session else [],
        "fair_base": _clean(data_loader.DEFAULT_FAIR),
        "fair_default": _clean(fair_values_scaled(500)),
        "active_name": Path(session.get("_path", "")).stem if session and session.get("_path") else None,
    }


def _session_budget(path):
    try:
        with open(path) as fh:
            data = json.load(fh)
        return int(data.get("meta", {}).get("budget", 0))
    except (OSError, ValueError, TypeError):
        return 0


def player_by_name(players, name):
    match = players[players["Nome"] == name]
    if match.empty:
        raise ValueError("Giocatore non trovato.")
    return match.iloc[0]


def advice_payload(players, session, name, current_price):
    unavailable = load_excluded() | asta_core.purchased_names(session)
    available = players[~players["Nome"].isin(unavailable)]
    available_records = available.to_dict("records")
    player = player_by_name(players, name)
    advice = asta_core.auction_advice(
        session, player, available_records, current_price=current_price
    )

    alternatives = available[
        (available["Ruolo"] == player["Ruolo"]) &
        (available["Nome"] != player["Nome"])
    ].copy()
    alternatives = alternatives[
        (alternatives["Cluster"] <= int(player["Cluster"]) + 1) &
        (alternatives["Rank"] > int(player["Rank"]))
    ]
    if alternatives.empty:
        alternatives = available[
            (available["Ruolo"] == player["Ruolo"]) &
            (available["Nome"] != player["Nome"])
        ].copy()
    alternatives = alternatives.sort_values(["Cluster", "Rank"]).head(3)

    comparison_rows = [_comparison_row("Chiamato ora", player, advice)]
    for _, alternative in alternatives.iterrows():
        alt_advice = asta_core.auction_advice(
            session, alternative, available_records
        )
        comparison_rows.append(
            _comparison_row("Alternativa", alternative, alt_advice)
        )

    return {
        "player": _clean(player.to_dict()),
        "advice": _clean(advice),
        "plan_label": closest_role_plan(advice["priority"]),
        "comparison": comparison_rows,
        "has_alternatives": not alternatives.empty,
        "unavailable_count": len(unavailable),
        "available_count": int(len(available)),
    }


def _comparison_row(label, player, advice):
    starter = "Sì" if float(player.get("Starter", 0) or 0) >= 0.5 else "No / dubbio"
    return {
        "scelta": label,
        "nome": player["Nome"],
        "squadra": player["Squadra"],
        "cluster": int(player.get("Cluster", 0) or 0),
        "season_value": float(player.get("SeasonValue", 0) or 0),
        "starter": starter,
        "recommended": advice["recommended"],
        "fixed_cap": advice["fixed_cap"],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send_security_headers(self, cache_control):
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers("no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        from urllib.parse import unquote
        relative = unquote(path).lstrip("/")
        if relative == "":
            relative = "index.html"
        file_path = (WEB_DIR / relative).resolve()
        try:
            file_path.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_json(404, {"ok": False, "error": f"Not found: {path}"})
            return
        if not file_path.is_file():
            self._send_json(404, {"ok": False, "error": f"Not found: {path}"})
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        cache_control = (
            "no-cache"
            if file_path.suffix.lower() in {".html", ".css", ".js"}
            else "public, max-age=3600"
        )
        self._send_security_headers(cache_control)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"ok": True, "app": "Asta Coach web UI"})
            return
        if path == "/api/state":
            self._send_json(200, {"ok": True, **build_state_payload()})
            return
        if path == "/api/players":
            self._send_json(200, {"ok": True, "players": records(get_players())})
            return
        if path == "/api/lineups":
            lineups = get_lineups()
            teams = sorted(lineups["Squadra"].unique())
            team_rows = {team: lineups[lineups["Squadra"] == team] for team in teams}
            attention_cols = ["Ballottaggi", "Squalificati", "Infortunati", "InDubbio"]
            attention_teams = [
                team for team, rows in team_rows.items()
                if any(str(rows.iloc[0].get(col, "")).strip() for col in attention_cols)
            ]
            bonus_teams = [
                team for team, rows in team_rows.items()
                if rows[["Rigorista", "Piazzati", "Punizioni", "Angoli"]].any(axis=None)
            ]
            self._send_json(200, {
                "ok": True,
                "lineups": records(lineups),
                "teams": teams,
                "attention_teams": attention_teams,
                "bonus_teams": bonus_teams,
                "updated_at": datetime.fromtimestamp(
                    scraper.FORMAZIONI.stat().st_mtime
                ).strftime("%d/%m/%Y %H:%M")
                if scraper.FORMAZIONI.exists() else None,
            })
            return
        if path == "/api/scrape/status":
            with SCRAPE_LOCK:
                self._send_json(200, {"ok": True, **dict(SCRAPE_JOB)})
            return
        self._send_file(path)

    def do_POST(self):
        path = urlparse(self.path).path
        content_type = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header."})
            return
        if length < 0:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length header."})
            return
        if length > MAX_REQUEST_BYTES:
            self._send_json(413, {"ok": False, "error": "Request body troppo grande."})
            return
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            if "multipart/form-data" in content_type:
                payload = {}
            else:
                self._send_json(400, {"ok": False, "error": "Invalid JSON body."})
                return
        if not isinstance(payload, dict):
            self._send_json(400, {"ok": False, "error": "JSON body must be an object."})
            return

        try:
            self._route(path, payload, content_type, body)
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unhandled request failure for %s", path)
            self._send_json(500, {"ok": False, "error": "Errore inatteso del server."})

    def _route(self, path, payload, content_type, body):
        if path == "/api/coach":
            players = get_players()
            player = player_by_name(players, payload.get("name", ""))
            session = active_session() or {
                "meta": {
                    "budget": 500,
                    "fair": fair_values_scaled(500),
                    "role_priorities": dict(asta_core.DEFAULT_ROLE_PRIORITIES),
                },
                "purchases": [],
                "watchlist": [],
            }
            info = asta_core.coach(session, player)
            self._send_json(200, {
                "ok": True,
                "player": _clean(player.to_dict()),
                "coach": _clean(info),
                "role_count": int((players["Ruolo"] == player["Ruolo"]).sum()),
            })
            return

        if path == "/api/advice":
            session = active_session()
            if not session:
                raise ValueError(
                    "Nessuna configurazione salvata: crea budget e fair value nella tab Setup."
                )
            asta_core.ensure_session(session)
            players = get_players()
            name = payload.get("name", "")
            current_price = payload.get("current_price")
            self._send_json(200, {
                "ok": True,
                **advice_payload(players, session, name, current_price),
            })
            return

        if path == "/api/session/save":
            budget = int(payload.get("budget", 500))
            fair = payload.get("fair") or fair_values_scaled(budget)
            fair = {role: [int(v) for v in fair.get(role, [])] for role in ROLE_ORDER}
            session = asta_core.new_session(budget=budget, fair=fair)
            asta_core.save_session(session)
            set_active_session(session)
            self._send_json(200, {
                "ok": True,
                "session": session_info(session),
                "sessions": [
                    {"name": p.stem, "label": p.stem.replace("config_", "")}
                    for p in asta_core.list_sessions()
                ],
            })
            return

        if path == "/api/session/load":
            name = payload.get("name", "")
            for session_path in asta_core.list_sessions():
                if session_path.stem == name:
                    session = asta_core.load_session(str(session_path))
                    set_active_session(session)
                    self._send_json(200, {"ok": True, "session": session_info(session)})
                    return
            raise ValueError("Configurazione non trovata.")

        if path == "/api/session/teams":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            updated = [str(name).strip() for name in payload.get("teams", [])]
            selected_own = payload.get("my_team", "")
            if len(updated) != 10 or len(set(updated)) != 10 or not all(updated):
                raise ValueError("Inserisci esattamente 10 nomi distinti e non vuoti.")
            meta = session["meta"]
            renamed = dict(zip(meta["teams"], updated))
            for purchase in session["purchases"]:
                purchase["team"] = renamed.get(purchase["team"], purchase["team"])
            meta["teams"] = updated
            meta["my_team"] = renamed.get(selected_own, updated[0])
            asta_core.save_session(session)
            self._send_json(200, {
                "ok": True,
                "session": session_info(session),
                "own": own_summary(session),
            })
            return

        if path == "/api/session/plan":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            priorities = payload.get("priorities", {})
            session["meta"]["role_priorities"] = {
                role: float(priorities.get(role, 1.0)) for role in ROLE_ORDER
            }
            asta_core.save_session(session)
            self._send_json(200, {"ok": True, "session": session_info(session)})
            return

        if path == "/api/session/watch_add":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            players = get_players()
            player = player_by_name(players, payload.get("name", ""))
            tier = payload.get("tier", "B")
            note = payload.get("note", "")
            asta_core.save_watchlist_item(session, player, tier, note)
            asta_core.save_session(session)
            self._send_json(200, {"ok": True, "session": session_info(session)})
            return

        if path == "/api/session/watch_remove":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            asta_core.remove_watchlist_item(session, payload.get("name", ""))
            asta_core.save_session(session)
            self._send_json(200, {"ok": True, "session": session_info(session)})
            return

        if path == "/api/session/purchase":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            asta_core.ensure_session(session)
            players = get_players()
            player = player_by_name(players, payload.get("name", ""))
            buyer = payload.get("team", "")
            price = int(payload.get("price", 0))
            purchase = asta_core.record_purchase(session, player, buyer, price)
            asta_core.save_session(session)
            over_cap = int(purchase["price"]) - int(purchase["cap"])
            self._send_json(200, {
                "ok": True,
                "purchase": _clean(purchase),
                "over_cap": max(0, over_cap),
                "own": own_summary(session),
                "session": session_info(session),
                "market": _clean(asta_core.market_snapshot(session)),
                "alerts": roster_alerts(session, players),
            })
            return

        if path == "/api/session/undo":
            session = active_session()
            if not session:
                raise ValueError("Nessuna configurazione attiva.")
            asta_core.undo_purchase(session, payload.get("name", ""))
            asta_core.save_session(session)
            self._send_json(200, {
                "ok": True,
                "own": own_summary(session),
                "session": session_info(session),
                "market": _clean(asta_core.market_snapshot(session)),
                "alerts": roster_alerts(session, get_players()),
            })
            return

        if path == "/api/weights":
            weights = {k: float(v) for k, v in payload.get("weights", {}).items()}
            method = payload.get("method", DEFAULT_METHOD)
            weights["_method"] = method
            save_ranking_weights(weights)
            invalidate_caches()
            players = get_players()
            self._send_json(200, {
                "ok": True,
                "weights": _clean(load_ranking_weights()),
                "rho": _clean(backtest_predictor(players)),
            })
            return

        if path == "/api/excluded":
            names = {str(n) for n in payload.get("names", [])}
            save_excluded(names)
            self._send_json(200, {"ok": True, "excluded": sorted(names)})
            return

        if path == "/api/scrape/start":
            started = start_scrape()
            with SCRAPE_LOCK:
                status = dict(SCRAPE_JOB)
            self._send_json(200, {"ok": True, "started": started, "status": status})
            return

        if path == "/api/advanced":
            filename, file_bytes = parse_multipart(content_type, body)
            if not filename:
                raise ValueError("Seleziona un file CSV prima di importare.")
            handle, tmp_path = tempfile.mkstemp(suffix=".csv")
            try:
                with os.fdopen(handle, "wb") as fh:
                    fh.write(file_bytes)
                imported = scraper.import_advanced_stats(tmp_path)
            finally:
                os.unlink(tmp_path)
            invalidate_caches()
            players = get_players()
            self._send_json(200, {
                "ok": True,
                "imported": int(len(imported)),
                "with_minutes": int(imported["Minuti"].notna().sum()),
                "summary": {
                    "players": int(len(players)),
                    "with_quote": int(players["QA"].notna().sum()),
                    "with_fm": int(players["FM"].notna().sum()),
                    "starters": int(players["Starter"].sum()),
                    "with_minutes": int(players["Minuti"].notna().sum()),
                },
            })
            return

        self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WebHandler)
    server.daemon_threads = True
    LOGGER.info("Asta Coach listening on port %s — open http://127.0.0.1:%s/", PORT, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
