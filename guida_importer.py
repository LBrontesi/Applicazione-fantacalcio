"""Import structured, local data from the macOS Guida app.

The importer only opens Guida's Realm database in read-only mode.  It stores a
small, portable JSON overlay in ``data/``; the original database and the Guida
application are never touched.
"""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMPORT_FILE = DATA_DIR / "guida_import.json"
READER_FILE = PROJECT_DIR / "tools" / "guida_realm_reader.js"
DEFAULT_REALM_PATH = (
    Path.home()
    / "Library/Containers/it.quadronica.app-guida.FG-Guida/Data/Library"
    / "Application Support/Realm/default.realm"
)


def _split_ids(value):
    return [int(part) for part in str(value or "").split(";") if part.strip().isdigit()]


def _rows(raw, name):
    """Return a Realm class from either the reader's old or new shape."""
    value = raw.get(name, []) if isinstance(raw, dict) else []
    return value if isinstance(value, list) else []


def normalize_payload(raw, source_path=DEFAULT_REALM_PATH):
    """Reduce the Realm dump to stable fields understood by Asta Coach."""
    players = _rows(raw, "PlayerObject")
    by_id = {int(row["id"]): row for row in players if str(row.get("id", "")).isdigit()}

    starter_ids = set()
    formations = []
    for row in _rows(raw, "ProbableStartersObject"):
        ids = _split_ids(row.get("players"))
        starter_ids.update(ids)
        team = by_id.get(ids[0], {}).get("team", "") if ids else ""
        formation = str(row.get("formation", ""))
        formations.append({
            "team": team,
            "formation": "-".join(formation) if formation.isdigit() else formation,
            "player_ids": ids,
        })

    player_rows = []
    for row in players:
        player_id = row.get("id")
        if not str(player_id).isdigit():
            continue
        player_rows.append({
            "id": int(player_id),
            "name": str(row.get("name", "")).strip(),
            "team": str(row.get("team", "")).strip(),
            "role": str(row.get("classicRoleValue", "")).strip(),
            "starter": int(player_id) in starter_ids,
            "aptitude_index": row.get("aptIndex"),
            "attendance_index": row.get("attendanceIndex"),
            "quotation": row.get("classicQuotation"),
            "fvm": row.get("fvm"),
            "played_games": row.get("playedGames"),
            "average_vote": row.get("averageVote"),
            "average_fantavote": row.get("averageFantavote"),
            "goals": row.get("scoredGoals"),
            "assists": row.get("assists"),
            "saved_penalties": row.get("savedPenalty"),
            "scored_penalties": row.get("scoredPenalty"),
        })

    set_pieces = []
    type_map = {
        "penaltiesTakers": "Rigorista",
        "cornersTakers": "Angoli",
        "freeKicksTakers": "Punizioni",
    }
    for team_row in _rows(raw, "ProbableSetPiecesTakersObject"):
        for field, kind in type_map.items():
            entries = team_row.get(field) or []
            for entry in entries:
                player_id = entry.get("playerId") if isinstance(entry, dict) else None
                if not str(player_id).isdigit() or int(player_id) not in by_id:
                    continue
                player = by_id[int(player_id)]
                set_pieces.append({
                    "team": str(player.get("team", "")).strip(),
                    "name": str(player.get("name", "")).strip(),
                    "role": str(player.get("classicRoleValue", "")).strip(),
                    "type": kind,
                    "order": int(entry.get("order", 0) or 0) + 1,
                })

    doubts = []
    for row in _rows(raw, "ProbableDoubtObject"):
        first = by_id.get(int(row.get("idFirstPlayer", 0) or 0))
        second = by_id.get(int(row.get("idSecondPlayer", 0) or 0))
        if not first or not second:
            continue
        doubts.append({
            "team": str(first.get("team", "")).strip(),
            "first": str(first.get("name", "")).strip(),
            "second": str(second.get("name", "")).strip(),
            "first_likelihood": row.get("likelihoodFirstPlayer"),
            "second_likelihood": row.get("likelihoodSecondPlayer"),
        })

    return {
        "schema_version": 1,
        "source": "Guida (database locale)",
        "source_path": str(source_path),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "players": player_rows,
        "formations": formations,
        "set_pieces": set_pieces,
        "doubts": doubts,
    }


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def import_guida(realm_path=None):
    """Extract and persist Guida data, returning a compact import summary."""
    source = Path(realm_path) if realm_path else DEFAULT_REALM_PATH
    if not source.exists():
        raise ValueError("Database di Guida non trovato: apri Guida almeno una volta su questo Mac.")
    node = shutil.which("node")
    if not node:
        raise ValueError("Node.js non è disponibile: installalo per leggere il database locale di Guida.")
    if not READER_FILE.exists():
        raise ValueError("Lettore di Guida mancante nell'app.")
    if not (PROJECT_DIR / "node_modules" / "realm").exists():
        raise ValueError("Supporto Guida non installato. Nel Terminale esegui: npm install")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        raw_path = Path(handle.name)
    try:
        result = subprocess.run(
            [node, str(READER_FILE), str(source), str(raw_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "errore sconosciuto").strip()
            raise ValueError(f"Impossibile leggere Guida: {detail[-500:]}")
        with raw_path.open(encoding="utf-8") as handle:
            payload = normalize_payload(json.load(handle), source)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Lettura di Guida scaduta: chiudi Guida e riprova.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Dati di Guida non leggibili: {exc}") from exc
    finally:
        raw_path.unlink(missing_ok=True)

    _atomic_json(IMPORT_FILE, payload)
    return {
        "players": len(payload["players"]),
        "starters": sum(1 for row in payload["players"] if row["starter"]),
        "set_pieces": len(payload["set_pieces"]),
        "doubts": len(payload["doubts"]),
        "imported_at": payload["imported_at"],
    }


def imported_data():
    if not IMPORT_FILE.exists():
        return {}
    try:
        with IMPORT_FILE.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def players_frame():
    rows = imported_data().get("players", [])
    return pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
