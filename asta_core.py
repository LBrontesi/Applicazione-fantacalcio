import json
from datetime import datetime
from pathlib import Path

from data_loader import ROLE_ORDER, fair_values_scaled

SESSION_DIR = Path(__file__).parent / "data" / "sessions"


def new_session(budget=500, fair=None, slots=None, created=None):
    if slots is None:
        from data_loader import DEFAULT_SLOTS
        slots = dict(DEFAULT_SLOTS)
    if fair is None:
        fair = fair_values_scaled(budget)
    return {
        "meta": {
            "created": created or datetime.now().isoformat(timespec="seconds"),
            "budget": int(budget),
            "slots": dict(slots),
            "fair": fair,
        }
    }


def save_session(session):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    created = session["meta"]["created"].replace(":", "-").replace("T", "_")
    path = SESSION_DIR / f"config_{created}.json"
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2))
    return path


def list_sessions():
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("config_*.json"), reverse=True)


def load_session(path):
    return json.loads(Path(path).read_text())


def coach(session, player):
    meta = session["meta"]
    fair = meta["fair"]
    role = player["Ruolo"]
    cluster = int(player["Cluster"])
    table = fair[role]
    cap = table[min(cluster - 1, len(table) - 1)]
    return {"role": role, "cluster": cluster, "cap": cap, "fair_table": table}
