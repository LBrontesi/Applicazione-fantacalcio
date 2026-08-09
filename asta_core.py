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
    session = {
        "meta": {
            "created": created or datetime.now().isoformat(timespec="seconds"),
            "budget": int(budget),
            "slots": dict(slots),
            "fair": fair,
        },
        "purchases": [],
    }
    return session


def ensure_session(session):
    """Bring saved sessions from older app versions up to the current shape."""
    session.setdefault("purchases", [])
    session.setdefault("meta", {})
    meta = session["meta"]
    meta.setdefault("budget", 500)
    meta.setdefault("fair", fair_values_scaled(meta["budget"]))
    if "slots" not in meta:
        from data_loader import DEFAULT_SLOTS
        meta["slots"] = dict(DEFAULT_SLOTS)
    return session


def save_session(session):
    ensure_session(session)
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
    return ensure_session(json.loads(Path(path).read_text()))


def roster_summary(session):
    """Return the budget and roster constraints that matter during an auction."""
    ensure_session(session)
    meta = session["meta"]
    purchases = session["purchases"]
    slots = {role: int(count) for role, count in meta["slots"].items()}
    bought_by_role = {role: 0 for role in slots}
    for purchase in purchases:
        role = purchase.get("role")
        if role in bought_by_role:
            bought_by_role[role] += 1
    remaining_by_role = {
        role: max(0, slots[role] - bought_by_role[role]) for role in slots
    }
    spent = sum(int(purchase["price"]) for purchase in purchases)
    remaining = int(meta["budget"]) - spent
    slots_remaining = sum(remaining_by_role.values())
    # Keep one credit for every other empty slot, as required in the usual
    # fantasy-football auction format.
    max_next_bid = max(0, remaining - max(0, slots_remaining - 1))
    return {
        "budget": int(meta["budget"]),
        "spent": spent,
        "remaining": remaining,
        "filled": len(purchases),
        "total_slots": sum(slots.values()),
        "slots_remaining": slots_remaining,
        "max_next_bid": max_next_bid,
        "bought_by_role": bought_by_role,
        "remaining_by_role": remaining_by_role,
    }


def record_purchase(session, player, price):
    """Add a won player, rejecting entries that would make the roster invalid."""
    ensure_session(session)
    try:
        price = int(price)
    except (TypeError, ValueError) as exc:
        raise ValueError("Il prezzo deve essere un numero intero.") from exc
    if price < 1:
        raise ValueError("Il prezzo deve essere almeno 1 credito.")

    name = str(player["Nome"]).strip()
    role = str(player["Ruolo"]).strip()
    summary = roster_summary(session)
    if not name:
        raise ValueError("Il giocatore non ha un nome valido.")
    if role not in summary["remaining_by_role"]:
        raise ValueError(f"Ruolo non valido: {role}.")
    if any(p["name"].casefold() == name.casefold()
           for p in session["purchases"]):
        raise ValueError(f"{name} è già nella tua rosa.")
    if summary["remaining_by_role"][role] <= 0:
        raise ValueError(f"Hai già riempito tutti gli slot {role}.")
    if price > summary["max_next_bid"]:
        raise ValueError(
            "Prezzo troppo alto: devi lasciare 1 credito per ogni altro "
            "slot ancora vuoto."
        )

    purchase = {
        "name": name,
        "team": str(player.get("Squadra", "")),
        "role": role,
        "price": price,
        "cluster": _optional_int(player.get("Cluster")),
        "fm": _optional_float(player.get("FM")),
    }
    session["purchases"].append(purchase)
    return purchase


def remove_purchase(session, index):
    ensure_session(session)
    try:
        return session["purchases"].pop(int(index))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("Acquisto non trovato.") from exc


def _optional_int(value):
    try:
        if value is None or value != value:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value):
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def coach(session, player):
    ensure_session(session)
    meta = session["meta"]
    fair = meta["fair"]
    role = player["Ruolo"]
    cluster = int(player["Cluster"])
    table = fair[role]
    fair_cap = table[min(cluster - 1, len(table) - 1)]
    summary = roster_summary(session)
    role_open = summary["remaining_by_role"].get(role, 0) > 0
    cap = min(fair_cap, summary["max_next_bid"]) if role_open else 0
    return {
        "role": role,
        "cluster": cluster,
        "cap": cap,
        "fair_cap": fair_cap,
        "fair_table": table,
        "max_next_bid": summary["max_next_bid"],
        "role_open": role_open,
    }
