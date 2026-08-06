import json
from datetime import datetime
from pathlib import Path

from data_loader import ROLE_ORDER, fair_values_scaled

SESSION_DIR = Path(__file__).parent / "data" / "sessions"


def new_session(league_name, teams, budget=500, slots=None, fair=None, created=None):
    if slots is None:
        from data_loader import DEFAULT_SLOTS
        slots = dict(DEFAULT_SLOTS)
    if fair is None:
        fair = fair_values_scaled(budget)
    return {
        "meta": {
            "league": league_name,
            "created": created or datetime.now().isoformat(timespec="seconds"),
            "budget": int(budget),
            "slots": dict(slots),
            "fair": fair,
        },
        "teams": {
            name: {"budget_left": int(budget), "roster": [], "must_have": []}
            for name in teams
        },
        "events": [],
        "transcripts": [],
    }


def save_session(session):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    created = session["meta"]["created"].replace(":", "-").replace("T", "_")
    path = SESSION_DIR / f"asta_{created}.json"
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2))
    return path


def list_sessions():
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("asta_*.json"), reverse=True)


def load_session(path):
    return json.loads(Path(path).read_text())


def _roster_counts(session, team):
    counts = {role: 0 for role in ROLE_ORDER}
    for item in session["teams"][team]["roster"]:
        counts[item["role"]] += 1
    return counts


def coach(session, player):
    meta = session["meta"]
    slots = meta["slots"]
    fair = meta["fair"]
    role = player["Ruolo"]
    cluster = int(player["Cluster"])
    my_team = meta.get("my_team", "")
    own = session["teams"].get(my_team)
    own_roster = own["roster"] if own else []
    count = len([i for i in own_roster if i["role"] == role])
    next_free = count + 1
    cap = fair[role][min(max(cluster - 1, next_free - 1), len(fair[role]) - 1)]
    full = count >= slots[role]
    return {
        "role": role,
        "cluster": cluster,
        "filled": count,
        "slots": slots[role],
        "next_free": next_free,
        "cap": cap,
        "full": full,
        "fair_table": fair[role],
        "qa": player.get("QA"),
        "fm": player.get("FM"),
        "squadra": player.get("Squadra"),
    }


def reserve(session):
    total = 0
    my_team = session["meta"].get("my_team", "")
    roster = session["teams"].get(my_team, {}).get("roster", [])
    for item in roster:
        total += max(0, item.get("cap", 0) - item["price"])
    return total


def register_bid(session, player_name, role, cluster, buyer, price, cap, note="",
                 source="manual"):
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "player": player_name,
        "role": role,
        "cluster": cluster,
        "buyer": buyer,
        "price": int(price),
        "cap": int(cap),
        "note": note,
        "source": source,
    }
    session["events"].append(event)
    data = session["teams"][buyer]
    data["roster"].append(
        {
            "player": player_name,
            "role": role,
            "cluster": cluster,
            "price": int(price),
            "cap": int(cap),
            "note": note,
        }
    )
    data["budget_left"] = max(0, data["budget_left"] - int(price))
    return event


def undo(session):
    events = session["events"]
    if not events:
        return None
    last = events.pop()
    roster = session["teams"][last["buyer"]]["roster"]
    for i in range(len(roster) - 1, -1, -1):
        if roster[i]["player"] == last["player"] and roster[i]["price"] == last["price"]:
            del roster[i]
            break
    session["teams"][last["buyer"]]["budget_left"] += last["price"]
    return last


def market_watch(session, exclude=None):
    out = []
    for team, data in session["teams"].items():
        if team == exclude:
            continue
        counts = _roster_counts(session, team)
        out.append({"team": team, "counts": counts, "budget_left": data["budget_left"]})
    return out
