import json
import math
from datetime import datetime
from pathlib import Path

from data_loader import ROLE_ORDER, fair_values_scaled

SESSION_DIR = Path(__file__).parent / "data" / "sessions"
DEFAULT_TEAMS = [
    "bro", "giorgio", "lolo", "pistacchio", "rochira",
    "piermattei", "babbo", "bubba", "riolo", "mattia",
]
LEGACY_DEFAULT_TEAMS = [f"Manager {number}" for number in range(1, 11)]
DEFAULT_ROLE_PRIORITIES = {"P": 0.8, "D": 1.0, "C": 1.0, "A": 1.2}
WATCHLIST_ADJUSTMENTS = {"A": 0.04, "B": 0.0, "C": -0.06}


def new_session(budget=500, fair=None, slots=None, created=None, teams=None):
    if slots is None:
        from data_loader import DEFAULT_SLOTS
        slots = dict(DEFAULT_SLOTS)
    if fair is None:
        fair = fair_values_scaled(budget)
    teams = list(teams or DEFAULT_TEAMS)
    if len(teams) != 10 or len(set(teams)) != 10 or any(not name.strip() for name in teams):
        raise ValueError("Servono esattamente 10 partecipanti con nomi distinti.")
    return {
        "meta": {
            "created": created or datetime.now().isoformat(timespec="seconds"),
            "budget": int(budget),
            "slots": dict(slots),
            "fair": fair,
            "teams": teams,
            "my_team": teams[0],
        },
        "purchases": [],
    }


def ensure_session(session):
    """Upgrade sessions created before the live-auction tracker existed."""
    meta = session.setdefault("meta", {})
    teams = meta.get("teams") or list(DEFAULT_TEAMS)
    if teams == LEGACY_DEFAULT_TEAMS:
        rename = dict(zip(LEGACY_DEFAULT_TEAMS, DEFAULT_TEAMS))
        for purchase in session.get("purchases", []):
            purchase["team"] = rename.get(purchase.get("team"), purchase.get("team"))
        if meta.get("my_team") in rename:
            meta["my_team"] = rename[meta["my_team"]]
        teams = list(DEFAULT_TEAMS)
    if len(teams) != 10 or len(set(teams)) != 10:
        teams = list(DEFAULT_TEAMS)
    meta["teams"] = teams
    meta.setdefault("my_team", teams[0])
    if meta["my_team"] not in teams:
        meta["my_team"] = teams[0]
    meta.setdefault("slots", {"P": 3, "D": 8, "C": 8, "A": 6})
    priorities = meta.setdefault("role_priorities", dict(DEFAULT_ROLE_PRIORITIES))
    meta["role_priorities"] = {
        role: min(1.5, max(0.5, _number(priorities.get(role), 1.0)))
        for role in ROLE_ORDER
    }
    session.setdefault("purchases", [])
    session.setdefault("watchlist", [])
    return session


def save_session(session):
    ensure_session(session)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(session.get("_path", ""))
    if not path or path.parent != SESSION_DIR:
        created = session["meta"]["created"].replace(":", "-").replace("T", "_")
        path = SESSION_DIR / f"config_{created}.json"
    serializable = {key: value for key, value in session.items() if key != "_path"}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))
    session["_path"] = str(path)
    return path


def list_sessions():
    if not SESSION_DIR.exists():
        return []
    return sorted(SESSION_DIR.glob("config_*.json"), reverse=True)


def load_session(path):
    session = json.loads(Path(path).read_text())
    session["_path"] = str(Path(path))
    return ensure_session(session)


def coach(session, player):
    ensure_session(session)
    meta = session["meta"]
    fair = meta["fair"]
    role = player["Ruolo"]
    cluster = int(player["Cluster"])
    table = fair[role]
    cap = table[min(cluster - 1, len(table) - 1)]
    return {"role": role, "cluster": cluster, "cap": cap, "fair_table": table}


def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def team_summary(session, team):
    """Return live budget and roster counts for one participant."""
    ensure_session(session)
    if team not in session["meta"]["teams"]:
        raise ValueError("Partecipante non valido.")
    purchases = [p for p in session["purchases"] if p["team"] == team]
    by_role = {role: 0 for role in ROLE_ORDER}
    for purchase in purchases:
        role = purchase.get("role")
        if role in by_role:
            by_role[role] += 1
    spent = sum(int(purchase["price"]) for purchase in purchases)
    budget = int(session["meta"]["budget"])
    return {
        "team": team,
        "budget": budget,
        "spent": spent,
        "remaining": budget - spent,
        "by_role": by_role,
        "purchases": purchases,
    }


def auction_summary(session):
    ensure_session(session)
    return [team_summary(session, team) for team in session["meta"]["teams"]]


def purchased_names(session):
    ensure_session(session)
    return {purchase["name"] for purchase in session["purchases"]}


def auction_advice(session, player, available_players, current_price=None):
    """Calculate a personal bid ceiling without ever exceeding cluster fair value.

    The dynamic recommendation rewards a high-ranked scarce player when the
    user's role is still open, and reduces the bid when comparable alternatives
    remain. It also reserves one credit for every other open roster slot.
    """
    ensure_session(session)
    me = session["meta"]["my_team"]
    summary = team_summary(session, me)
    role = str(player["Ruolo"])
    slots = int(session["meta"]["slots"].get(role, 0))
    role_left = max(0, slots - summary["by_role"].get(role, 0))
    total_left = sum(
        max(0, int(session["meta"]["slots"].get(r, 0)) - summary["by_role"].get(r, 0))
        for r in ROLE_ORDER
    )
    fixed_cap = int(coach(session, player)["cap"])
    reserve = max(0, total_left - 1)
    affordable = max(0, summary["remaining"] - reserve)
    rank = int(_number(player.get("Rank"), 1))
    # Quality belongs to the player; opportunity depends on the live market.
    # Keep a rank fallback for old player exports without SeasonValue.
    quality = _number(player.get("SeasonValue"), -1.0)
    if quality < 0.0:
        quality = 1.0 - ((rank - 1) % 10) / 9.0
    need = role_left / slots if slots else 0.0
    priority = session["meta"]["role_priorities"].get(role, 1.0)
    priority_n = (priority - 0.5) / 1.0
    score = _number(player.get("Score"))
    cluster = int(_number(player.get("Cluster"), 1))
    alternatives = 0
    replacement_values = []
    for candidate in available_players:
        if str(candidate.get("Nome", "")) == str(player["Nome"]):
            continue
        if str(candidate.get("Ruolo", "")) != role:
            continue
        candidate_cluster = int(_number(candidate.get("Cluster"), 999))
        candidate_score = _number(candidate.get("Score"))
        if candidate_cluster <= cluster + 1 and candidate_score >= score - 0.10:
            alternatives += 1
            replacement_values.append(_number(candidate.get("SeasonValue"), 0.0))
    scarcity = 1.0 - min(alternatives, 6) / 6.0
    replacement_value = max(replacement_values, default=0.0)
    replacement_gap = max(0.0, quality - replacement_value)
    confidence = _number(player.get("DataConfidence"), 0.5)

    # Warn before a third player from the same Serie A club. The warning is
    # explicit and the recommendation is only gently reduced: it remains the
    # user's choice to pursue a deliberate stack.
    own_clubs = [
        str(purchase.get("club", "")).strip()
        for purchase in summary["purchases"]
        if str(purchase.get("club", "")).strip()
    ]
    same_club_owned = sum(club == str(player.get("Squadra", "")).strip() for club in own_clubs)
    club_stack_warning = same_club_owned >= 2
    watch_tier = next(
        (item.get("tier") for item in session["watchlist"]
         if item.get("name") == str(player["Nome"])),
        None,
    )
    watch_adjustment = WATCHLIST_ADJUSTMENTS.get(watch_tier, 0.0)
    target_ratio = (
        0.56 + 0.20 * quality + 0.13 * need + 0.05 * priority_n
        + 0.04 * scarcity + 0.07 * replacement_gap
        + 0.03 * confidence + watch_adjustment
    )
    if club_stack_warning:
        target_ratio -= 0.05
    target = max(1, round(fixed_cap * min(1.0, target_ratio)))
    personal_max = min(fixed_cap, affordable) if role_left else 0
    recommended = min(target, personal_max) if personal_max else 0
    price = _number(current_price, 0)
    price_pressure = price / personal_max if personal_max else 1.0
    opportunity = min(1.0, quality + replacement_gap) * max(0.0, 1.0 - price_pressure)
    if not role_left or not personal_max:
        verdict = "NON COMPRARE"
    elif price and price > personal_max:
        verdict = "LASCIA"
    elif price and price > recommended:
        verdict = "SOLO SE È UNA PRIORITÀ"
    else:
        verdict = "PUNTA"
    return {
        "fixed_cap": fixed_cap,
        "personal_max": personal_max,
        "recommended": recommended,
        "affordable": affordable,
        "reserve": reserve,
        "role_left": role_left,
        "total_left": total_left,
        "quality": quality,
        "confidence": confidence,
        "need": need,
        "priority": priority,
        "alternatives": alternatives,
        "scarcity": scarcity,
        "replacement_value": replacement_value,
        "replacement_gap": replacement_gap,
        "opportunity": opportunity,
        "same_club_owned": same_club_owned,
        "club_stack_warning": club_stack_warning,
        "watch_tier": watch_tier,
        "watch_adjustment": watch_adjustment,
        "verdict": verdict,
    }


def market_snapshot(session):
    """Summarise realised prices against fixed caps, role by role."""
    ensure_session(session)
    rows = []
    for role in ROLE_ORDER:
        purchases = [p for p in session["purchases"] if p.get("role") == role]
        paid = sum(_number(p.get("price")) for p in purchases)
        caps = sum(_number(p.get("cap")) for p in purchases)
        delta = (paid / caps - 1.0) if caps else 0.0
        rows.append({"role": role, "count": len(purchases), "delta": delta})
    return rows


def save_watchlist_item(session, player, tier="B", note=""):
    ensure_session(session)
    tier = str(tier).upper()
    if tier not in {"A", "B", "C"}:
        raise ValueError("La priorità deve essere A, B o C.")
    name = str(player["Nome"])
    item = {
        "name": name,
        "club": str(player.get("Squadra", "")),
        "role": str(player.get("Ruolo", "")),
        "tier": tier,
        "note": str(note).strip(),
    }
    for index, saved in enumerate(session["watchlist"]):
        if saved.get("name") == name:
            session["watchlist"][index] = item
            return item
    session["watchlist"].append(item)
    return item


def remove_watchlist_item(session, name):
    ensure_session(session)
    session["watchlist"] = [
        item for item in session["watchlist"] if item.get("name") != name
    ]


def record_purchase(session, player, team, price):
    """Register a completed auction bid after checking budget and roster slots."""
    ensure_session(session)
    try:
        price = int(price)
    except (TypeError, ValueError) as exc:
        raise ValueError("Il prezzo deve essere un numero intero.") from exc
    if price < 1:
        raise ValueError("Il prezzo deve essere almeno 1 credito.")
    if team not in session["meta"]["teams"]:
        raise ValueError("Seleziona un partecipante valido.")
    name = str(player["Nome"])
    role = str(player["Ruolo"])
    if role not in ROLE_ORDER:
        raise ValueError("Il ruolo del giocatore non è valido.")
    if name in purchased_names(session):
        raise ValueError(f"{name} risulta già assegnato.")
    summary = team_summary(session, team)
    if price > summary["remaining"]:
        raise ValueError(
            f"Budget insufficiente: a {team} restano {summary['remaining']} crediti."
        )
    max_slots = int(session["meta"]["slots"].get(role, 0))
    if summary["by_role"][role] >= max_slots:
        raise ValueError(f"{team} ha già completato gli slot {role} ({max_slots}).")
    cap = coach(session, player)["cap"]
    session["purchases"].append({
        "name": name,
        "club": str(player.get("Squadra", "")),
        "role": role,
        "price": price,
        "team": team,
        "rank": int(player.get("Rank", 0)),
        "cluster": int(player.get("Cluster", 0)),
        "cap": int(cap),
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    return session["purchases"][-1]


def undo_purchase(session, name):
    ensure_session(session)
    for index in range(len(session["purchases"]) - 1, -1, -1):
        if session["purchases"][index]["name"] == name:
            return session["purchases"].pop(index)
    raise ValueError("Acquisto non trovato.")
