import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from statistics import median

from data_loader import ROLE_ORDER, fair_values_scaled

SESSION_DIR = Path(__file__).parent / "data" / "sessions"
DEFAULT_TEAMS = [
    "bro", "giorgio", "lolo", "pistacchio", "rochira",
    "piermattei", "babbo", "bubba", "riolo", "mattia",
]
LEGACY_DEFAULT_TEAMS = [f"Manager {number}" for number in range(1, 11)]
DEFAULT_ROLE_PRIORITIES = {"P": 0.8, "D": 1.0, "C": 1.0, "A": 1.2}
WATCHLIST_FACTORS = {"A": 1.04, "B": 1.0, "C": 0.94}
ADVICE_VERSION = "2.0"
SCORING_PROFILE = "classic"


def validate_fair(fair):
    """Return a safe fair-value table for every role."""
    if not isinstance(fair, dict):
        raise ValueError("I fair value devono essere una tabella per ruolo.")
    clean = {}
    for role in ROLE_ORDER:
        values = fair.get(role)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"Inserisci almeno un fair value valido per il ruolo {role}.")
        try:
            parsed = [int(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fair value non valido per il ruolo {role}.") from exc
        if any(value < 1 for value in parsed):
            raise ValueError(f"I fair value del ruolo {role} devono essere almeno 1.")
        clean[role] = parsed
    return clean


def new_session(budget=500, fair=None, slots=None, created=None, teams=None):
    if slots is None:
        from data_loader import DEFAULT_SLOTS
        slots = dict(DEFAULT_SLOTS)
    if fair is None:
        fair = fair_values_scaled(budget)
    fair = validate_fair(fair)
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
            "role_priorities": dict(DEFAULT_ROLE_PRIORITIES),
            "scoring_profile": SCORING_PROFILE,
            "completed": False,
            "advice_version": ADVICE_VERSION,
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
    meta.setdefault("scoring_profile", SCORING_PROFILE)
    meta.setdefault("completed", False)
    meta.setdefault("advice_version", ADVICE_VERSION)
    try:
        meta["fair"] = validate_fair(meta.get("fair"))
    except ValueError:
        meta["fair"] = fair_values_scaled(max(1, int(meta.get("budget", 500))))
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
    contents = json.dumps(serializable, ensure_ascii=False, indent=2)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists():
        shutil.copy2(path, path.with_suffix(f"{path.suffix}.bak"))
    temporary.write_text(contents)
    temporary.replace(path)
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


def _clamp(value, low, high):
    return max(low, min(high, value))


def _planned_slot_envelope(session, summary, called_role):
    """Allocate remaining discretionary credits across the personal plan."""
    meta = session["meta"]
    open_slots = []
    for role in ROLE_ORDER:
        filled = int(summary["by_role"].get(role, 0))
        total = int(meta["slots"].get(role, 0))
        table = meta["fair"][role]
        priority = _number(meta["role_priorities"].get(role), 1.0)
        for slot_index in range(filled, total):
            fair = int(table[min(slot_index, len(table) - 1)])
            weight = max(0.0, fair - 1.0) * priority
            open_slots.append({
                "role": role,
                "slot_index": slot_index,
                "fair": fair,
                "weight": weight,
            })
    total_left = len(open_slots)
    if not total_left:
        return {
            "slot_envelope": 0.0,
            "planned_reserve": 0,
            "total_left": 0,
            "discretionary": 0,
        }
    discretionary = max(0.0, float(summary["remaining"]) - total_left)
    total_weight = sum(slot["weight"] for slot in open_slots)
    for slot in open_slots:
        share = (
            slot["weight"] / total_weight
            if total_weight > 0 else 1.0 / total_left
        )
        slot["envelope"] = 1.0 + discretionary * share
    called_index = int(summary["by_role"].get(called_role, 0))
    called = next(
        (slot for slot in open_slots
         if slot["role"] == called_role and slot["slot_index"] == called_index),
        None,
    )
    envelope = float(called["envelope"]) if called else 0.0
    hard_reserve = max(0, total_left - 1)
    planned_reserve = max(hard_reserve, round(float(summary["remaining"]) - envelope))
    return {
        "slot_envelope": envelope,
        "planned_reserve": planned_reserve,
        "total_left": total_left,
        "discretionary": discretionary,
    }


def _inflation_for_rows(rows):
    valid = [
        purchase for purchase in rows
        if _number(purchase.get("cap")) > 0 and _number(purchase.get("price")) > 0
    ]
    count = len(valid)
    if not count:
        return 0.0, 0
    paid = sum(_number(purchase.get("price")) for purchase in valid)
    caps = sum(_number(purchase.get("cap")) for purchase in valid)
    raw = paid / caps - 1.0 if caps else 0.0
    shrunk = raw * count / (count + 3.0)
    return _clamp(shrunk, -0.10, 0.10), count


def _market_adjustment(session, role, cluster):
    role_rows = [p for p in session["purchases"] if p.get("role") == role]
    role_inflation, role_count = _inflation_for_rows(role_rows)
    nearby = [
        p for p in role_rows
        if abs(int(_number(p.get("cluster"), 999)) - int(cluster)) <= 1
    ]
    cluster_inflation, cluster_count = _inflation_for_rows(nearby)
    inflation = (
        0.70 * role_inflation + 0.30 * cluster_inflation
        if cluster_count >= 3 else role_inflation
    )
    return {
        "factor": 1.0 + _clamp(inflation, -0.10, 0.10),
        "inflation": _clamp(inflation, -0.10, 0.10),
        "role_count": role_count,
        "cluster_count": cluster_count,
    }


def _opponent_demand(session, role, current_price):
    meta = session["meta"]
    opponents = [team for team in meta["teams"] if team != meta["my_team"]]
    threshold = max(1, int(_number(current_price, 1)) + 1)
    eligible = 0
    for team in opponents:
        summary = team_summary(session, team)
        has_slot = summary["by_role"].get(role, 0) < int(meta["slots"].get(role, 0))
        if has_slot and summary["remaining"] >= threshold:
            eligible += 1
    pressure = eligible / len(opponents) if opponents else 0.0
    return {
        "eligible": eligible,
        "total": len(opponents),
        "pressure": pressure,
        "factor": 1.0 + 0.10 * (pressure - 0.5),
    }


def _completed_sessions(exclude_path=None):
    sessions = []
    for path in list_sessions():
        if exclude_path and Path(exclude_path) == path:
            continue
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        sessions.append(ensure_session(saved))
    return sessions


def _historical_calibration(session, role, cluster, completed_sessions=None):
    """Empirical-Bayes price/cap factor from explicitly completed auctions."""
    meta = session["meta"]
    sessions = completed_sessions
    if sessions is None:
        sessions = _completed_sessions(session.get("_path"))
    ratios = []
    for saved in sessions:
        saved = ensure_session(saved)
        saved_meta = saved["meta"]
        if not saved_meta.get("completed"):
            continue
        if saved_meta.get("scoring_profile") != meta.get("scoring_profile"):
            continue
        if len(saved_meta.get("teams", [])) != len(meta.get("teams", [])):
            continue
        for purchase in saved.get("purchases", []):
            if purchase.get("role") != role:
                continue
            if int(_number(purchase.get("cluster"), 0)) != int(cluster):
                continue
            cap = _number(purchase.get("cap"))
            price = _number(purchase.get("price"))
            if cap > 0 and price > 0:
                ratios.append(price / cap)
    count = len(ratios)
    if count < 5:
        return {"factor": 1.0, "count": count}
    observed = median(ratios)
    factor = 1.0 + (observed - 1.0) * count / (count + 10.0)
    return {"factor": _clamp(factor, 0.85, 1.15), "count": count}


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
    plan = _planned_slot_envelope(session, summary, role)
    total_left = int(plan["total_left"])
    fixed_cap = int(coach(session, player)["cap"])
    reserve = max(0, total_left - 1)
    affordable = max(0, summary["remaining"] - reserve)
    rank = int(_number(player.get("Rank"), 1))
    # Quality belongs to the player; opportunity depends on the live market.
    # Keep a rank fallback for old player exports without SeasonValue.
    quality = _number(
        player.get(
            "RankingQuality",
            player.get("Score", player.get("SeasonValue")),
        ),
        -1.0,
    )
    if quality < 0.0:
        quality = 1.0 - ((rank - 1) % 10) / 9.0
    quality = _clamp(quality, 0.0, 1.0)
    need = role_left / slots if slots else 0.0
    priority = session["meta"]["role_priorities"].get(role, 1.0)
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
    confidence = _clamp(_number(player.get("DataConfidence"), 0.5), 0.0, 1.0)

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
    watch_factor = WATCHLIST_FACTORS.get(watch_tier, 1.0)
    club_factor = 0.95 if club_stack_warning else 1.0
    player_factor = _clamp(
        0.80 + 0.15 * quality + 0.10 * replacement_gap + 0.05 * scarcity,
        0.80,
        1.05,
    )
    risk_factor = 0.90 + 0.10 * confidence
    price = _number(current_price, 0)
    market = _market_adjustment(session, role, cluster)
    demand = _opponent_demand(session, role, price)
    historical = _historical_calibration(session, role, cluster)
    slot_envelope = float(plan["slot_envelope"])
    planned_price = min(float(fixed_cap), slot_envelope * player_factor)
    personal_max = min(fixed_cap, affordable) if role_left else 0
    adjusted = (
        planned_price * risk_factor * watch_factor * club_factor
        * market["factor"] * demand["factor"] * historical["factor"]
    )
    recommended = (
        min(max(1, round(adjusted)), personal_max, fixed_cap)
        if personal_max else 0
    )
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
        "advice_version": ADVICE_VERSION,
        "personal_max": personal_max,
        "recommended": recommended,
        "base_recommended": min(max(1, round(planned_price)), fixed_cap)
        if role_left else 0,
        "affordable": affordable,
        "reserve": reserve,
        "planned_reserve": int(plan["planned_reserve"]),
        "slot_envelope": slot_envelope,
        "budget_quota": (
            slot_envelope / summary["remaining"] if summary["remaining"] else 0.0
        ),
        "discretionary_budget": plan["discretionary"],
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
        "watch_adjustment": watch_factor - 1.0,
        "watch_factor": watch_factor,
        "club_factor": club_factor,
        "player_factor": player_factor,
        "risk_factor": risk_factor,
        "market_factor": market["factor"],
        "market_inflation": market["inflation"],
        "market_role_samples": market["role_count"],
        "market_cluster_samples": market["cluster_count"],
        "demand_factor": demand["factor"],
        "opponent_pressure": demand["pressure"],
        "eligible_bidders": demand["eligible"],
        "historical_factor": historical["factor"],
        "historical_samples": historical["count"],
        "zones": {
            "punta_until": recommended,
            "priority_until": personal_max,
            "leave_above": personal_max,
        },
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


def record_purchase(session, player, team, price, advice_snapshot=None):
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
    advice_snapshot = advice_snapshot or {}
    session["meta"]["completed"] = False
    purchase = {
        "name": name,
        "club": str(player.get("Squadra", "")),
        "role": role,
        "price": price,
        "team": team,
        "rank": int(player.get("Rank", 0)),
        "cluster": int(player.get("Cluster", 0)),
        "cap": int(cap),
        "created": datetime.now().isoformat(timespec="seconds"),
        "advice_version": str(advice_snapshot.get("advice_version", ADVICE_VERSION)),
        "recommended_at_sale": int(_number(advice_snapshot.get("recommended"), 0)),
        "personal_max_at_sale": int(_number(advice_snapshot.get("personal_max"), 0)),
        "market_factor_at_sale": _number(advice_snapshot.get("market_factor"), 1.0),
        "market_inflation_at_sale": _number(advice_snapshot.get("market_inflation"), 0.0),
        "demand_factor_at_sale": _number(advice_snapshot.get("demand_factor"), 1.0),
        "opponent_pressure_at_sale": _number(advice_snapshot.get("opponent_pressure"), 0.5),
        "risk_factor_at_sale": _number(advice_snapshot.get("risk_factor"), 1.0),
        "player_factor_at_sale": _number(advice_snapshot.get("player_factor"), 1.0),
        "historical_factor_at_sale": _number(advice_snapshot.get("historical_factor"), 1.0),
        "watch_tier_at_sale": advice_snapshot.get("watch_tier"),
        "watch_factor_at_sale": _number(advice_snapshot.get("watch_factor"), 1.0),
        "club_factor_at_sale": _number(advice_snapshot.get("club_factor"), 1.0),
        "slot_envelope_at_sale": _number(advice_snapshot.get("slot_envelope"), 0.0),
        "planned_reserve_at_sale": int(_number(advice_snapshot.get("planned_reserve"), 0)),
        "role_priority_at_sale": _number(advice_snapshot.get("priority"), 1.0),
        "quality_at_sale": _number(advice_snapshot.get("quality"), 0.5),
        "confidence_at_sale": _number(advice_snapshot.get("confidence"), 0.5),
    }
    session["purchases"].append(purchase)
    return purchase


def undo_purchase(session, name):
    ensure_session(session)
    for index in range(len(session["purchases"]) - 1, -1, -1):
        if session["purchases"][index]["name"] == name:
            session["meta"]["completed"] = False
            return session["purchases"].pop(index)
    raise ValueError("Acquisto non trovato.")


def set_session_completed(session, completed=True):
    """Opt a session in or out of future price calibration."""
    ensure_session(session)
    session["meta"]["completed"] = bool(completed)
    return session["meta"]["completed"]
