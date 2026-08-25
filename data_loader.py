import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scraper import (
    ADVANCED_STATS, DATA_DIR, FORMAZIONI, PANCHINARI, PLAYERS_FCP, QUOTAZIONI,
    SET_PIECES, STAT_TEAM_CODES, ROSTER_OVERRIDES, normalize_name,
)
from guida_importer import imported_data

ROLE_ORDER = ["P", "D", "C", "A"]
CLUSTER_SIZE = 10

RANK_WEIGHTS_FILE = DATA_DIR / "ranking_weights.json"
RANK_WEIGHTS_VERSION = 2
RANKING_VERSION = "2.0"

DEFAULT_RANK_WEIGHTS = {
    "FM": 1.0,
    "FVM": 0.3,
    "ALG": 0.3,
    "Starter": 0.5,
    "SetPieces": 0.5,
    "Tags": 0.3,
    "Injury": 0.5,
    "Availability": 0.4,
    "ExpectedOutput": 0.35,
    "GolSubiti": 0.4,
    "MediaVoto": 0.4,
    "Presenze": 0.3,
    "Rigori": 0.4,
    "Produttivita": 0.3,
    "TeamContext": 0.15,
    # In v2 confidence is no longer an additive quality bonus.  It controls
    # how strongly uncertain estimates are pulled back towards the role mean.
    "Confidence": 0.65,
}

DEFAULT_METHOD = "blend"

SP_TYPE_WEIGHTS = {
    "Rigorista": 1.0, "Piazzati": 0.6, "Punizioni": 0.6, "Angoli": 0.3,
}
# "Piazzati" (Fantacalcio) è l'alternativa editoriale a punizioni+angoli
# (FCP), non un quarto bonus che possa gonfiare il massimo teorico.
SP_MAX_SCORE = 1.9
# Il primo nome della gerarchia conta pienamente; le alternative sono utili,
# ma non equivalgono al battitore designato.
SP_PRIORITY_WEIGHTS = {1: 1.0, 2: 0.55, 3: 0.25}

ROLE_FACTORS = {
    "P": {"Starter": 1.0, "SetPieces": 0.2},
    "D": {"Starter": 1.0, "SetPieces": 1.0},
    "C": {"Starter": 0.8, "SetPieces": 1.0},
    "A": {"Starter": 0.5, "SetPieces": 0.5},
}

# Il modello cattura meglio il rendimento atteso dove i dati storici sono ricchi;
# per i portieri, che hanno meno osservazioni, il giudizio manuale resta dominante.
# Sono coefficienti prudenti per ruolo, non una promessa di accuratezza futura.
ROLE_BLEND_MODEL_WEIGHT = {"P": 0.35, "D": 0.55, "C": 0.60, "A": 0.65}

LEGACY_MODEL_FEATURES = [
    "FM2", "FM3", "ALGnum", "FVM", "TagScore", "Starter", "SetPieces",
    "InjuryP", "Availability", "MinutesPct", "StartsPct", "xG90", "xA90",
    "HistMV", "HistGol90", "HistAss90", "HistRigori", "HistPresenze",
]

ROLE_MODEL_FEATURES = {
    "P": [
        "FM2", "FM3", "ALGnum", "FVM", "Starter", "InjuryP",
        "Availability", "HistMV", "HistPresenze", "GolSubiti90",
        "RigoriParati", "TeamContext",
    ],
    "D": [
        "FM2", "FM3", "ALGnum", "FVM", "TagScore", "Starter",
        "SetPieces", "InjuryP", "Availability", "HistMV", "HistGol90",
        "HistAss90", "HistRigori", "HistPresenze", "TeamContext",
    ],
    "C": [
        "FM2", "FM3", "ALGnum", "FVM", "TagScore", "Starter",
        "SetPieces", "InjuryP", "Availability", "MinutesPct", "StartsPct",
        "xG90", "xA90", "HistMV", "HistGol90", "HistAss90",
        "HistRigori", "HistPresenze", "TeamContext",
    ],
    "A": [
        "FM2", "FM3", "ALGnum", "FVM", "TagScore", "Starter",
        "SetPieces", "InjuryP", "Availability", "MinutesPct", "StartsPct",
        "xG90", "xA90", "HistMV", "HistGol90", "HistAss90",
        "HistRigori", "HistPresenze", "TeamContext",
    ],
}

TAG_SCORES = {
    "Fuoriclasse": 2.0,
    "Goleador": 1.0,
    "Assistman": 1.0,
    "Titolare": 1.0,
    "Buona Media": 1.0,
    "Piazzati": 1.0,
    "Giovane talento": 0.0,
    "Outsider": 0.0,
    "Panchinaro": -1.0,
    "Falloso": -1.0,
}


def load_ranking_weights():
    if not RANK_WEIGHTS_FILE.exists():
        return {
            **DEFAULT_RANK_WEIGHTS,
            "_method": DEFAULT_METHOD,
            "_version": RANK_WEIGHTS_VERSION,
        }
    try:
        with open(RANK_WEIGHTS_FILE) as fh:
            saved = json.load(fh)
        legacy = int(saved.get("_version", 1) or 1) < RANK_WEIGHTS_VERSION
        out = {k: float(saved.get(k, DEFAULT_RANK_WEIGHTS[k]))
               for k in DEFAULT_RANK_WEIGHTS}
        if legacy:
            # The old value rewarded confidence as if it were quality.  Do
            # not carry that meaning into v2: start from the calibrated
            # uncertainty-shrinkage default instead.
            out["Confidence"] = DEFAULT_RANK_WEIGHTS["Confidence"]
        out["_method"] = saved.get("_method", DEFAULT_METHOD)
        out["_version"] = RANK_WEIGHTS_VERSION
        return validate_ranking_weights(out)
    except (OSError, ValueError, KeyError):
        return {
            **DEFAULT_RANK_WEIGHTS,
            "_method": DEFAULT_METHOD,
            "_version": RANK_WEIGHTS_VERSION,
        }


def save_ranking_weights(weights):
    DATA_DIR.mkdir(exist_ok=True)
    data = validate_ranking_weights(weights)
    with open(RANK_WEIGHTS_FILE, "w") as fh:
        json.dump(data, fh, indent=2)


def validate_ranking_weights(weights):
    """Reject values that would make ranking scores non-finite or inverted."""
    if not isinstance(weights, dict):
        raise ValueError("I pesi classifica devono essere un oggetto.")
    data = {}
    for key, default in DEFAULT_RANK_WEIGHTS.items():
        try:
            value = float(weights.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Peso non valido: {key}.") from exc
        if not np.isfinite(value) or not 0.0 <= value <= 2.0:
            raise ValueError(f"Il peso {key} deve essere compreso tra 0 e 2.")
        data[key] = value
    method = weights.get("_method", DEFAULT_METHOD)
    if method not in {"blend", "model", "manual"}:
        raise ValueError("Metodo ranking non valido.")
    data["_method"] = method
    data["_version"] = RANK_WEIGHTS_VERSION
    return data

DEFAULT_FAIR = {
    "P": [56, 1, 1],
    "D": [37, 18, 12, 5, 1, 1, 1, 1],
    "C": [135, 70, 32, 18, 12, 1, 1, 1],
    "A": [322, 150, 97, 26, 1, 1],
}

DEFAULT_SLOTS = {"P": 3, "D": 8, "C": 8, "A": 6}


def _to_float(value):
    try:
        return float(str(value).replace(",", ".").replace("N/A", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def fair_values_scaled(budget, base=DEFAULT_FAIR):
    scale = max(budget, 1) / 1000.0
    out = {}
    for role, values in base.items():
        out[role] = [max(1, round(v * scale)) for v in values]
    return out


def _rank_norm(s):
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True)
    return r.where(s.notna(), 0.0)


def _rank_norm_neutral(s, neutral=0.5):
    """Percentile-rank observed continuous data; keep missing values neutral."""
    s = pd.to_numeric(s, errors="coerce")
    observed = s.notna()
    if not observed.any():
        return pd.Series(float(neutral), index=s.index, dtype=float)
    ranked = s[observed].rank(pct=True)
    out = pd.Series(float(neutral), index=s.index, dtype=float)
    out.loc[observed] = ranked
    return out


def _tag_score(attr):
    if not isinstance(attr, str) or not attr:
        return 0.0
    total = 0.0
    for tag in attr.split("|"):
        total += TAG_SCORES.get(tag.strip(), 0.0)
    return max(-2.0, min(2.0, total))


def _set_piece_score(entries):
    """Score set-pieces, accepting legacy type names or (type, order) pairs."""
    best_by_type = {}
    for entry in entries:
        if isinstance(entry, tuple):
            tipo, order = entry
        else:
            tipo, order = entry, 1
        try:
            order = max(1, int(order))
        except (TypeError, ValueError):
            order = 1
        best_by_type[tipo] = min(order, best_by_type.get(tipo, order))
    weighted = sum(
        SP_TYPE_WEIGHTS.get(tipo, 0.0)
        * SP_PRIORITY_WEIGHTS.get(order, 0.10)
        for tipo, order in best_by_type.items()
    )
    return min(SP_MAX_SCORE, weighted) / SP_MAX_SCORE


def _confidence_label(value):
    if value >= 0.75:
        return "Alta"
    if value >= 0.45:
        return "Media"
    return "Bassa"


def load_fcp():
    if not PLAYERS_FCP.exists():
        return pd.DataFrame()
    df = pd.read_csv(PLAYERS_FCP)
    df["NomeFCP"] = df["NomeFCP"].astype(str).str.title()
    return df


def load_quotazioni():
    if not QUOTAZIONI.exists():
        return pd.DataFrame()
    df = pd.read_csv(QUOTAZIONI)
    if not ROSTER_OVERRIDES.exists() or "NomeGaz" not in df.columns:
        return df
    try:
        overrides = pd.read_csv(ROSTER_OVERRIDES)
    except (OSError, ValueError):
        return df
    if overrides.empty or not {"NomeGaz", "AttivoSerieA"}.issubset(overrides.columns):
        return df
    excluded = overrides[
        overrides.get("AttivoSerieA", True).astype(str).str.lower().isin(
            {"0", "false", "no", "fuori_serie_a", "escluso"}
        )
    ]
    if excluded.empty:
        return df
    blocked = {normalize_name(name) for name in excluded["NomeGaz"].dropna()}
    return df[~df["NomeGaz"].astype(str).map(normalize_name).isin(blocked)].copy()


def load_advanced_stats():
    if not ADVANCED_STATS.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(ADVANCED_STATS)
    except (OSError, ValueError):
        return pd.DataFrame()
    for col in ["NomeStats", "SquadraStats"]:
        if col not in df:
            return pd.DataFrame()
        df[col] = df[col].fillna("").astype(str)
    return df


def load_guida_data():
    """Optional, locally imported overlay from the Guida macOS app."""
    data = imported_data()
    return data if isinstance(data, dict) else {}


def _guida_player_matches(df, guida_players):
    """Map Guida records conservatively: team, role and a clear name match."""
    matches = {}
    for index, row in df.iterrows():
        team = normalize_name(str(row.get("Squadra", "")))
        role = str(row.get("Ruolo", ""))
        candidates = []
        for candidate in guida_players:
            if normalize_name(str(candidate.get("team", ""))) != team:
                continue
            if str(candidate.get("role", "")) != role:
                continue
            score = _short_source_name_score(
                str(row.get("NomeGaz", row.get("Nome", ""))),
                str(candidate.get("name", "")),
            )
            if score >= 0.90:
                candidates.append((score, candidate))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.04:
            continue
        matches[index] = candidates[0][1]
    return matches


def load_statistiche():
    """Season stats tables from fantacalcio.it (one CSV per season)."""
    frames = []
    paths = sorted(DATA_DIR.glob("statistiche_*.csv"))
    for index, path in enumerate(paths):
        try:
            frame = pd.read_csv(path)
            # The newest season receives full weight; older seasons decay so
            # that a player's current level matters more than distant form.
            frame["Stagione"] = path.stem.removeprefix("statistiche_")
            frame["PesoRecenza"] = 0.70 ** (len(paths) - index - 1)
            frames.append(frame)
        except (OSError, ValueError):
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _keeper_name_score(player_name, stat_name):
    """Strict name match for goalkeepers against the season stats table.

    Both names are reduced to their significant tokens (single-letter
    initials are dropped, so 'Milinkovic-Savic V.' loses its trailing 'V').
    A candidate matches when its tokens equal the player's or form a prefix
    of them ('Milinkovic-Savic' -> 'Milinkovic Savic Vanja'), or when a
    single-surname stat name equals the player's first surname. Rejecting
    single-letter prefixes avoids 'V.' matching an unrelated 'Vismara'.
    """
    p_tokens = _tokens(player_name)
    s_tokens = _tokens(stat_name)
    if not p_tokens or not s_tokens:
        return 0.0
    p_sig = [t for t in p_tokens if len(t) > 1]
    s_sig = [t for t in s_tokens if len(t) > 1]
    if not s_sig:
        return 0.0
    if p_sig == s_sig:
        return 1.0
    shorter, longer = (s_sig, p_sig) if len(s_sig) <= len(p_sig) else (p_sig, s_sig)
    if longer[:len(shorter)] == shorter and len(shorter) >= 2:
        return 0.98
    if len(s_sig) == 1 and len(p_sig) >= 1 and s_sig[0] == p_sig[0]:
        return 0.95
    return 0.0


def _apply_season_stats(players):
    """Fill historical season stats for every player from fantacalcio.it.

    Goalkeepers are the rows with GolSubiti/RigoriParati > 0, everyone else
    is an outfield row. Totals (goals, assists, appearances, penalties) are
    summed across the scraped seasons and MediaVoto is averaged weighted by
    appearances; a player who switched clubs is matched by name. For keepers
    with a meaningful sample (10+ appearances) a goals-conceded-per-
    appearance rate is also derived. Rows without a match stay NaN (not in
    Serie A).
    """
    stats = load_statistiche()
    if stats.empty:
        return players
    stats = stats.copy()
    stats["Squadra"] = stats["Squadra"].map(
        lambda code: STAT_TEAM_CODES.get(str(code).strip().upper(), str(code).strip())
    )
    for col in ["GolSubiti", "RigoriParati"]:
        if col not in stats:
            stats[col] = 0.0
    stats["GolSubiti"] = pd.to_numeric(stats["GolSubiti"], errors="coerce").fillna(0.0)
    stats["RigoriParati"] = pd.to_numeric(stats["RigoriParati"], errors="coerce").fillna(0.0)
    # The stats source has no role column.  A goalkeeper can nevertheless
    # have a season with appearances and zero goals conceded, so a stat name
    # counts as a keeper as soon as ANY of its seasons concedes goals or
    # saves penalties.  Grouping case-insensitively also keeps name-variant
    # seasons together.  This retains a keeper's zero-conceded seasons without
    # misclassifying same-surname outfielders (e.g. "Martinez L." the striker
    # vs "Martinez Jo." the keeper, which a name-only rule merges at 0.95).
    nome_key = stats["Nome"].astype(str).str.strip().str.lower()
    is_keeper = (
        stats.groupby(nome_key)["GolSubiti"].transform("max") > 0
    ) | (
        stats.groupby(nome_key)["RigoriParati"].transform("max") > 0
    )
    keeper_rows = stats[is_keeper].copy()
    outfield_rows = stats[~is_keeper].copy()

    def aggregate(frame):
        if frame.empty:
            return pd.DataFrame()
        frame = frame.copy()
        frame["PesoRecenza"] = pd.to_numeric(
            frame.get("PesoRecenza", 1.0), errors="coerce"
        ).fillna(1.0)
        for col in ["Gol", "Assist", "Presenze", "RigoriSegnati", "GolSubiti", "RigoriParati"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        # Group on a case-insensitive key: the same player can appear as
        # "Ederson D.s." in one season and "Ederson D.S." in another, and a
        # case-sensitive split would drop whole seasons from the sample.
        frame["_NomeKey"] = frame["Nome"].astype(str).str.strip().str.lower()
        out = frame.groupby("_NomeKey", as_index=False).agg(
            Nome=("Nome", "first"),
            Gol=("Gol", "sum"),
            Assist=("Assist", "sum"),
            Presenze=("Presenze", "sum"),
            RigoriSegnati=("RigoriSegnati", "sum"),
            GolSubiti=("GolSubiti", "sum"),
            RigoriParati=("RigoriParati", "sum"),
            StatSeasons=("Stagione", "nunique"),
            StatTeams=("Squadra", lambda values: "|".join(sorted(set(
                str(value) for value in values if str(value).strip()
            )))),
        )
        def weighted_rates(group):
            exposure = group["Presenze"] * group["PesoRecenza"]
            denominator = float(exposure.sum())
            if denominator <= 0:
                return pd.Series({
                    "MediaVoto": float("nan"), "GolRate": float("nan"),
                    "AssistRate": float("nan"), "GolSubitiRate": float("nan"),
                })
            return pd.Series({
                "MediaVoto": float(np.average(
                    pd.to_numeric(group["MediaVoto"], errors="coerce").fillna(0.0),
                    weights=exposure,
                )),
                "GolRate": float((group["Gol"] * group["PesoRecenza"]).sum() / denominator),
                "AssistRate": float((group["Assist"] * group["PesoRecenza"]).sum() / denominator),
                "GolSubitiRate": float((group["GolSubiti"] * group["PesoRecenza"]).sum() / denominator),
            })
        weighted = frame.groupby("_NomeKey").apply(
            weighted_rates, include_groups=False
        ).reset_index()
        return out.merge(weighted, on="_NomeKey")

    agg_keeper = aggregate(keeper_rows)
    agg_outfield = aggregate(outfield_rows)
    for col in ["Gol", "Assist", "Presenze", "RigoriSegnati", "GolSubiti", "RigoriParati"]:
        for frame in [agg_keeper, agg_outfield]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    def match_candidates(name, candidates):
        best, best_score = None, 0.0
        for _, candidate in candidates.iterrows():
            score = _keeper_name_score(name, str(candidate["Nome"]))
            if score > best_score:
                best, best_score = candidate, score
        return best, best_score

    for index, row in players.iterrows():
        name = str(row.get("NomeGaz", "")) or str(row.get("Nome", ""))
        if row["Ruolo"] == "P":
            candidate, score = match_candidates(name, agg_keeper)
        else:
            candidate, score = match_candidates(name, agg_outfield)
        if candidate is None or score < 0.9:
            continue
        players.at[index, "HistGol"] = float(candidate["Gol"])
        players.at[index, "HistAss"] = float(candidate["Assist"])
        players.at[index, "HistPresenze"] = float(candidate["Presenze"])
        players.at[index, "HistRigori"] = float(candidate["RigoriSegnati"])
        players.at[index, "HistMV"] = float(candidate["MediaVoto"]) if pd.notna(candidate["MediaVoto"]) else float("nan")
        players.at[index, "StatSeasons"] = float(candidate["StatSeasons"])
        historical_teams = {
            normalize_name(team) for team in str(candidate.get("StatTeams", "")).split("|")
            if team
        }
        players.at[index, "CurrentClubSeen"] = float(
            normalize_name(str(row.get("Squadra", ""))) in historical_teams
        ) if historical_teams else float("nan")
        presenze = float(candidate["Presenze"])
        if presenze > 0:
            # The source publishes appearances but not minutes.  Keep these
            # features as per-appearance rates rather than claiming a false
            # per-90 calculation.
            players.at[index, "HistGol90"] = float(candidate["GolRate"])
            players.at[index, "HistAss90"] = float(candidate["AssistRate"])
        else:
            players.at[index, "HistGol90"] = float("nan")
            players.at[index, "HistAss90"] = float("nan")
        if row["Ruolo"] == "P" and presenze >= 10:
            players.at[index, "GolSubiti"] = float(candidate["GolSubiti"]) \
                if "GolSubiti" in candidate else float("nan")
            players.at[index, "GolSubiti90"] = float(candidate["GolSubitiRate"])
            players.at[index, "RigoriParati"] = float(candidate["RigoriParati"]) \
                if "RigoriParati" in candidate else float("nan")
    return players


def _tokens(name):
    return normalize_name(name).split()


def _gaz_parts(tokens):
    if len(tokens) > 1 and len(tokens[-1]) == 1:
        return tokens[:-1], tokens[-1]
    return tokens, None


def _fcp_parts(tokens):
    return tokens[:-1], tokens[-1]


def _match_score(g_tokens, f_tokens):
    """Conservative match for source-specific spellings of the same player."""
    if not g_tokens or not f_tokens:
        return 0.0
    g_surname, g_init = _gaz_parts(g_tokens)
    f_surname, f_given = _fcp_parts(f_tokens)
    if g_tokens == f_tokens:
        return 1.0
    if g_surname == f_surname:
        if g_init is None:
            return 0.95
        if f_given.startswith(g_init):
            return 1.0
        return 0.0
    ratio = difflib.SequenceMatcher(
        None, " ".join(g_tokens), " ".join(f_tokens)
    ).ratio()
    if ratio >= 0.86:
        return 0.85

    # Fonti diverse alternano nome/cognome, omettono secondi nomi o usano
    # abbreviazioni (es. "Ederson D.S." vs "Ederson Dos Santos"). Abbiniamo
    # solo se ogni token della quotazione trova una controparte non ambigua.
    remaining = list(f_tokens)
    matched = 0
    for token in sorted(g_tokens, key=len, reverse=True):
        position = next(
            (i for i, candidate in enumerate(remaining)
            if token == candidate or (
                len(token) >= 2 and len(candidate) >= 2 and (
                    candidate.startswith(token) or token.startswith(candidate)
                )
            )),
            None,
        )
        if position is not None:
            remaining.pop(position)
            matched += 1
            continue
        # "ds" è l'acronimo dei token FCP ancora liberi: "dos santos".
        initials = "".join(part[0] for part in remaining if part)
        if len(token) >= 2 and token == initials:
            remaining = []
            matched += 1
    if matched == len(g_tokens):
        return 0.98 if len(g_tokens) >= 2 else 0.92
    return 0.0


def _short_source_name_score(full_name, source_name):
    """Match a short editorial label ("Ederson") to a quoted player name."""
    full_tokens = _tokens(full_name)
    source_tokens = _tokens(source_name)
    score = _match_score(full_tokens, source_tokens)
    if score >= 0.90:
        return score
    # Editorial set-piece lists often omit all secondary names. This is safe
    # only when every listed token occurs verbatim in the full player name;
    # callers still resolve ties within the same club.
    if source_tokens and all(
        any(token == full or (len(token) >= 3 and full.startswith(token))
            for full in full_tokens)
        for token in source_tokens
    ):
        return 0.91
    return 0.0


def _advanced_for_player(name, team, stats):
    """Find one imported row; club is used as a tie-breaker for homonyms."""
    if stats.empty:
        return None
    player_tokens = _tokens(name)
    wanted_team = normalize_name(team)
    best, best_score = None, 0.0
    for _, row in stats.iterrows():
        stat_tokens = _tokens(row["NomeStats"])
        score = _match_score(player_tokens, stat_tokens)
        if set(player_tokens) == set(stat_tokens):
            score = 1.0
        if score <= 0:
            continue
        stat_team = normalize_name(row.get("SquadraStats", ""))
        if wanted_team and stat_team and wanted_team == stat_team:
            score += 0.05
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= 0.85 else None


def _resolve_fcp_player(gaz_row, fcp):
    """Resolve a quote row against the current FCP roster.

    Same-club matches remain the safest choice. If the quote source still
    carries an old club, accept a unique, very strong match in another Serie A
    club and let that current roster win. Ambiguous or weak matches are left
    untouched and marked for review by the caller.
    """
    role = str(gaz_row.get("Ruolo", ""))
    name_tokens = _tokens(gaz_row.get("NomeGaz", ""))
    team = normalize_name(gaz_row.get("SquadraNome", ""))
    role_rows = fcp[fcp["Ruolo"] == role]

    def ranked(frame):
        values = [
            {
                "score": _match_score(name_tokens, _tokens(row["NomeFCP"])),
                "name": str(row["NomeFCP"]),
                "team": str(row["SquadraFCP"]),
            }
            for _, row in frame.iterrows()
        ]
        return sorted(values, key=lambda item: item["score"], reverse=True)

    same_team = ranked(
        role_rows[role_rows["SquadraFCP"].fillna("").map(normalize_name) == team]
    )
    if same_team and same_team[0]["score"] >= 0.90:
        if len(same_team) == 1 or same_team[0]["score"] - same_team[1]["score"] >= 0.04:
            return {**same_team[0], "status": "confermato_fcp"}

    # A transfer is accepted only with a stronger threshold and a clear
    # winner. This avoids moving homonyms or short surnames by guesswork.
    global_matches = ranked(role_rows)
    if global_matches and global_matches[0]["score"] >= 0.95:
        winner = global_matches[0]
        runner_up = global_matches[1]["score"] if len(global_matches) > 1 else 0.0
        if winner["score"] - runner_up >= 0.04:
            return {**winner, "status": "club_corretto_fcp"}
    return None


def build_players(progress_cb=None, weights=None):
    gaz = load_quotazioni()
    fcp = load_fcp()
    advanced = load_advanced_stats()
    guida = load_guida_data()

    matched_fcp = {}
    for _, g in gaz.iterrows():
        resolved = _resolve_fcp_player(g, fcp)
        if resolved:
            matched_fcp[g["NomeGaz"]] = resolved

    matched_rows = []
    for _, g in gaz.iterrows():
        fcp_match = matched_fcp.get(g["NomeGaz"])
        fcp_name = fcp_match["name"] if fcp_match else None
        f = fcp[fcp["NomeFCP"] == fcp_name].iloc[0] if fcp_name else None
        roster_status = fcp_match["status"] if fcp_match else "non_confermato"
        row = {
            "Nome": f["NomeFCP"] if f is not None and f["NomeFCP"] else g["NomeGaz"],
            "NomeGaz": g["NomeGaz"],
            "Squadra": (
                f["SquadraFCP"]
                if f is not None and fcp_match["status"] == "club_corretto_fcp"
                else g["SquadraNome"] or g["SquadraGaz"]
            ),
            "Ruolo": g["Ruolo"],
            "RosterStatus": roster_status,
            "QI": g["QI"],
            "QA": g["QA"],
            "FVM": g["FVM"],
        }
        if f is not None:
            row.update(
                {
                    "ALG": f["ALG"],
                    "FM1": _to_float(f["FM1"]),
                    "FM2": _to_float(f["FM2"]),
                    "FM3": _to_float(f["FM3"]),
                    "Attributi": f["Attributi"],
                    "ResInf": f["ResInf"],
                    "SquadraFCP": f["SquadraFCP"],
                }
            )
        else:
            row.update({"ALG": "", "FM1": float("nan"), "FM2": float("nan"),
                        "FM3": float("nan"), "Attributi": "", "ResInf": "",
                        "SquadraFCP": ""})
        advanced_row = _advanced_for_player(row["Nome"], row["Squadra"], advanced)
        if advanced_row is None:
            advanced_row = _advanced_for_player(
                row["NomeGaz"], row["Squadra"], advanced
            )
        for col in [
            "Presenze", "Titolarita", "Minuti", "xG", "xA", "Gol", "Assist",
            "Gialli", "Rossi", "xGI90", "GiorniInfortunio", "GareSaltate",
            "GolSubiti", "GolSubiti90", "CleanSheet",
        ]:
            row[col] = _to_float(advanced_row[col]) if advanced_row is not None and col in advanced_row else float("nan")
        matched_rows.append(row)

    df = pd.DataFrame(matched_rows)
    if df.empty:
        return df

    fm_cols = [c for c in ["FM1", "FM2", "FM3"] if c in df.columns]
    if fm_cols:
        fm_arr = df[fm_cols].to_numpy(dtype=float)
        avail = ~np.isnan(fm_arr)
        decay = np.array([0.6, 0.3, 0.1])[:len(fm_cols)]
        denom = avail @ decay
        weighted = np.where(avail, fm_arr, 0.0) @ decay
        df["FM"] = np.divide(
            weighted,
            denom,
            out=np.full(len(df), float("nan")),
            where=denom > 0,
        )
        df["FMAvg"] = np.where(
            avail.any(axis=1),
            np.nansum(fm_arr, axis=1) / np.maximum(avail.sum(axis=1), 1),
            float("nan"),
        )
        df["HistorySeasons"] = avail.sum(axis=1).astype(int)
    else:
        df["FM"] = float("nan")
        df["FMAvg"] = float("nan")
        df["HistorySeasons"] = 0

    df["ALGnum"] = pd.to_numeric(
        df["ALG"].astype(str).str.extract(
            r"(\d+\.?\d*)\s*/\s*100", expand=False
        ),
        errors="coerce",
    ) / 100.0
    df["InjuryP"] = pd.to_numeric(
        df["ResInf"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    ) / 100.0
    df["TagScore"] = df["Attributi"].fillna("").apply(_tag_score)

    starter_keys = set()
    form = load_formazioni()
    if not form.empty:
        for _, formation in form.iterrows():
            team = normalize_name(str(formation.get("Squadra", "")))
            tit = formation.get("Titolari", "")
            for n in str(tit).split("|"):
                if n:
                    starter_keys.add((team, normalize_name(n)))
    df["Starter"] = df.apply(
        lambda r: 1.0 if (
            normalize_name(str(r["Squadra"])),
            normalize_name(str(r["NomeGaz"])),
        ) in starter_keys else 0.0,
        axis=1,
    )

    # Guida is a local editorial source. Its expected XI can confirm a
    # starter missed by the web source, but never turns a web starter off.
    # The raw Guida fields stay visible for auditability and future tuning.
    guida_matches = _guida_player_matches(df, guida.get("players", []))
    for col in [
        "GuidaStarter", "GuidaQuotation", "GuidaFVM", "GuidaPresenze",
        "GuidaMV", "GuidaFM", "GuidaGol", "GuidaAssist",
        "GuidaRigoriSegnati", "GuidaRigoriParati", "GuidaAttendanceIndex",
    ]:
        df[col] = float("nan")
    field_map = {
        "GuidaStarter": "starter", "GuidaQuotation": "quotation",
        "GuidaFVM": "fvm", "GuidaPresenze": "played_games",
        "GuidaMV": "average_vote", "GuidaFM": "average_fantavote",
        "GuidaGol": "goals", "GuidaAssist": "assists",
        "GuidaRigoriSegnati": "scored_penalties",
        "GuidaRigoriParati": "saved_penalties",
        "GuidaAttendanceIndex": "attendance_index",
    }
    for index, source in guida_matches.items():
        for column, key in field_map.items():
            if column == "GuidaStarter":
                df.at[index, column] = 1.0 if bool(source.get(key)) else 0.0
            else:
                df.at[index, column] = _to_float(source.get(key))
    df["GuidaStarter"] = df["GuidaStarter"].fillna(0.0).clip(0.0, 1.0)
    df["Starter"] = np.maximum(df["Starter"], df["GuidaStarter"])

    sp_types = {}
    sp = load_set_pieces()
    if not sp.empty:
        for _, r in sp.iterrows():
            team = normalize_name(str(r["Squadra"]))
            sp_types.setdefault(team, []).append((
                str(r["Giocatore"]), str(r["Tipo"]), r.get("Ordine", 1),
            ))

    # Guida fills only missing team/type lists. Confirmed rows from the web
    # source retain priority, which prevents a stale offline guide from
    # silently replacing a newer hierarchy.
    existing_types = {
        (team, str(kind))
        for team, entries in sp_types.items()
        for _, kind, _ in entries
    }
    for entry in guida.get("set_pieces", []):
        team = normalize_name(str(entry.get("team", "")))
        kind = str(entry.get("type", ""))
        if not team or kind not in SP_TYPE_WEIGHTS or (team, kind) in existing_types:
            continue
        if str(entry.get("role", "")) == "P":
            continue
        sp_types.setdefault(team, []).append((
            str(entry.get("name", "")), kind, entry.get("order", 1),
        ))

    def _sp_score(team, name, role):
        # Set-piece lists describe attacking dead balls.  A fuzzy surname or
        # initial must never turn a goalkeeper into a penalty taker.
        if role == "P":
            return 0.0
        candidates = [
            (_short_source_name_score(name, sp_name), tipo, order)
            for sp_name, tipo, order in sp_types.get(team, [])
        ]
        return _set_piece_score([
            (tipo, order) for score, tipo, order in candidates if score >= 0.90
        ])

    df["SetPieces"] = df.apply(
        lambda r: _sp_score(
            normalize_name(str(r["Squadra"])), str(r["NomeGaz"]), str(r["Ruolo"])
        ),
        axis=1,
    )

    # Proxy esplicito: non spacciamo una previsione dei minuti senza una fonte
    # affidabile di presenze/minuti. Combina formazione probabile e robustezza.
    df["Availability"] = (
        0.35 + 0.45 * df["Starter"] + 0.20 * df["InjuryP"].fillna(0.5)
    ).clip(0.0, 1.0)
    # Le statistiche importate coprono il passato: non sostituiscono le
    # formazioni probabili, ma rendono meno ottimistica la stima per chi ha
    # giocato poco o ha iniziato raramente.
    df["MinutesPct"] = (df["Minuti"] / 3000.0).clip(0.0, 1.0)
    df["StartsPct"] = (df["Titolarita"] / 33.0).clip(0.0, 1.0)
    df["xG90"] = np.where(
        df["Minuti"] > 0, df["xG"] / df["Minuti"] * 90.0, float("nan")
    )
    df["xA90"] = np.where(
        df["Minuti"] > 0, df["xA"] / df["Minuti"] * 90.0, float("nan")
    )
    computed_xgi90 = df["xG90"] + df["xA90"]
    df["xGI90"] = df["xGI90"].where(df["xGI90"].notna(), computed_xgi90)
    historical_usage = 0.55 * df["MinutesPct"] + 0.45 * df["StartsPct"]
    df["HistoricalUsage"] = historical_usage
    has_usage = historical_usage.notna()
    df.loc[has_usage, "Availability"] = (
        0.55 * df.loc[has_usage, "Availability"]
        + 0.45 * historical_usage.loc[has_usage]
    ).clip(0.0, 1.0)
    injury_history = (1.0 - df["GareSaltate"] / 20.0).clip(0.0, 1.0)
    has_injury_history = injury_history.notna()
    df.loc[has_injury_history, "Availability"] = (
        0.85 * df.loc[has_injury_history, "Availability"]
        + 0.15 * injury_history.loc[has_injury_history]
    ).clip(0.0, 1.0)
    df["HistoryStrength"] = (
        df["HistorySeasons"].clip(lower=0, upper=3) / 3.0
    )

    # Storico stagionale fantacalcio.it per TUTTI i ruoli (gol, assist,
    # presenze, media voto, rigori) + gol subiti/90 per i portieri. Le nuove
    # componenti C_* vengono calcolate nel loop di ranking qui sotto.
    df = _apply_season_stats(df)
    stat_seasons = pd.to_numeric(df.get("StatSeasons"), errors="coerce").fillna(0.0)
    stat_appearances = pd.to_numeric(df.get("HistPresenze"), errors="coerce").fillna(0.0)
    df["StatsConfidence"] = (
        0.55 * (stat_seasons / 3.0).clip(0.0, 1.0)
        + 0.45 * (stat_appearances / 90.0).clip(0.0, 1.0)
    )
    # A move to a new club can change role, coach and competition. Historical
    # production remains useful, but carries slightly less certainty until
    # current-team data has accumulated.
    club_seen = pd.to_numeric(df.get("CurrentClubSeen"), errors="coerce")
    df["TransferUncertainty"] = (1.0 - club_seen).where(club_seen.notna(), 0.0)
    df["StatsConfidence"] *= (1.0 - 0.15 * df["TransferUncertainty"])

    if weights is None:
        weights = load_ranking_weights()

    df["FMEst"] = df["FM"].astype(float)
    df["FMImputed"] = False
    for role in ROLE_ORDER:
        known = df[(df["Ruolo"] == role) & df["FM"].notna()]
        if len(known) < 20:
            continue
        feats = known[["ALGnum", "FVM"]].to_numpy(dtype=float)
        ok = ~np.isnan(feats).any(axis=1)
        if ok.sum() < 15:
            continue
        coef, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(ok.sum()), feats[ok]]),
            known.loc[ok, "FM"].to_numpy(dtype=float),
            rcond=None,
        )
        missing = df.index[(df["Ruolo"] == role) & df["FM"].isna()]
        if len(missing) == 0:
            continue
        feature_means = np.nanmean(feats[ok], axis=0)
        Xm = df.loc[missing, ["ALGnum", "FVM"]].to_numpy(dtype=float)
        Xm = np.where(np.isnan(Xm), feature_means, Xm)
        pred = np.column_stack([np.ones(len(missing)), Xm]) @ coef
        lo = float(known.loc[ok, "FM"].min())
        hi = float(known.loc[ok, "FM"].max())
        df.loc[missing, "FMEst"] = np.clip(pred, lo, hi)
        df.loc[missing, "FMImputed"] = True

    # Una sola stagione osservata è utile, ma troppo rumorosa per dominare il
    # ranking. La riportiamo gradualmente verso la mediana del ruolo; FM resta
    # sempre disponibile come valore storico grezzo nella UI.
    for role in ROLE_ORDER:
        role_rows = df["Ruolo"] == role
        observed = role_rows & df["FM"].notna()
        if not observed.any():
            continue
        prior = float(df.loc[observed, "FM"].median())
        strength = df.loc[observed, "HistoryStrength"]
        shrink = strength / (strength + 0.50)
        df.loc[observed, "FMEst"] = (
            shrink * df.loc[observed, "FM"] + (1.0 - shrink) * prior
        )

    source_coverage = (
        df[["ALGnum", "FVM", "InjuryP"]].notna().sum(axis=1) / 3.0
    )
    advanced_coverage = df[["Minuti", "Titolarita", "xG", "xA", "xGI90"]].notna().sum(axis=1) / 5.0
    df["DataConfidence"] = (
        0.35 * df["HistoryStrength"] + 0.20 * df["StatsConfidence"]
        + 0.20 * source_coverage + 0.10 * (~df["FMImputed"]).astype(float)
        + 0.15 * advanced_coverage
    ).clip(0.0, 1.0)
    df["ConfidenceLabel"] = df["DataConfidence"].apply(_confidence_label)

    # Current club context: a robust team-level signal from current FVM and
    # the source algorithm. It is deliberately small, so it contextualises a
    # player without turning a team ranking into a player ranking.
    current_quality = (
        0.60 * _rank_norm(df["FVM"].fillna(0.0))
        + 0.40 * _rank_norm(df["ALGnum"].fillna(0.0))
    )
    df["_CurrentQuality"] = current_quality
    team_quality = df.groupby("Squadra")["_CurrentQuality"].mean()
    df["TeamContext"] = df["Squadra"].map(_rank_norm(team_quality)).fillna(0.5)
    df = df.drop(columns=["_CurrentQuality"])

    df["PredFM"] = float("nan")
    ranks = []
    for role in ROLE_ORDER:
        sub = df[df["Ruolo"] == role].copy()
        if sub.empty:
            continue
        fm_n = _rank_norm_neutral(sub["FMEst"])
        fvm_n = _rank_norm_neutral(sub["FVM"])
        for col in ["ALGnum", "InjuryP", "TagScore", "Starter", "SetPieces"]:
            sub[col] = sub[col].fillna(0.0)
        fac = ROLE_FACTORS.get(role, {"Starter": 1.0, "SetPieces": 1.0})
        sub["C_FM"] = weights["FM"] * fm_n
        sub["C_FVM"] = weights["FVM"] * fvm_n
        sub["C_ALG"] = weights["ALG"] * sub["ALGnum"]
        sub["C_Starter"] = weights["Starter"] * fac["Starter"] * sub["Starter"]
        sub["C_SetPieces"] = (
            weights["SetPieces"] * fac["SetPieces"] * sub["SetPieces"]
        )
        sub["C_Tags"] = weights["Tags"] * (sub["TagScore"] + 2.0) / 4.0
        sub["C_Injury"] = weights["Injury"] * sub["InjuryP"]
        sub["C_Availability"] = weights["Availability"] * sub["Availability"]
        # Continuous missing history is genuinely neutral in v2: it receives
        # the role median and uncertainty is handled separately by confidence.
        xgi = pd.to_numeric(sub.get("xGI90"), errors="coerce")
        xgi_n = _rank_norm_neutral(xgi)
        sub["C_ExpectedOutput"] = weights["ExpectedOutput"] * xgi_n
        sub["C_TeamContext"] = weights.get("TeamContext", 0.0) * sub["TeamContext"]
        sub["C_Confidence"] = 0.0
        # Storico stagionale (fantacalcio.it): gol subiti per presenza per i portieri,
        # media voto / presenze / produttività / rigori per tutti.
        sub["C_GolSubiti"] = 0.0
        ga_active = False
        if role == "P" and "GolSubiti90" in sub.columns:
            ga90 = pd.to_numeric(sub["GolSubiti90"], errors="coerce")
            ga_active = bool(ga90.notna().any())
            if ga_active:
                ga_rank = _rank_norm_neutral(ga90)
                ga_inv = 1.0 - ga_rank
                sub["C_GolSubiti"] = weights.get("GolSubiti", 0.0) * ga_inv
        hist_mv = pd.to_numeric(sub.get("HistMV"), errors="coerce")
        hist_apps = pd.to_numeric(sub.get("HistPresenze"), errors="coerce")
        hist_penalties = pd.to_numeric(sub.get("HistRigori"), errors="coerce")
        hist_output = (
            pd.to_numeric(sub.get("HistGol90"), errors="coerce")
            + pd.to_numeric(sub.get("HistAss90"), errors="coerce")
        )
        sub["C_MediaVoto"] = weights.get("MediaVoto", 0.0) * _rank_norm_neutral(hist_mv)
        sub["C_Presenze"] = weights.get("Presenze", 0.0) * _rank_norm_neutral(hist_apps)
        sub["C_Rigori"] = weights.get("Rigori", 0.0) * _rank_norm(
            hist_penalties.where(hist_penalties > 0)
        )
        sub["C_Produttivita"] = (
            weights.get("Produttivita", 0.0) * _rank_norm_neutral(hist_output)
        )
        manual = (
            sub["C_FM"] + sub["C_FVM"] + sub["C_ALG"]
            + sub["C_Starter"] + sub["C_SetPieces"]
            + sub["C_Tags"] + sub["C_Injury"] + sub["C_Availability"]
            + sub["C_ExpectedOutput"] + sub["C_GolSubiti"]
            + sub["C_MediaVoto"] + sub["C_Presenze"]
            + sub["C_Rigori"] + sub["C_Produttivita"]
            + sub["C_TeamContext"]
        )
        manual_scale = (
            abs(weights["FM"]) + abs(weights["FVM"]) + abs(weights["ALG"])
            + abs(weights["Starter"]) * fac["Starter"]
            + abs(weights["SetPieces"]) * fac["SetPieces"]
            + abs(weights["Tags"]) + abs(weights["Injury"])
            + abs(weights["Availability"])
            # A completely unavailable continuous source is still neutral,
            # not silently removed from the scale: its role median is 0.5.
            + abs(weights["ExpectedOutput"])
            + (abs(weights.get("GolSubiti", 0.0)) if ga_active else 0.0)
            + abs(weights.get("MediaVoto", 0.0))
            + abs(weights.get("Presenze", 0.0))
            + (abs(weights.get("Rigori", 0.0))
               if bool((hist_penalties > 0).any()) else 0.0)
            + abs(weights.get("Produttivita", 0.0))
            + abs(weights.get("TeamContext", 0.0))
        )
        manual_n = manual / manual_scale if manual_scale else manual * 0.0
        temporal_guard = _temporal_role_validation(sub)
        model_features = (
            ROLE_MODEL_FEATURES.get(role, ROLE_MODEL_FEATURES["C"])
            if temporal_guard["accepted"] is not False
            else LEGACY_MODEL_FEATURES
        )
        model = _role_model(sub, cols=model_features)
        if model is not None:
            sub["PredFM"] = _model_predict(sub, model)
        pred_n = _rank_norm(sub["PredFM"]) if model is not None \
            else pd.Series(0.0, index=sub.index)
        sub["C_Model"] = pred_n
        method = weights.get("_method", DEFAULT_METHOD)
        if method == "model":
            raw_quality = pred_n if model is not None else manual_n
        elif method == "blend":
            model_weight = ROLE_BLEND_MODEL_WEIGHT.get(role, 0.5)
            raw_quality = (
                model_weight * pred_n + (1.0 - model_weight) * manual_n
                if model is not None else manual_n
            )
        else:
            raw_quality = manual_n
        sub["RawQuality"] = pd.to_numeric(raw_quality, errors="coerce").fillna(0.5).clip(0.0, 1.0)
        shrink_strength = min(1.0, max(0.0, float(weights.get("Confidence", 0.65))))
        reliability = 1.0 - shrink_strength * (1.0 - sub["DataConfidence"])
        sub["RankingQuality"] = (
            0.5 + (sub["RawQuality"] - 0.5) * reliability
        ).clip(0.0, 1.0)
        sub["Score"] = sub["RankingQuality"]
        sub["SeasonValue"] = (
            0.80 * sub["RankingQuality"] + 0.20 * sub["Availability"]
        ).clip(0.0, 1.0)
        sub["QualityScore"] = sub["RankingQuality"]
        sub["Upside"] = (
            0.35 * sub["ALGnum"] + 0.25 * sub["SetPieces"]
            + 0.20 * ((sub["TagScore"] + 2.0) / 4.0) + 0.20 * xgi_n
        ).clip(0.0, 1.0)
        sub["BlendModelWeight"] = (
            ROLE_BLEND_MODEL_WEIGHT.get(role, 0.5) if model is not None else 0.0
        )
        sub["RankingVersion"] = RANKING_VERSION
        sub["TemporalModelAccepted"] = temporal_guard["accepted"] is not False
        sub = sub.sort_values(
            ["SeasonValue", "QA", "Nome"],
            ascending=[False, False, True],
            na_position="last",
        )
        sub = sub.reset_index(drop=True)
        sub["Rank"] = sub.index + 1
        sub["Cluster"] = (sub["Rank"] - 1) // CLUSTER_SIZE + 1
        ranks.append(sub)
    df = pd.concat(ranks, ignore_index=True)

    if progress_cb:
        progress_cb(f"{len(df)} players merged, "
                    f"{len(df[df['FM'].notna()])} with fantamedia")
    return df


def _ridge(X, y, alpha=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    coef, *_ = np.linalg.lstsq(A.T @ A + reg, A.T @ y, rcond=None)
    return coef


def _role_model(role_df, cols=None):
    if cols is None:
        roles = role_df["Ruolo"].dropna().astype(str).unique().tolist()
        role = roles[0] if len(roles) == 1 else ""
        cols = ROLE_MODEL_FEATURES.get(role, ROLE_MODEL_FEATURES["C"])
    known = role_df[role_df["FM1"].notna()]
    minimum = 20 if (known["Ruolo"] == "P").all() else 30
    if len(known) < minimum:
        return None
    # reindex mantiene leggibili anche cache/CSV creati prima dell'aggiunta di
    # una feature: quella informazione diventa semplicemente mancante.
    X = known.reindex(columns=cols).to_numpy(dtype=float)
    present = ~np.isnan(X)
    mu = np.zeros(X.shape[1])
    has_values = present.any(axis=0)
    mu[has_values] = np.nanmean(X[:, has_values], axis=0)
    sd = np.ones(X.shape[1])
    sd[has_values] = np.nanstd(X[:, has_values], axis=0)
    sd[(sd == 0) | np.isnan(sd)] = 1.0
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)
    coef = _ridge(Xs, known["FM1"].to_numpy(dtype=float))
    return (
        tuple(cols), mu, sd, coef,
        float(known["FM1"].min()), float(known["FM1"].max()),
    )


def _model_predict(role_df, model):
    cols, mu, sd, coef, lo, hi = model
    X = role_df.reindex(columns=cols).to_numpy(dtype=float)
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)
    pred = np.column_stack([np.ones(len(Xs)), Xs]) @ coef
    return np.clip(pred, lo, hi)


def _temporal_role_validation(role_df):
    """Compare the v2 historical signal with the previous one, forward only."""
    actual = []
    candidate = []
    baseline = []
    for _, row in role_df.iterrows():
        fm1 = _to_float(row.get("FM1"))
        fm2 = _to_float(row.get("FM2"))
        fm3 = _to_float(row.get("FM3"))
        if np.isfinite(fm2) and np.isfinite(fm3):
            actual.append(fm2)
            candidate.append(fm3)
            baseline.append(fm3)
        if np.isfinite(fm1) and (np.isfinite(fm2) or np.isfinite(fm3)):
            values = []
            if np.isfinite(fm2):
                values.append((fm2, 0.70))
            if np.isfinite(fm3):
                values.append((fm3, 0.30))
            actual.append(fm1)
            candidate.append(
                sum(value * weight for value, weight in values)
                / sum(weight for _, weight in values)
            )
            baseline.append(fm2 if np.isfinite(fm2) else fm3)
    n = len(actual)
    if n < 20:
        return {
            "spearman": None,
            "candidate_spearman": None,
            "baseline_spearman": None,
            "mae": None,
            "n": n,
            "accepted": None,
        }
    actual_s = pd.Series(actual, dtype=float)
    candidate_s = pd.Series(candidate, dtype=float)
    baseline_s = pd.Series(baseline, dtype=float)
    candidate_rho = candidate_s.rank().corr(actual_s.rank())
    baseline_rho = baseline_s.rank().corr(actual_s.rank())
    candidate_rho = float(candidate_rho) if pd.notna(candidate_rho) else None
    baseline_rho = float(baseline_rho) if pd.notna(baseline_rho) else None
    accepted = (
        candidate_rho is not None
        and (baseline_rho is None or candidate_rho >= baseline_rho - 0.02)
    )
    chosen = candidate_s if accepted else baseline_s
    return {
        "spearman": candidate_rho if accepted else baseline_rho,
        "candidate_spearman": candidate_rho,
        "baseline_spearman": baseline_rho,
        "mae": float((chosen - actual_s).abs().mean()),
        "n": n,
        "accepted": accepted,
    }


def ranking_diagnostics(players=None):
    """Honest time-forward diagnostics from older FM to the next season.

    Current-season FVM, ALG and lineup information are deliberately excluded:
    they did not exist at the time of the historical target.  Two temporal
    folds are used where available: FM3 -> FM2 and FM2/FM3 -> FM1.
    """
    if players is None:
        players = build_players()
    results = {}
    for role in ROLE_ORDER:
        sub = players[players["Ruolo"] == role]
        results[role] = _temporal_role_validation(sub)
    return results


def backtest_predictor(players=None, folds=5, seed=42):
    """Backward-compatible Spearman-only view of temporal diagnostics."""
    del folds, seed
    diagnostics = ranking_diagnostics(players)
    return {
        role: (
            float(values["spearman"])
            if values["spearman"] is not None else float("nan")
        )
        for role, values in diagnostics.items()
    }


def players_cache():
    if not hasattr(players_cache, "_df"):
        players_cache._df = build_players()
    return players_cache._df


def suggest_player(query, players_df, limit=8):
    q = normalize_name(query).split()
    if not q:
        return players_df.head(0)
    if query and len(query) >= 2:
        sub = players_df[
            players_df["Nome"].str.lower().str.contains(query.lower(), regex=False)
        ].copy()
        if not sub.empty:
            sub["_exact"] = sub["Nome"].apply(
                lambda n: 0 if query.lower() in normalize_name(n).split() else 1
            )
            sub = sub.sort_values(["_exact", "FM", "Nome"],
                                  ascending=[True, False, True],
                                  na_position="last").drop(columns="_exact")
            return sub.head(limit)
    scores = []
    for _, p in players_df.iterrows():
        tokens = normalize_name(p["Nome"]).split()
        common = len(set(q) & set(tokens))
        if common == 0:
            ratio = difflib.SequenceMatcher(
                None, normalize_name(query), normalize_name(p["Nome"])
            ).ratio()
            if ratio < 0.6:
                continue
            score = ratio
        else:
            score = common / max(len(set(q) | set(tokens)), 1)
            if q[-1] == tokens[-1]:
                score += 0.3
        scores.append((score, p))
    scores.sort(key=lambda x: x[0], reverse=True)
    if scores and scores[0][0] > 0.35:
        return pd.DataFrame([p for _, p in scores[:limit]])
    return players_df.head(0)


def _fuzzy_match(name, candidates):
    n = normalize_name(name)
    for c in candidates:
        if normalize_name(c) == n:
            return c
    best = None
    best_score = 0.0
    for c in candidates:
        ratio = difflib.SequenceMatcher(
            None, n, normalize_name(c)
        ).ratio()
        if ratio > best_score:
            best_score = ratio
            best = c
    return best if best_score >= 0.8 else None


def load_formazioni():
    if not FORMAZIONI.exists():
        return pd.DataFrame()
    df = pd.read_csv(FORMAZIONI)
    for col in [
        "Titolari", "RuoliTitolari", "Panchina", "RuoliPanchina",
        "Ballottaggi", "Squalificati", "Diffidati", "Infortunati", "InDubbio",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df


def load_set_pieces():
    if not SET_PIECES.exists():
        return pd.DataFrame()
    df = pd.read_csv(SET_PIECES)
    if "Ordine" not in df:
        # Compatibilità con i CSV già scaricati: l'ordine delle righe è quello
        # della gerarchia pubblicata dalla fonte.
        df["Ordine"] = df.groupby(["Squadra", "Tipo"]).cumcount() + 1
    return df


def load_panchinari():
    """Probabili panchinari per posizione (fonte sosfanta, stagionale)."""
    if not PANCHINARI.exists():
        return pd.DataFrame()
    return pd.read_csv(PANCHINARI)


def _apply_bench_panchinari(lineups, players_df, panch):
    """Add the 'possibile panchinaro' to every starter of the expected XI.

    The sosfanta source stores each spot as the favourite starter plus his
    alternatives (``Nome1|Nome2``). When a starter of the Fantacalcio lineup
    belongs to a spot with more than one player, the other player becomes his
    bench alternative.
    """
    slots_by_team = {}
    for _, r in panch.iterrows():
        team = normalize_name(str(r["Squadra"]))
        names = [n for n in str(r["Nomi"]).split("|") if n]
        slots_by_team.setdefault(team, []).append(names)

    for col in ["Panchina", "PanchinaRuolo", "PanchinaCluster"]:
        if col in lineups.columns:
            lineups[col] = lineups[col].astype(object)

    gaz_names = players_df["NomeGaz"].dropna().tolist()
    for index, row in lineups.iterrows():
        slots = slots_by_team.get(normalize_name(str(row["Squadra"])), [])
        if not slots:
            continue
        tokens = _tokens(row["NomeLineup"])
        bench_name = None
        for names in slots:
            if not any(
                _match_score(tokens, _tokens(n)) >= 0.9 for n in names
            ):
                continue
            for candidate in names:
                if not _match_score(tokens, _tokens(candidate)) >= 0.9:
                    bench_name = candidate
                    break
            break
        if not bench_name:
            continue
        lineups.at[index, "Panchina"] = bench_name
        matched = _fuzzy_match(bench_name, gaz_names)
        if matched:
            p = players_df[players_df["NomeGaz"] == matched].iloc[0]
            lineups.at[index, "PanchinaFM"] = p["FMEst"]
            lineups.at[index, "PanchinaFMImputed"] = bool(p["FMImputed"])
            lineups.at[index, "PanchinaRuolo"] = p["Ruolo"]
            lineups.at[index, "PanchinaCluster"] = p["Cluster"]
        else:
            lineups.at[index, "PanchinaFM"] = float("nan")
            lineups.at[index, "PanchinaFMImputed"] = False
            lineups.at[index, "PanchinaRuolo"] = row["Ruolo"] or ""
            lineups.at[index, "PanchinaCluster"] = ""
    return lineups


def build_lineups(players_df, progress_cb=None):
    form = load_formazioni()
    sp = load_set_pieces()
    if form.empty:
        return pd.DataFrame()

    for col in ["RuoliTitolari", "Panchina", "RuoliPanchina"]:
        if col not in form.columns:
            form[col] = ""
        form[col] = form[col].fillna("")

    sp_map = {}
    for _, r in sp.iterrows():
        key = (r["Squadra"], normalize_name(r["Giocatore"]))
        tipo = r["Tipo"]
        ordine = int(r.get("Ordine", 1))
        current = sp_map.setdefault(key, {}).get(tipo, ordine)
        sp_map[key][tipo] = min(current, ordine)

    guida = load_guida_data()
    guida_doubts = {}
    for doubt in guida.get("doubts", []):
        team = normalize_name(str(doubt.get("team", "")))
        first, second = str(doubt.get("first", "")), str(doubt.get("second", ""))
        if not team or not first or not second:
            continue
        likelihood = doubt.get("first_likelihood")
        suffix = f" ({likelihood}%)" if likelihood is not None else ""
        guida_doubts.setdefault(team, []).append(f"{first} / {second}{suffix}")
    gaz_names = players_df["NomeGaz"].dropna().tolist()
    rows = []
    for _, f in form.iterrows():
        team = f["Squadra"]
        modulo = f["Modulo"]
        starters = str(f["Titolari"]).split("|")
        s_roles = str(f["RuoliTitolari"]).split("|") if f["RuoliTitolari"] else []
        bench = str(f["Panchina"]).split("|") if f["Panchina"] else []
        b_roles = str(f["RuoliPanchina"]).split("|") if f["RuoliPanchina"] else []

        bench_info = []
        for i, bname in enumerate(bench):
            if not bname:
                continue
            role = b_roles[i] if i < len(b_roles) else ""
            matched = _fuzzy_match(bname, gaz_names)
            if matched:
                p = players_df[players_df["NomeGaz"] == matched].iloc[0]
                bench_info.append(
                    {
                        "nome": p["Nome"],
                        "ruolo": role or p["Ruolo"],
                        "fm": p["FMEst"],
                        "cluster": p["Cluster"],
                    }
                )
            else:
                bench_info.append({"nome": bname, "ruolo": role,
                                   "fm": float("nan"), "cluster": ""})

        for i, name in enumerate(starters):
            if not name:
                continue
            matched = _fuzzy_match(name, gaz_names)
            row = {
                "Squadra": team,
                "Modulo": modulo,
                "NomeLineup": name,
                "Rigorista": False,
                "Punizioni": False,
                "Angoli": False,
                "Piazzati": False,
                "RigoristaOrdine": 0,
                "PunizioniOrdine": 0,
                "AngoliOrdine": 0,
                "PiazzatiOrdine": 0,
                "Panchina": "",
                "PanchinaFM": float("nan"),
                "PanchinaRuolo": "",
                "PanchinaCluster": "",
                "Ballottaggi": f.get("Ballottaggi", "") or " · ".join(
                    guida_doubts.get(normalize_name(str(team)), [])
                ),
                "Squalificati": f.get("Squalificati", ""),
                "Diffidati": f.get("Diffidati", ""),
                "Infortunati": f.get("Infortunati", ""),
                "InDubbio": f.get("InDubbio", ""),
            }
            if matched:
                p = players_df[players_df["NomeGaz"] == matched].iloc[0]
                row["Nome"] = p["Nome"]
                row["Ruolo"] = p["Ruolo"]
                row["FM"] = p["FMEst"]
                row["FMImputed"] = bool(p["FMImputed"])
                row["QA"] = p["QA"]
                row["Cluster"] = p["Cluster"]
            else:
                row["Nome"] = name
                row["Ruolo"] = ""
                row["FM"] = float("nan")
                row["FMImputed"] = False
                row["QA"] = float("nan")
                row["Cluster"] = ""
            s_role = s_roles[i] if i < len(s_roles) else ""
            sub_role = s_role or row["Ruolo"]
            for b in bench_info:
                if b["ruolo"] and b["ruolo"] == sub_role:
                    row["Panchina"] = b["nome"]
                    row["PanchinaFM"] = b["fm"]
                    row["PanchinaRuolo"] = b["ruolo"]
                    row["PanchinaCluster"] = b["cluster"]
                    break
            flags = {}
            sp_names = sorted({g for (t, g), _ in sp_map.items() if t == team})
            if sp_names and row["Ruolo"] != "P":
                scored = sorted(
                    [(_short_source_name_score(name, candidate), candidate)
                     for candidate in sp_names],
                    reverse=True,
                )
                if scored and scored[0][0] >= 0.90:
                    runner_up = scored[1][0] if len(scored) > 1 else 0.0
                    if scored[0][0] - runner_up >= 0.04:
                        flags = sp_map.get((team, scored[0][1]), {})
            for tipo, ordine in flags.items():
                if tipo in ("Rigorista", "Piazzati", "Punizioni", "Angoli"):
                    row[tipo] = True
                    row[f"{tipo}Ordine"] = ordine
            rows.append(row)
    df = pd.DataFrame(rows)
    panch = load_panchinari()
    if not panch.empty:
        df = _apply_bench_panchinari(df, players_df, panch)
    if progress_cb:
        matched_n = df["Ruolo"].astype(str).str.len().gt(0).sum()
        bench_n = df["Panchina"].fillna("").astype(str).str.len().gt(0).sum()
        progress_cb(f"{len(df)} titolari ({matched_n} abbinati ai giocatori, "
                    f"{bench_n} con possibile panchinaro)")
    return df
