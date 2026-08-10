import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scraper import (
    ADVANCED_STATS, DATA_DIR, FORMAZIONI, PLAYERS_FCP, QUOTAZIONI, SET_PIECES,
    normalize_name,
)

ROLE_ORDER = ["P", "D", "C", "A"]
CLUSTER_SIZE = 10

RANK_WEIGHTS_FILE = DATA_DIR / "ranking_weights.json"

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
}

DEFAULT_METHOD = "blend"

SP_TYPE_WEIGHTS = {"Rigorista": 1.0, "Punizioni": 0.6, "Angoli": 0.3}
SP_MAX_SCORE = sum(SP_TYPE_WEIGHTS.values())

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

MODEL_FEATURES = [
    "FM2", "FM3", "ALGnum", "FVM", "TagScore", "Starter", "SetPieces",
    "InjuryP", "Availability", "MinutesPct", "StartsPct", "xG90", "xA90",
]

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
        return {**DEFAULT_RANK_WEIGHTS, "_method": DEFAULT_METHOD}
    try:
        with open(RANK_WEIGHTS_FILE) as fh:
            saved = json.load(fh)
        out = {k: float(saved.get(k, DEFAULT_RANK_WEIGHTS[k]))
               for k in DEFAULT_RANK_WEIGHTS}
        out["_method"] = saved.get("_method", DEFAULT_METHOD)
        return out
    except (OSError, ValueError, KeyError):
        return {**DEFAULT_RANK_WEIGHTS, "_method": DEFAULT_METHOD}


def save_ranking_weights(weights):
    DATA_DIR.mkdir(exist_ok=True)
    data = {k: float(weights.get(k, DEFAULT_RANK_WEIGHTS[k]))
            for k in DEFAULT_RANK_WEIGHTS}
    data["_method"] = weights.get("_method", DEFAULT_METHOD)
    with open(RANK_WEIGHTS_FILE, "w") as fh:
        json.dump(data, fh, indent=2)

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


def _tag_score(attr):
    if not isinstance(attr, str) or not attr:
        return 0.0
    total = 0.0
    for tag in attr.split("|"):
        total += TAG_SCORES.get(tag.strip(), 0.0)
    return max(-2.0, min(2.0, total))


def _set_piece_score(types):
    weighted = sum(SP_TYPE_WEIGHTS.get(t, 0.0) for t in set(types))
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
    return pd.read_csv(QUOTAZIONI)


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


def _tokens(name):
    return normalize_name(name).split()


def _gaz_parts(tokens):
    if len(tokens) > 1 and len(tokens[-1]) == 1:
        return tokens[:-1], tokens[-1]
    return tokens, None


def _fcp_parts(tokens):
    return tokens[:-1], tokens[-1]


def _match_score(g_tokens, f_tokens):
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
    return 0.85 if ratio >= 0.86 else 0.0


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


def build_players(progress_cb=None, weights=None):
    gaz = load_quotazioni()
    fcp = load_fcp()
    advanced = load_advanced_stats()

    gaz_tokens = {name: _tokens(name) for name in gaz["NomeGaz"]}
    fcp_tokens = {name: _tokens(name) for name in fcp["NomeFCP"]}

    matched_fcp = {}
    for fcp_name, f_tokens in fcp_tokens.items():
        best = (None, 0.0)
        for gaz_name, g_tokens in gaz_tokens.items():
            score = _match_score(g_tokens, f_tokens)
            if score > best[1]:
                best = (gaz_name, score)
        if best[1] >= 0.85:
            matched_fcp[best[0]] = fcp_name

    matched_rows = []
    for _, g in gaz.iterrows():
        fcp_name = matched_fcp.get(g["NomeGaz"])
        f = fcp[fcp["NomeFCP"] == fcp_name].iloc[0] if fcp_name else None
        row = {
            "Nome": f["NomeFCP"] if f is not None and f["NomeFCP"] else g["NomeGaz"],
            "NomeGaz": g["NomeGaz"],
            "Squadra": g["SquadraNome"] or g["SquadraGaz"],
            "Ruolo": g["Ruolo"],
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

    sp_types = {}
    sp = load_set_pieces()
    if not sp.empty:
        for _, r in sp.iterrows():
            k = (normalize_name(str(r["Squadra"])),
                 normalize_name(str(r["Giocatore"])))
            sp_types.setdefault(k, set()).add(str(r["Tipo"]))

    def _sp_score(key):
        return _set_piece_score(sp_types.get(key, ()))

    df["SetPieces"] = df.apply(
        lambda r: _sp_score(
            (normalize_name(str(r["Squadra"])),
             normalize_name(str(r["NomeGaz"])))
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
        0.55 * df["HistoryStrength"] + 0.20 * source_coverage
        + 0.10 * (~df["FMImputed"]).astype(float) + 0.15 * advanced_coverage
    ).clip(0.0, 1.0)
    df["ConfidenceLabel"] = df["DataConfidence"].apply(_confidence_label)

    df["PredFM"] = float("nan")
    ranks = []
    for role in ROLE_ORDER:
        sub = df[df["Ruolo"] == role].copy()
        if sub.empty:
            continue
        fm_n = _rank_norm(sub["FMEst"])
        fvm_n = _rank_norm(sub["FVM"])
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
        xgi_n = _rank_norm(sub["xGI90"].fillna(0))
        sub["C_ExpectedOutput"] = weights["ExpectedOutput"] * xgi_n
        manual = (
            sub["C_FM"] + sub["C_FVM"] + sub["C_ALG"]
            + sub["C_Starter"] + sub["C_SetPieces"]
            + sub["C_Tags"] + sub["C_Injury"] + sub["C_Availability"]
            + sub["C_ExpectedOutput"]
        )
        manual_scale = (
            abs(weights["FM"]) + abs(weights["FVM"]) + abs(weights["ALG"])
            + abs(weights["Starter"]) * fac["Starter"]
            + abs(weights["SetPieces"]) * fac["SetPieces"]
            + abs(weights["Tags"]) + abs(weights["Injury"])
            + abs(weights["Availability"])
            + (abs(weights["ExpectedOutput"])
               if sub["xGI90"].notna().any() else 0.0)
        )
        manual_n = manual / manual_scale if manual_scale else manual * 0.0
        model = _role_model(sub)
        if model is not None:
            sub["PredFM"] = _model_predict(sub, model)
        pred_n = _rank_norm(sub["PredFM"]) if model is not None \
            else pd.Series(0.0, index=sub.index)
        sub["C_Model"] = pred_n
        method = weights.get("_method", DEFAULT_METHOD)
        if method == "model":
            sub["Score"] = pred_n if model is not None else manual_n
        elif method == "blend":
            model_weight = ROLE_BLEND_MODEL_WEIGHT.get(role, 0.5)
            sub["Score"] = (
                model_weight * pred_n + (1.0 - model_weight) * manual_n
                if model is not None else manual_n
            )
        else:
            sub["Score"] = manual_n
        quality = pred_n if model is not None else fm_n
        sub["SeasonValue"] = (0.75 * quality + 0.25 * sub["Availability"])
        sub["Upside"] = (
            0.35 * sub["ALGnum"] + 0.25 * sub["SetPieces"]
            + 0.20 * ((sub["TagScore"] + 2.0) / 4.0) + 0.20 * xgi_n
        ).clip(0.0, 1.0)
        sub["BlendModelWeight"] = (
            ROLE_BLEND_MODEL_WEIGHT.get(role, 0.5) if model is not None else 0.0
        )
        sub = sub.sort_values(
            ["Score", "QA", "Nome"],
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


def _role_model(role_df, cols=MODEL_FEATURES):
    known = role_df[role_df["FM1"].notna()]
    if len(known) < 30:
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
    return mu, sd, coef, float(known["FM1"].min()), float(known["FM1"].max())


def _model_predict(role_df, model, cols=MODEL_FEATURES):
    mu, sd, coef, lo, hi = model
    X = role_df.reindex(columns=cols).to_numpy(dtype=float)
    Xs = np.nan_to_num((X - mu) / sd, nan=0.0)
    pred = np.column_stack([np.ones(len(Xs)), Xs]) @ coef
    return np.clip(pred, lo, hi)


def backtest_predictor(players=None, folds=5, seed=42):
    if players is None:
        players = build_players()
    results = {}
    for role in ROLE_ORDER:
        sub = players[players["Ruolo"] == role].dropna(subset=["FM1"])
        if len(sub) < 40:
            results[role] = float("nan")
            continue
        idx = np.arange(len(sub))
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        fold_idx = np.array_split(idx, folds)
        pred = np.empty(len(sub))
        for fold in fold_idx:
            tr = np.setdiff1d(idx, fold)
            model = _role_model(sub.iloc[tr])
            if model is None:
                pred[fold] = np.nan
            else:
                pred[fold] = _model_predict(sub.iloc[fold], model)
        ok = ~np.isnan(pred)
        if ok.sum() < 20:
            results[role] = float("nan")
            continue
        actual = sub["FM1"].to_numpy(dtype=float)[ok]
        results[role] = float(
            pd.Series(pred[ok]).rank().corr(pd.Series(actual).rank())
        )
    return results


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
    return pd.read_csv(SET_PIECES)


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
        sp_map.setdefault(key, set()).add(r["Tipo"])
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
                "Panchina": "",
                "PanchinaFM": float("nan"),
                "PanchinaRuolo": "",
                "PanchinaCluster": "",
                "Ballottaggi": f.get("Ballottaggi", ""),
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
            flags = set()
            sp_names = [g for (t, g), _ in sp_map.items() if t == team]
            if sp_names:
                hit = _fuzzy_match(name, sp_names)
                if hit:
                    flags = sp_map.get((team, hit), set())
            for tipo in flags:
                if tipo in ("Rigorista", "Punizioni", "Angoli"):
                    row[tipo] = True
            rows.append(row)
    df = pd.DataFrame(rows)
    if progress_cb:
        matched_n = df["Ruolo"].astype(str).str.len().gt(0).sum()
        progress_cb(f"{len(df)} titolari ({matched_n} abbinati ai giocatori)")
    return df
