import difflib
import re
from pathlib import Path

import pandas as pd

from scraper import (
    DATA_DIR, FORMAZIONI, PLAYERS_FCP, QUOTAZIONI, SET_PIECES, normalize_name,
)

ROLE_ORDER = ["P", "D", "C", "A"]
CLUSTER_SIZE = 10

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


def build_players(progress_cb=None):
    gaz = load_quotazioni()
    fcp = load_fcp()

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
        matched_rows.append(row)

    df = pd.DataFrame(matched_rows)
    if df.empty:
        return df

    fm_cols = [c for c in ["FM1", "FM2", "FM3"] if c in df.columns]
    df["FM"] = df[fm_cols].bfill(axis=1).iloc[:, 0] if fm_cols else float("nan")
    df["FMAvg"] = df[fm_cols].mean(axis=1) if fm_cols else float("nan")

    ranks = []
    for role in ROLE_ORDER:
        sub = df[df["Ruolo"] == role].copy()
        sub = sub.sort_values(
            ["FM", "QA", "Nome"],
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
    df["Titolari"] = df["Titolari"].fillna("")
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

    sp_map = {}
    for _, r in sp.iterrows():
        key = (r["Squadra"], normalize_name(r["Giocatore"]))
        sp_map.setdefault(key, set()).add(r["Tipo"])
    gaz_names = players_df["NomeGaz"].dropna().tolist()
    rows = []
    for _, f in form.iterrows():
        team = f["Squadra"]
        modulo = f["Modulo"]
        for name in str(f["Titolari"]).split("|"):
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
            }
            if matched:
                p = players_df[players_df["NomeGaz"] == matched].iloc[0]
                row["Nome"] = p["Nome"]
                row["Ruolo"] = p["Ruolo"]
                row["FM"] = p["FM"]
                row["QA"] = p["QA"]
            else:
                row["Nome"] = name
                row["Ruolo"] = ""
                row["FM"] = float("nan")
                row["QA"] = float("nan")
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
