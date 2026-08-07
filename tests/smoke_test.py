import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_asta import load_excluded, save_excluded
from data_loader import (
    DEFAULT_METHOD, DEFAULT_RANK_WEIGHTS, ROLE_ORDER, backtest_predictor,
    build_lineups, build_players, load_ranking_weights,
    save_ranking_weights, _set_piece_score,
)


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    players = build_players()
    check("players loaded", not players.empty, f"({len(players)} players)")
    check("cluster column", "Cluster" in players.columns)
    check("score column", "Score" in players.columns)
    check("roles present", {"P", "D", "C", "A"}.issubset(
        set(players["Ruolo"].dropna())))
    check("ALG normalized 0-1", players["ALGnum"].max() <= 1.01,
          f"(max {players['ALGnum'].max():.3f})")
    check("imputed FM", players["FMImputed"].sum() > 100,
          f"({int(players['FMImputed'].sum())} imputed)")
    check("predicted FM", players["PredFM"].notna().sum() > 100,
          f"({int(players['PredFM'].notna().sum())} predicted)")
    cluster_ok = True
    for role in ROLE_ORDER:
        sub = players[players["Ruolo"] == role]
        n = len(sub)
        if not n:
            continue
        k = math.ceil(n / 10)
        sizes = sub.groupby("Cluster").size()
        best_cluster = sub.loc[sub["Rank"].idxmin(), "Cluster"]
        cluster_ok = cluster_ok and \
            sorted(sub["Cluster"].unique()) == list(range(1, k + 1)) and \
            sizes.max() <= 10 and \
            best_cluster == 1
    check("clusters of 10", cluster_ok,
          f"(max {players.groupby(['Ruolo', 'Cluster']).size().max()} per cluster)")

    fm_only = {k: 0.0 for k in DEFAULT_RANK_WEIGHTS}
    fm_only["FM"] = 1.0
    fm_only["_method"] = "manual"
    players_fm = build_players(weights=fm_only)
    regress = True
    for role in ROLE_ORDER:
        sub = players_fm[players_fm["Ruolo"] == role]
        order = sub.sort_values(
            ["FMEst", "QA", "Nome"], ascending=[False, False, True],
            na_position="last")["Rank"].tolist()
        regress = regress and order == list(range(1, len(sub) + 1))
    check("FM-only weights reproduce FM ordering", regress)

    partial = players[players["FM1"].isna() & (
        players["FM2"].notna() | players["FM3"].notna()
    )]
    fm_ok = True
    for _, p in partial.iterrows():
        values = [(p[col], weight) for col, weight in (
            ("FM1", 0.6), ("FM2", 0.3), ("FM3", 0.1)
        ) if pd.notna(p[col])]
        expected = sum(value * weight for value, weight in values) / \
            sum(weight for _, weight in values)
        fm_ok = fm_ok and math.isclose(p["FM"], expected, rel_tol=1e-9)
    check("missing FM seasons renormalized", fm_ok,
          f"({len(partial)} partial histories)")
    check("set-piece types use full scale",
          math.isclose(_set_piece_score(
              ["Rigorista", "Punizioni", "Angoli"]), 1.0))
    check("blend score normalized", players["Score"].between(0.0, 1.0).all())

    rho = backtest_predictor(players)
    ok_roles = {r: v for r, v in rho.items() if pd.notna(v)}
    check("backtest model (Spearman > 0.2)", len(ok_roles) >= 3 and
          all(v > 0.2 for v in ok_roles.values()),
          f"{ {r: round(v, 2) for r, v in rho.items()} }")

    lineups = build_lineups(players)
    check("lineups built", not lineups.empty, f"({len(lineups)} starters)")
    check("substitute column", "Panchina" in lineups.columns)
    check("cluster columns", {"Cluster", "PanchinaCluster"}.issubset(
        lineups.columns))
    subs = lineups[lineups["Panchina"] != ""]
    check("substitutes matched", len(subs) > 0,
          f"({len(subs)}/{len(lineups)} with substitute)")

    save_excluded({"zz-smoke-test-player"})
    ex = load_excluded()
    check("exclusion round-trip", "zz-smoke-test-player" in ex)
    save_excluded(set())
    check("exclusion cleanup", len(load_excluded()) == 0)

    before = load_ranking_weights()
    save_ranking_weights(before)
    check("weights round-trip", load_ranking_weights() == before)
    check("default ranking method", before.get("_method") == DEFAULT_METHOD,
          f"({before.get('_method')})")
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app_asta.py"),
        default_timeout=60,
    )
    at.run()
    check("app runs without exceptions", len(at.exception) == 0,
          f"({len(at.exception)} exceptions)")
    tabs = [t.label for t in at.tabs]
    check("all tabs present",
          {"Setup", "Giocatori", "Formazioni"}.issubset(tabs), f"{tabs}")
    if "Formazioni" in tabs:
        form = [t for t in at.tabs if t.label == "Formazioni"][0]
        cards = sum(1 for m in form.markdown
                    if "Sostituto probabile" in m.value)
        check("substitute cards rendered", cards > 0, f"({cards} cards)")

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} problems")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
