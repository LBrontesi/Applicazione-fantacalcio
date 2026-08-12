import math
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from streamlit.testing.v1 import AppTest

import asta_core
import scraper
from app_asta import load_excluded, roster_alerts, save_excluded, source_freshness
from data_loader import (
    DEFAULT_METHOD, DEFAULT_RANK_WEIGHTS, ROLE_ORDER, backtest_predictor,
    build_lineups, build_players, load_ranking_weights, load_set_pieces,
    save_ranking_weights, _match_score, _set_piece_score, _short_source_name_score,
    _tokens,
)


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    players = build_players()
    default_session = asta_core.new_session()
    check("default participants", default_session["meta"]["teams"] == [
        "bro", "giorgio", "lolo", "pistacchio", "rochira", "piermattei",
        "babbo", "bubba", "riolo", "mattia",
    ] and default_session["meta"]["my_team"] == "bro")
    check("players loaded", not players.empty, f"({len(players)} players)")
    check("cluster column", "Cluster" in players.columns)
    check("score column", "Score" in players.columns)
    check("season profile columns", {
        "Availability", "SeasonValue", "Upside", "DataConfidence",
        "ConfidenceLabel", "HistorySeasons",
    }.issubset(players.columns))
    check("season profile ranges",
          players["Availability"].between(0, 1).all() and
          players["SeasonValue"].between(0, 1).all() and
          players["Upside"].between(0, 1).all() and
          players["DataConfidence"].between(0, 1).all())
    freshness = source_freshness()
    check("source freshness status", {
        "Quotazioni", "Giocatori/FCP", "Formazioni", "Tiratori",
    }.issubset({item["label"] for item in freshness}) and
          all(item["state"] in {"🟢", "🟠", "🔴"} for item in freshness))
    stack_sample = players.head(3)
    stacked_own = {
        "purchases": [
            {"name": row["Nome"], "club": "Inter", "role": row["Ruolo"], "price": 1}
            for _, row in stack_sample.iterrows()
        ],
        "spent": 3,
        "by_role": stack_sample["Ruolo"].value_counts().to_dict(),
    }
    check("same-club roster alert", any(
        "Troppi giocatori della stessa squadra" in alert
        for alert in roster_alerts(
            stacked_own, players, {"P": 3, "D": 8, "C": 8, "A": 6}
        )
    ))
    stats_backup = scraper.ADVANCED_STATS.read_bytes() \
        if scraper.ADVANCED_STATS.exists() else None
    try:
        imported = scraper.import_advanced_stats(StringIO(
            "Player,Squad,MP,Starts,Min,xG,xA,Gls,Ast\n"
            "Martinez Lautaro,Inter,35,32,2800,18.2,4.1,20,5\n"
        ))
        enriched = build_players()
        check("advanced stats import", len(imported) == 1 and
              int(enriched["Minuti"].notna().sum()) >= 1)
    finally:
        if stats_backup is None:
            scraper.ADVANCED_STATS.unlink(missing_ok=True)
        else:
            scraper.ADVANCED_STATS.write_bytes(stats_backup)
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
    check("set-piece hierarchy reduces backup value",
          _set_piece_score([("Rigorista", 2)]) <
          _set_piece_score([("Rigorista", 1)]))
    check("source aliases resolve initials and short labels",
          _match_score(_tokens("Ederson D.S."), _tokens("Ederson Dos Santos")) >= 0.90 and
          _short_source_name_score("Ederson D.S.", "Ederson") >= 0.90)
    ederson = players[players["NomeGaz"] == "Ederson D.S."]
    check("Ederson identity sources linked", len(ederson) == 1 and
          ederson.iloc[0]["FM1"] > 6.0 and
          ederson.iloc[0]["SetPieces"] == 0.0 and
          ederson.iloc[0]["DataConfidence"] >= 0.80,
          ederson[["Rank", "Cluster"]].to_dict("records"))
    atalanta_set_pieces = load_set_pieces().query("Squadra == 'Atalanta'")
    check("Fantacalcio Atalanta hierarchy", [
        (row["Giocatore"], row["Tipo"], int(row["Ordine"]))
        for _, row in atalanta_set_pieces.iterrows()
    ] == [
        ("Scamacca", "Rigorista", 1), ("Krstovic", "Rigorista", 2),
        ("Samardzic", "Rigorista", 3), ("De Ketelaere", "Piazzati", 1),
        ("Samardzic", "Piazzati", 2), ("Gaetano", "Piazzati", 3),
    ])
    check("blend score normalized", players["Score"].between(0.0, 1.0).all())

    # Neutral-missing ranking: players who never took a penalty or have no
    # history contribute zero instead of a percentile-tie boost, and a fully
    # missing xGI90 source stays at zero rather than a constant 0.5 tie.
    no_pen = players["HistRigori"].fillna(0.0) <= 0
    check("penalty-less players get zero C_Rigori",
          players.loc[no_pen, "C_Rigori"].abs().max() < 1e-9)
    check("missing xGI90 source is neutral",
          players["C_ExpectedOutput"].abs().max() < 1e-9)

    rho = backtest_predictor(players)
    ok_roles = {r: v for r, v in rho.items() if pd.notna(v)}
    check("backtest model (Spearman > 0.2)", len(ok_roles) >= 3 and
          all(v > 0.2 for v in ok_roles.values()),
          f"{ {r: round(v, 2) for r, v in rho.items()} }")

    lineups = build_lineups(players)
    check("lineups built", not lineups.empty, f"({len(lineups)} starters)")
    check("substitute column", "Panchina" in lineups.columns)
    scamacca_lineup = lineups[lineups["Nome"].astype(str).str.contains("Scamacca")]
    check("lineup hierarchy rendered", not scamacca_lineup.empty and
          bool(scamacca_lineup.iloc[0]["Rigorista"]) and
          int(scamacca_lineup.iloc[0]["RigoristaOrdine"]) == 1)
    check("cluster columns", {"Cluster", "PanchinaCluster"}.issubset(
        lineups.columns))
    check("availability columns", {
        "Ballottaggi", "Squalificati", "Infortunati", "InDubbio",
    }.issubset(lineups.columns))
    subs = lineups[lineups["Panchina"] != ""]
    bench_source = lineups["Panchina"].fillna("").astype(str).str.strip().ne("").any()
    check("substitutes matched", len(subs) > 0 or not bench_source,
          f"({len(subs)}/{len(lineups)} with substitute; source={bench_source})")

    excluded_before = load_excluded()
    save_excluded(excluded_before | {"zz-smoke-test-player"})
    ex = load_excluded()
    check("exclusion round-trip", "zz-smoke-test-player" in ex)
    save_excluded(excluded_before)
    check("exclusion cleanup", load_excluded() == excluded_before)

    auction = asta_core.new_session(
        budget=20,
        fair={role: [20] for role in ROLE_ORDER},
        slots={"P": 1, "D": 1, "C": 1, "A": 1},
        teams=[f"Test {i}" for i in range(1, 11)],
        created="2000-01-01T00:00:00",
    )
    sample = players.iloc[0]
    advice = asta_core.auction_advice(
        auction, sample, players.to_dict("records"), current_price=1
    )
    check("dynamic cap reserves remaining slots",
          advice["fixed_cap"] == 20 and advice["personal_max"] == 17 and
          advice["recommended"] <= advice["personal_max"] and
          advice["verdict"] == "PUNTA")
    asta_core.save_watchlist_item(auction, sample, "A", "obiettivo test")
    check("watchlist saved", auction["watchlist"][0]["tier"] == "A")
    advice_a = asta_core.auction_advice(auction, sample, players.to_dict("records"))
    asta_core.save_watchlist_item(auction, sample, "C", "occasione test")
    advice_c = asta_core.auction_advice(auction, sample, players.to_dict("records"))
    check("watchlist adjusts recommendation",
          advice_a["recommended"] >= advice["recommended"] and
          advice_c["recommended"] <= advice_a["recommended"])
    asta_core.remove_watchlist_item(auction, sample["Nome"])
    check("watchlist removed", not auction["watchlist"])
    purchase = asta_core.record_purchase(auction, sample, "Test 1", 7)
    live = asta_core.team_summary(auction, "Test 1")
    check("live auction records purchase", purchase["name"] == sample["Nome"] and
          live["remaining"] == 13 and live["by_role"][sample["Ruolo"]] == 1)
    try:
        asta_core.record_purchase(auction, sample, "Test 2", 1)
        duplicate_blocked = False
    except ValueError:
        duplicate_blocked = True
    check("live auction blocks duplicate", duplicate_blocked)
    asta_core.undo_purchase(auction, sample["Nome"])
    check("live auction undo", not auction["purchases"] and
          asta_core.team_summary(auction, "Test 1")["remaining"] == 20)

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
          {"Setup", "Asta live · giocatori", "Formazioni"}.issubset(tabs), f"{tabs}")
    check("one-click auction refresh available",
          any(button.label == "🚀 Aggiorna tutto per l'asta (~10 min)"
              for button in at.button))
    check("legacy refresh buttons removed",
          not any(button.label in {
              "Scarica quotazioni Gazzetta (veloce)",
              "Scarica lista giocatori FCP (lento, ~10 min)",
              "Scarica formazioni + tiratori (~30s)",
              "🔄 Aggiorna ora formazioni",
          } for button in at.button))
    at.session_state["session"] = asta_core.new_session(
        created="2000-01-01T00:00:00"
    )
    at.run()
    check("personal auction view runs", len(at.exception) == 0,
          f"({len(at.exception)} exceptions)")
    called_search = next(widget for widget in at.text_input
                         if widget.label == "Cerca giocatore chiamato")
    called_search.set_value(str(players.iloc[0]["Nome"])).run()
    check("called-player advice rendered", len(at.exception) == 0 and
          any(metric.label == "Punta fino a" for metric in at.metric),
          f"({len(at.exception)} exceptions)")
    if "Formazioni" in tabs:
        form = [t for t in at.tabs if t.label == "Formazioni"][0]
        cards = sum(1 for c in form.caption
                    if "Panchina / coperture" in c.value)
        check("substitute coverage rendered", cards > 0 or not bench_source,
              f"({cards} cards; source={bench_source})")

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} problems")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
