import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_asta import load_excluded, save_excluded
from data_loader import build_lineups, build_players


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    players = build_players()
    check("players loaded", not players.empty, f"({len(players)} players)")
    check("cluster column", "Cluster" in players.columns)
    check("roles present", {"P", "D", "C", "A"}.issubset(
        set(players["Ruolo"].dropna())))

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
