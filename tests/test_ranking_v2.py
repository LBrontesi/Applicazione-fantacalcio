import unittest

import pandas as pd

from data_loader import (
    _match_score,
    _rank_norm_neutral,
    _tokens,
    build_lineups,
    build_players,
    ranking_diagnostics,
)


class RankingV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.players = build_players()
        cls.lineups = build_lineups(cls.players)

    def test_single_letter_initial_cannot_match_unrelated_player(self):
        self.assertEqual(
            _match_score(_tokens("Svilar"), _tokens("Castro S.")),
            0.0,
        )

    def test_goalkeepers_never_receive_attacking_set_pieces(self):
        keeper_players = self.players[
            (self.players["Ruolo"] == "P") & (self.players["SetPieces"] > 0)
        ]
        keeper_lineups = self.lineups[
            (self.lineups["Ruolo"] == "P")
            & self.lineups[["Rigorista", "Punizioni", "Angoli", "Piazzati"]].any(axis=1)
        ]
        self.assertTrue(keeper_players.empty)
        self.assertTrue(keeper_lineups.empty)
        regressions = self.players[
            self.players["Nome"].isin(["Svilar Mile", "Martinez Jo."])
        ]
        self.assertEqual(set(regressions["Nome"]), {"Svilar Mile", "Martinez Jo."})
        self.assertTrue((regressions["SetPieces"] == 0.0).all())

    def test_lautaro_keeps_inter_penalty_hierarchy(self):
        lautaro = self.players[self.players["Nome"] == "Martinez Lautaro"]
        self.assertEqual(len(lautaro), 1)
        self.assertGreater(float(lautaro.iloc[0]["SetPieces"]), 0.0)
        lineup = self.lineups[self.lineups["Nome"] == "Martinez Lautaro"]
        self.assertEqual(len(lineup), 1)
        self.assertTrue(bool(lineup.iloc[0]["Rigorista"]))
        self.assertEqual(int(lineup.iloc[0]["RigoristaOrdine"]), 3)

    def test_missing_continuous_data_is_role_neutral(self):
        values = pd.Series([1.0, 3.0, float("nan")])
        ranked = _rank_norm_neutral(values)
        self.assertEqual(float(ranked.iloc[2]), 0.5)

    def test_confidence_shrinks_extreme_quality_and_clusters_stay_fixed(self):
        distance_raw = (self.players["RawQuality"] - 0.5).abs()
        distance_ranked = (self.players["RankingQuality"] - 0.5).abs()
        self.assertTrue((distance_ranked <= distance_raw + 1e-12).all())
        self.assertTrue(self.players["RawQuality"].between(0.0, 1.0).all())
        self.assertTrue(self.players["RankingQuality"].between(0.0, 1.0).all())
        self.assertTrue(self.players["SeasonValue"].between(0.0, 1.0).all())
        self.assertEqual(
            int(self.players.groupby(["Ruolo", "Cluster"]).size().max()),
            10,
        )
        for role in ("P", "D", "C", "A"):
            ranked = self.players[self.players["Ruolo"] == role].sort_values("Rank")
            expected = ranked.sort_values(
                ["SeasonValue", "QA", "Nome"],
                ascending=[False, False, True],
                na_position="last",
            )
            self.assertEqual(ranked["Nome"].tolist(), expected["Nome"].tolist())

    def test_temporal_diagnostics_are_forward_and_informative(self):
        diagnostics = ranking_diagnostics(self.players)
        observed = [
            values["spearman"]
            for values in diagnostics.values()
            if values["spearman"] is not None
        ]
        self.assertGreaterEqual(len(observed), 3)
        self.assertGreater(sum(observed) / len(observed), 0.20)
        self.assertTrue(all(values["n"] >= 0 for values in diagnostics.values()))
        for values in diagnostics.values():
            if values["spearman"] is None:
                continue
            self.assertGreaterEqual(
                values["spearman"], values["baseline_spearman"] - 0.02
            )
            if values["accepted"] is False:
                self.assertEqual(values["spearman"], values["baseline_spearman"])


if __name__ == "__main__":
    unittest.main()
