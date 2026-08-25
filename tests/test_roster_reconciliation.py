import unittest

import pandas as pd

from data_loader import _resolve_fcp_player


class RosterReconciliationTests(unittest.TestCase):
    def test_strong_cross_club_match_corrects_stale_quote_team(self):
        fcp = pd.DataFrame([
            {"NomeFCP": "Frattesi Davide", "Ruolo": "C", "SquadraFCP": "Lazio"},
            {"NomeFCP": "Politano Matteo", "Ruolo": "C", "SquadraFCP": "Napoli"},
        ])
        resolved = _resolve_fcp_player(
            {"NomeGaz": "Frattesi", "Ruolo": "C", "SquadraNome": "Inter"},
            fcp,
        )
        self.assertEqual(resolved["status"], "club_corretto_fcp")
        self.assertEqual(resolved["team"], "Lazio")

    def test_weak_or_ambiguous_match_is_not_guessed(self):
        fcp = pd.DataFrame([
            {"NomeFCP": "Rossi Marco", "Ruolo": "A", "SquadraFCP": "Roma"},
            {"NomeFCP": "Rossi Mario", "Ruolo": "A", "SquadraFCP": "Lazio"},
        ])
        resolved = _resolve_fcp_player(
            {"NomeGaz": "Rossi", "Ruolo": "A", "SquadraNome": "Inter"},
            fcp,
        )
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
