import unittest

import asta_core


def player(name, role, cluster=1):
    return {
        "Nome": name,
        "Squadra": "Test FC",
        "Ruolo": role,
        "Cluster": cluster,
        "FM": 7.1,
    }


class AuctionSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = asta_core.new_session(
            budget=20,
            fair={"P": [12], "D": [10], "C": [9], "A": [15]},
            slots={"P": 1, "D": 1, "C": 1, "A": 1},
            created="2026-08-10T12:00:00",
        )

    def test_budget_reserves_one_credit_for_each_other_slot(self):
        advice = asta_core.auction_advice(
            self.session, player("Troppo caro", "P"), [], current_price=18
        )
        self.assertEqual(advice["personal_max"], 12)
        self.assertEqual(advice["verdict"], "LASCIA")
        asta_core.record_purchase(self.session, player("Portiere", "P"), "bro", 8)
        self.assertEqual(asta_core.team_summary(self.session, "bro")["remaining"], 12)

    def test_coach_uses_the_lower_of_value_and_live_budget_limit(self):
        advice = asta_core.coach(self.session, player("Attaccante", "A"))
        self.assertEqual(advice["cap"], 15)
        asta_core.record_purchase(self.session, player("Portiere", "P"), "bro", 12)
        advice = asta_core.auction_advice(
            self.session, player("Attaccante", "A"), []
        )
        self.assertLessEqual(advice["personal_max"], 8)

    def test_duplicate_and_full_role_are_rejected_and_can_be_undone(self):
        asta_core.record_purchase(self.session, player("Portiere", "P"), "bro", 5)
        with self.assertRaises(ValueError):
            asta_core.record_purchase(self.session, player("Portiere", "P"), "bro", 1)
        with self.assertRaises(ValueError):
            asta_core.record_purchase(self.session, player("Altro", "P"), "bro", 1)
        removed = asta_core.undo_purchase(self.session, "Portiere")
        self.assertEqual(removed["name"], "Portiere")
        self.assertEqual(asta_core.team_summary(self.session, "bro")["spent"], 0)

    def test_opportunity_stays_in_unit_range(self):
        advice = asta_core.auction_advice(
            self.session,
            {**player("Top", "A"), "SeasonValue": 0.99, "Score": 0.9, "DataConfidence": 0.9},
            [{"Nome": "Debole", "Squadra": "Test FC", "Ruolo": "A",
              "Cluster": 1, "Score": 0.9, "SeasonValue": 0.05}],
            current_price=0,
        )
        self.assertGreaterEqual(advice["opportunity"], 0.0)
        self.assertLessEqual(advice["opportunity"], 1.0)

    def test_old_saved_session_is_upgraded(self):
        old = {"meta": {"budget": 100, "fair": {"P": [1], "D": [1],
                                                     "C": [1], "A": [1]}}}
        upgraded = asta_core.ensure_session(old)
        self.assertEqual(upgraded["purchases"], [])
        self.assertIn("slots", upgraded["meta"])


if __name__ == "__main__":
    unittest.main()
