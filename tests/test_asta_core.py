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
        state = asta_core.roster_summary(self.session)
        self.assertEqual(state["max_next_bid"], 17)
        with self.assertRaises(ValueError):
            asta_core.record_purchase(self.session, player("Troppo caro", "P"), 18)
        asta_core.record_purchase(self.session, player("Portiere", "P"), 8)
        state = asta_core.roster_summary(self.session)
        self.assertEqual(state["remaining"], 12)
        self.assertEqual(state["slots_remaining"], 3)
        self.assertEqual(state["max_next_bid"], 10)

    def test_coach_uses_the_lower_of_value_and_live_budget_limit(self):
        advice = asta_core.coach(self.session, player("Attaccante", "A"))
        self.assertEqual(advice["fair_cap"], 15)
        self.assertEqual(advice["cap"], 15)
        asta_core.record_purchase(self.session, player("Portiere", "P"), 12)
        advice = asta_core.coach(self.session, player("Attaccante", "A"))
        self.assertEqual(advice["max_next_bid"], 6)
        self.assertEqual(advice["cap"], 6)

    def test_duplicate_and_full_role_are_rejected_and_can_be_undone(self):
        asta_core.record_purchase(self.session, player("Portiere", "P"), 5)
        with self.assertRaises(ValueError):
            asta_core.record_purchase(self.session, player("Portiere", "P"), 1)
        with self.assertRaises(ValueError):
            asta_core.record_purchase(self.session, player("Altro", "P"), 1)
        removed = asta_core.remove_purchase(self.session, 0)
        self.assertEqual(removed["name"], "Portiere")
        self.assertEqual(asta_core.roster_summary(self.session)["spent"], 0)

    def test_old_saved_session_is_upgraded(self):
        old = {"meta": {"budget": 100, "fair": {"P": [1], "D": [1],
                                                     "C": [1], "A": [1]}}}
        upgraded = asta_core.ensure_session(old)
        self.assertEqual(upgraded["purchases"], [])
        self.assertIn("slots", upgraded["meta"])


if __name__ == "__main__":
    unittest.main()
