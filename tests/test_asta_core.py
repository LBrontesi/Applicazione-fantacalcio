import unittest
from copy import deepcopy
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import asta_core
import data_loader


def player(name, role, cluster=1):
    return {
        "Nome": name,
        "Squadra": "Test FC",
        "Ruolo": role,
        "Cluster": cluster,
        "FM": 7.1,
        "Rank": 1,
        "Score": 0.8,
        "SeasonValue": 0.8,
        "DataConfidence": 0.8,
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

    def test_session_save_is_atomic_and_keeps_previous_backup(self):
        with TemporaryDirectory() as directory, patch.object(
            asta_core, "SESSION_DIR", Path(directory)
        ):
            session = asta_core.new_session(
                budget=20,
                fair={"P": [12], "D": [10], "C": [9], "A": [15]},
                slots={"P": 1, "D": 1, "C": 1, "A": 1},
                created="2026-08-10T12:10:00",
            )
            path = asta_core.save_session(session)
            asta_core.record_purchase(session, player("Salvato", "P"), "bro", 2)
            asta_core.save_session(session)
            backup = path.with_suffix(f"{path.suffix}.bak")
            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(backup.read_text())["purchases"], [])
            self.assertEqual(len(json.loads(path.read_text())["purchases"]), 1)

    def test_invalid_fair_values_are_rejected_or_repaired(self):
        with self.assertRaises(ValueError):
            asta_core.new_session(fair={"P": [], "D": [1], "C": [1], "A": [1]})
        with self.assertRaises(ValueError):
            asta_core.new_session(fair={"P": [0], "D": [1], "C": [1], "A": [1]})
        corrupted = {"meta": {"budget": 100, "fair": {"P": [], "D": [], "C": [], "A": []}}}
        self.assertTrue(all(asta_core.ensure_session(corrupted)["meta"]["fair"].values()))

    def test_invalid_ranking_weights_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 2.1):
            with self.assertRaises(ValueError):
                data_loader.validate_ranking_weights({"FM": value})

    def test_v2_advice_is_deterministic_and_never_crosses_stop(self):
        candidate = player("Top", "A")
        start = perf_counter()
        first = asta_core.auction_advice(
            self.session, candidate, [candidate], current_price=4
        )
        second = asta_core.auction_advice(
            self.session, candidate, [candidate], current_price=4
        )
        elapsed = perf_counter() - start
        self.assertEqual(first, second)
        self.assertEqual(first["advice_version"], "2.0")
        self.assertLessEqual(first["recommended"], first["personal_max"])
        self.assertLessEqual(first["personal_max"], first["fixed_cap"])
        self.assertGreaterEqual(first["reserve"], first["total_left"] - 1)
        self.assertLess(elapsed, 0.2)

    def test_random_sessions_always_preserve_other_slot_credits(self):
        rng = Random(20260813)
        for iteration in range(100):
            slots = {role: rng.randint(1, 3) for role in data_loader.ROLE_ORDER}
            fair = {
                role: sorted(
                    [rng.randint(2, 40) for _ in range(count)], reverse=True
                )
                for role, count in slots.items()
            }
            session = asta_core.new_session(
                budget=rng.randint(60, 160),
                fair=fair,
                slots=slots,
                created=f"2026-08-11T12:{iteration % 60:02d}:00",
            )
            for role in data_loader.ROLE_ORDER:
                bought = rng.randrange(slots[role])
                for index in range(bought):
                    asta_core.record_purchase(
                        session,
                        player(f"Preso {iteration}-{role}-{index}", role, index + 1),
                        "bro",
                        1,
                    )
            called_role = rng.choice(data_loader.ROLE_ORDER)
            if asta_core.team_summary(session, "bro")["by_role"][called_role] >= slots[called_role]:
                continue
            candidate = player(f"Chiamato {iteration}", called_role, 1)
            advice = asta_core.auction_advice(session, candidate, [candidate])
            remaining = asta_core.team_summary(session, "bro")["remaining"]
            other_slots = advice["total_left"] - 1
            self.assertLessEqual(advice["recommended"], advice["personal_max"])
            self.assertLessEqual(advice["personal_max"], advice["fixed_cap"])
            self.assertGreaterEqual(remaining - advice["personal_max"], other_slots)

    def test_personal_plan_changes_real_budget_envelope(self):
        low = deepcopy(self.session)
        high = deepcopy(self.session)
        low["meta"]["role_priorities"]["A"] = 0.8
        high["meta"]["role_priorities"]["A"] = 1.2
        candidate = player("Attaccante", "A")
        low_advice = asta_core.auction_advice(low, candidate, [candidate])
        high_advice = asta_core.auction_advice(high, candidate, [candidate])
        self.assertGreater(high_advice["slot_envelope"], low_advice["slot_envelope"])
        self.assertGreaterEqual(high_advice["recommended"], low_advice["recommended"])

    def test_watchlist_risk_and_third_club_player_are_monotonic(self):
        base = asta_core.new_session(
            budget=200,
            fair={"P": [20], "D": [20, 20], "C": [20], "A": [100]},
            slots={"P": 1, "D": 2, "C": 1, "A": 1},
            created="2026-08-10T12:30:00",
        )
        candidate = {**player("Obiettivo", "A"), "RankingQuality": 0.8}
        watch_results = {}
        for tier in ("A", "B", "C"):
            session = deepcopy(base)
            asta_core.save_watchlist_item(session, candidate, tier)
            watch_results[tier] = asta_core.auction_advice(
                session, candidate, [candidate]
            )
        self.assertGreater(
            watch_results["A"]["watch_factor"], watch_results["B"]["watch_factor"]
        )
        self.assertGreater(
            watch_results["B"]["watch_factor"], watch_results["C"]["watch_factor"]
        )
        self.assertGreaterEqual(
            watch_results["A"]["recommended"], watch_results["B"]["recommended"]
        )
        self.assertGreaterEqual(
            watch_results["B"]["recommended"], watch_results["C"]["recommended"]
        )

        low_risk = asta_core.auction_advice(
            base, {**candidate, "DataConfidence": 0.1}, [candidate]
        )
        high_risk = asta_core.auction_advice(
            base, {**candidate, "DataConfidence": 0.9}, [candidate]
        )
        self.assertLess(low_risk["risk_factor"], high_risk["risk_factor"])
        self.assertLessEqual(low_risk["recommended"], high_risk["recommended"])

        diverse = deepcopy(base)
        stacked = deepcopy(base)
        first = player("Primo", "P")
        second = player("Secondo", "D")
        asta_core.record_purchase(diverse, {**first, "Squadra": "Club A"}, "bro", 1)
        asta_core.record_purchase(diverse, {**second, "Squadra": "Club B"}, "bro", 1)
        asta_core.record_purchase(stacked, first, "bro", 1)
        asta_core.record_purchase(stacked, second, "bro", 1)
        diverse_advice = asta_core.auction_advice(diverse, candidate, [candidate])
        stacked_advice = asta_core.auction_advice(stacked, candidate, [candidate])
        self.assertEqual(diverse_advice["club_factor"], 1.0)
        self.assertEqual(stacked_advice["club_factor"], 0.95)
        self.assertLessEqual(stacked_advice["recommended"], diverse_advice["recommended"])

    def test_fewer_alternatives_raise_player_factor(self):
        candidate = player("Top", "A")
        many = [candidate] + [
            {**player(f"Alternativa {index}", "A"), "SeasonValue": 0.79}
            for index in range(6)
        ]
        scarce = asta_core.auction_advice(self.session, candidate, [candidate])
        abundant = asta_core.auction_advice(self.session, candidate, many)
        self.assertGreater(scarce["player_factor"], abundant["player_factor"])
        self.assertGreaterEqual(scarce["recommended"], abundant["recommended"])

    def test_opponent_demand_lowers_advice_when_role_is_full(self):
        session = asta_core.new_session(
            budget=100,
            fair={"P": [1], "D": [1], "C": [1], "A": [100]},
            slots={"P": 0, "D": 0, "C": 0, "A": 1},
            created="2026-08-10T13:00:00",
        )
        candidate = player("Mio obiettivo", "A")
        before = asta_core.auction_advice(session, candidate, [candidate], current_price=1)
        for index, team in enumerate(session["meta"]["teams"][1:], start=1):
            asta_core.record_purchase(
                session, player(f"Preso {index}", "A"), team, 100
            )
        after = asta_core.auction_advice(session, candidate, [candidate], current_price=1)
        self.assertEqual(before["eligible_bidders"], 9)
        self.assertEqual(after["eligible_bidders"], 0)
        self.assertLess(after["demand_factor"], before["demand_factor"])
        self.assertLess(after["recommended"], before["recommended"])

    def test_hot_market_raises_advice_without_crossing_stop(self):
        base = asta_core.new_session(
            budget=100,
            fair={"P": [1], "D": [1], "C": [1], "A": [100, 20]},
            slots={"P": 0, "D": 0, "C": 0, "A": 2},
            created="2026-08-10T14:00:00",
        )
        hot = deepcopy(base)
        cold = deepcopy(base)
        for index, team in enumerate(base["meta"]["teams"][1:6], start=1):
            asta_core.record_purchase(hot, player(f"Hot {index}", "A", 2), team, 25)
            asta_core.record_purchase(cold, player(f"Cold {index}", "A", 2), team, 10)
        candidate = player("Obiettivo", "A")
        hot_advice = asta_core.auction_advice(hot, candidate, [candidate], current_price=1)
        cold_advice = asta_core.auction_advice(cold, candidate, [candidate], current_price=1)
        self.assertGreater(hot_advice["market_factor"], cold_advice["market_factor"])
        self.assertGreater(hot_advice["recommended"], cold_advice["recommended"])
        self.assertLessEqual(hot_advice["recommended"], hot_advice["fixed_cap"])

    def test_completed_history_is_shrunk_and_purchase_keeps_snapshot(self):
        historical = asta_core.new_session(
            budget=100,
            fair={"P": [10], "D": [10], "C": [10], "A": [10]},
            slots={"P": 1, "D": 1, "C": 1, "A": 1},
            created="2025-08-10T12:00:00",
        )
        historical["meta"]["completed"] = True
        historical["purchases"] = [
            {"name": f"Storico {index}", "role": "A", "cluster": 1,
             "price": 12, "cap": 10, "team": historical["meta"]["teams"][0]}
            for index in range(5)
        ]
        calibration = asta_core._historical_calibration(
            self.session, "A", 1, completed_sessions=[historical]
        )
        self.assertGreater(calibration["factor"], 1.0)
        self.assertLess(calibration["factor"], 1.2)

        candidate = player("Con snapshot", "A")
        advice = asta_core.auction_advice(self.session, candidate, [candidate])
        purchase = asta_core.record_purchase(
            self.session, candidate, "bro", 5, advice_snapshot=advice
        )
        self.assertEqual(purchase["advice_version"], "2.0")
        self.assertEqual(purchase["recommended_at_sale"], advice["recommended"])
        asta_core.set_session_completed(self.session, True)
        self.assertTrue(self.session["meta"]["completed"])
        asta_core.undo_purchase(self.session, candidate["Nome"])
        self.assertFalse(self.session["meta"]["completed"])


if __name__ == "__main__":
    unittest.main()
