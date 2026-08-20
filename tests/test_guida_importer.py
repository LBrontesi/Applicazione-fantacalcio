import unittest

from guida_importer import normalize_payload


class GuidaImporterTests(unittest.TestCase):
    def test_normalizer_keeps_only_structured_fields_and_one_based_orders(self):
        raw = {
            "PlayerObject": [
                {"id": 7, "name": "Rossi", "team": "Roma", "classicRoleValue": "A", "playedGames": 30},
                {"id": 8, "name": "Bianchi", "team": "Roma", "classicRoleValue": "C"},
            ],
            "ProbableStartersObject": [{"formation": "433", "players": "7;8"}],
            "ProbableSetPiecesTakersObject": [{
                "penaltiesTakers": [{"playerId": 7, "order": 0}],
                "cornersTakers": [], "freeKicksTakers": [],
            }],
            "ProbableDoubtObject": [{
                "idFirstPlayer": 7, "idSecondPlayer": 8, "likelihoodFirstPlayer": 60,
            }],
        }
        payload = normalize_payload(raw)
        self.assertTrue(payload["players"][0]["starter"])
        self.assertEqual(payload["formations"][0]["formation"], "4-3-3")
        self.assertEqual(payload["set_pieces"][0]["order"], 1)
        self.assertEqual(payload["doubts"][0]["second"], "Bianchi")
        self.assertNotIn("playerDescription", payload["players"][0])


if __name__ == "__main__":
    unittest.main()
