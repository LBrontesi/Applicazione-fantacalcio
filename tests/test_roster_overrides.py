import unittest

import data_loader


class RosterOverrideTests(unittest.TestCase):
    def test_lukaku_is_not_in_current_serie_a_quotations(self):
        quotations = data_loader.load_quotazioni()
        self.assertNotIn("Lukaku", quotations["NomeGaz"].astype(str).tolist())


if __name__ == "__main__":
    unittest.main()
