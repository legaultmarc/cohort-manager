import unittest

from ..drugs import atc


class TestATC(unittest.TestCase):
    def test_get_atc_code_level_1(self):
        codes = list("ABCDGHJLMNPRSV")
        for code in codes:
            self.assertEqual(atc.get_atc_code_level(code), 1)

    def test_get_atc_code_level_2(self):
        self.assertEqual(atc.get_atc_code_level("C03"), 2)
        self.assertEqual(atc.get_atc_code_level("N07"), 2)

    def test_get_atc_code_level_3(self):
        self.assertEqual(atc.get_atc_code_level("C03C"), 3)
        self.assertEqual(atc.get_atc_code_level("C03D"), 3)
        self.assertEqual(atc.get_atc_code_level("V06B"), 3)

    def test_get_atc_code_level_4(self):
        self.assertEqual(atc.get_atc_code_level("V06DB"), 4)
        self.assertEqual(atc.get_atc_code_level("B01AA"), 4)
        self.assertEqual(atc.get_atc_code_level("B01AF"), 4)

    def test_get_atc_code_level_5(self):
        self.assertEqual(atc.get_atc_code_level("B01AA03"), 5)
        self.assertEqual(atc.get_atc_code_level("B01AC06"), 5)
        self.assertEqual(atc.get_atc_code_level("C03BA07"), 5)

    def test_get_atc_code_level_invalids(self):
        with self.assertRaises(ValueError):
            atc.get_atc_code_level("Z")

        with self.assertRaises(ValueError):
            atc.get_atc_code_level("C1")

        with self.assertRaises(ValueError):
            atc.get_atc_code_level("C999")

        with self.assertRaises(ValueError):
            atc.get_atc_code_level("C03BA089")
