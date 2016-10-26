import unittest
import datetime
import shutil

import numpy as np

from ..core import CohortManager
from ..drugs.chembl import ChEMBL, ChEMBLNotInstalled


NO_CHEMBL_MESSAGE = "Local ChEMBL database not installed."
try:
    chembl = ChEMBL()
    CHEMBL_INSTALLED = True
except ChEMBLNotInstalled:
    CHEMBL_INSTALLED = False


def _build_manager():
    return CohortManager("_TestManager")


@unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
class TestManagerDrugs(unittest.TestCase):
    def setUp(self):
        self.manager = _build_manager()

    def tearDown(self):
        if not self.manager.closed:
            self.manager.close()

        try:
            shutil.rmtree("_TestManager")
        except FileNotFoundError:
            pass

    def test_get_set_drug_user(self):
        self.manager.set_samples(["a", "b", "c"])
        self.manager.register_drug_user(12345, "a")
        self.manager.register_drug_user(12345, "c")
        self.manager.register_drug_user(11111, "b")
        self.manager.register_drug_user(11111, "a")
        self.manager.register_drug_user(22222, "b")

        # 2 on.
        v = self.manager.get_drug_users(12345)
        self.assertTrue(np.all(
            np.array([1, 0, 1]) == v
        ))

        # As bool.
        v = self.manager.get_drug_users(12345, as_bool=True)
        self.assertTrue(np.all(
            np.array([True, False, True], dtype=bool) == v
        ))

        # Drug not in db.
        v = self.manager.get_drug_users(94141)
        self.assertTrue(np.all(
            np.array([0, 0, 0]) == v
        ))

        # Single user.
        v = self.manager.get_drug_users(22222)
        self.assertTrue(np.all(
            np.array([0, 1, 0]) == v
        ))

        # Other samples.
        v = self.manager.get_drug_users(11111)
        self.assertTrue(np.all(
            np.array([1, 1, 0]) == v
        ))

    def test_get_set_bad_drug_user(self):
        self.manager.set_samples(["a", "b"])

        with self.assertRaises(ValueError):
            self.manager.register_drug_user(12345, "z")

    def test_date_range(self):
        """Test the 'between_dates' filter.

        Brackets: Pharmacotherapy time frame.
        Parenthesis: Query.

           1 3 4  6 7 9  12 15  (dates)
        ---[ ( ]--[ ]-)--[  ]--

        Sample 'd' is on another month.

        """
        self.manager.set_samples(list("abcd"))
        self.manager.register_drug_user(
            12345, "a",
            start_date=datetime.date(year=2016, month=1, day=1),
            end_date=datetime.date(year=2016, month=1, day=4),
        )

        self.manager.register_drug_user(
            12345, "b",
            start_date=datetime.date(year=2016, month=1, day=6),
            end_date=datetime.date(year=2016, month=1, day=9),
        )

        self.manager.register_drug_user(
            12345, "c",
            start_date=datetime.date(year=2016, month=1, day=12),
            end_date=datetime.date(year=2016, month=1, day=15),
        )

        self.manager.register_drug_user(
            12345, "d",
            start_date=datetime.date(year=2016, month=2, day=6),
            end_date=datetime.date(year=2016, month=2, day=9),
        )

        res = self.manager.get_drug_users(
            12345,
            between_dates=("2016-01-03", "2016-01-09")
        )

        np.testing.assert_array_equal(
            res, np.array([1, 1, 0, 0], dtype=float)
        )

    def test_date_range_start_only(self):
        """Test the 'between_dates' start only filter.

        See convention from `test_date_range`

            3  5   7   10 12   16 18
        ----[  ]---(---[  ]----[  ]--

        """
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "a",
            start_date=datetime.date(year=2016, month=3, day=3),
            end_date=datetime.date(year=2016, month=3, day=5),
        )

        self.manager.register_drug_user(
            12345, "b",
            start_date=datetime.date(year=2016, month=3, day=10),
            end_date=datetime.date(year=2016, month=3, day=12),
        )

        self.manager.register_drug_user(
            12345, "c",
            start_date=datetime.date(year=2016, month=3, day=16),
            end_date=datetime.date(year=2016, month=3, day=18),
        )

        res = self.manager.get_drug_users(
            12345, between_dates=(
                datetime.date(year=2016, month=3, day=7), None
            )
        )

        np.testing.assert_array_equal(
            res, np.array([0, 1, 1], dtype=float)
        )

    def test_date_range_end_only(self):
        """Test the 'between_dates' end only filter.

        See convention from `test_date_range`

            3  5   7   10 12   16 18
        ----[  ]---)---[  ]----[  ]--

        """
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "a",
            start_date=datetime.date(year=2016, month=3, day=3),
            end_date=datetime.date(year=2016, month=3, day=5),
        )

        self.manager.register_drug_user(
            12345, "b",
            start_date=datetime.date(year=2016, month=3, day=10),
            end_date=datetime.date(year=2016, month=3, day=12),
        )

        self.manager.register_drug_user(
            12345, "c",
            start_date=datetime.date(year=2016, month=3, day=16),
            end_date=datetime.date(year=2016, month=3, day=18),
        )

        res = self.manager.get_drug_users(
            12345, between_dates=(
                None, datetime.date(year=2016, month=3, day=7)
            )
        )

        np.testing.assert_array_equal(
            res, np.array([1, 0, 0], dtype=float)
        )

    def test_date_range_nones(self):
        """Test two None dates in the drug between_dates query."""
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "a",
            start_date=datetime.date(year=2016, month=3, day=16),
            end_date=datetime.date(year=2016, month=3, day=18),
        )

        with self.assertRaises(ValueError):
            self.manager.get_drug_users(
                12345, between_dates=(None, None)
            )

    def test_date_range_bad(self):
        """Test a malformed date in the drug between_dates query."""
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "a",
            start_date=datetime.date(year=2016, month=3, day=16),
            end_date=datetime.date(year=2016, month=3, day=18),
        )

        with self.assertRaises(ValueError):
            self.manager.get_drug_users(
                12345, between_dates=("2015-13-01", None)
            )

    def test_indication_filter(self):
        """Test the drug indication filter."""
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "a",
            indication="This patient has headaches."
        )
        self.manager.register_drug_user(
            12345, "c",
            indication="Diabetes"
        )
        res = self.manager.get_drug_users(12345, indication="%HEADACHE%")
        np.testing.assert_array_equal(res, np.array([1, 0, 0], dtype=float))

    def test_dose_filter(self):
        """Test filtering for dose."""
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(12345, "a", dose=10, dose_unit="mg")
        self.manager.register_drug_user(12345, "b", dose=40, dose_unit="mg")
        self.manager.register_drug_user(12345, "c", dose=80, dose_unit="mg")

        res = self.manager.get_drug_users(12345, dose=(40, "mg"))
        np.testing.assert_array_equal(res, np.array([0, 1, 0], dtype=float))

        res = self.manager.get_drug_users(12345, dose=(10, None))
        np.testing.assert_array_equal(res, np.array([1, 0, 0], dtype=float))

    def test_bad_drug_date(self):
        """Test inserting a drug with a bad dates."""
        self.manager.set_samples(list("abc"))
        self.manager.register_drug_user(
            12345, "b", start_date=datetime.date.today(), end_date=None
        )

        self.manager.register_drug_user(
            12345, "c", start_date="2015-01-01", end_date=datetime.date.today()
        )

        with self.assertRaises(ValueError):
            self.manager.register_drug_user(
                12345, "a", start_date="2015-13-01"
            )

        with self.assertRaises(ValueError):
            self.manager.register_drug_user(
                12345, "a", end_date="2015-13-01"
            )

    def test_drug_start_before_end(self):
        """Test that an error is raised if the start date is after the end
        date.
        """
        self.manager.set_samples(list("abc"))

        with self.assertRaises(ValueError):
            self.manager.register_drug_user(
                12345, "a", start_date="2015-01-01",
                end_date=datetime.date(year=2013, month=1, day=2)
            )

    def test_non_numeric_dose(self):
        """Test inserting with a non-numeric dose."""
        self.manager.set_samples(list("abc"))
        with self.assertRaises(ValueError):
            self.manager.register_drug_user(12345, "a", dose="potato")

    def test_get_drug_users_atc(self):
        """Test getting drug users by ATC codes."""
        self.manager.set_samples(list("ABCDEF"))

        # A, C and D take different statins.
        # A: Atorvastatin (parent): 417180
        # C: Atorvastatin calcium (child): 407354
        # D: Simvastatin: 138562
        self.manager.register_drug_user(417180, "A")
        self.manager.register_drug_user(407354, "C")
        self.manager.register_drug_user(138562, "D")

        statin = self.manager.get_drug_users_atc("C10AA")

        self.assertTrue(
            #                          A    B    C    D    E    F
            np.all(statin == np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
        )

    def test_get_drug_users_protein(self):
        """Test getting drug users from protein ID."""
        self.manager.set_samples(list("ABCDEF"))

        # A, C and D take different statins.
        # A: Atorvastatin (parent): 417180
        # C: Atorvastatin calcium (child): 407354
        # D: Simvastatin: 138562
        self.manager.register_drug_user(417180, "A")
        self.manager.register_drug_user(407354, "C")
        self.manager.register_drug_user(138562, "D")

        statin = self.manager.get_drug_users_protein("P04035")

        self.assertTrue(
            #                          A    B    C    D    E    F
            np.all(statin == np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
        )

    def test_virtuals_including_drugs(self):
        """Test the virtual variable interface for drugs."""
        self.manager.set_samples(list("ABCD"))
        self.manager.add_phenotype(
            name="d1", variable_type="discrete"
        )
        self.manager.add_data("d1", [0, 1, np.nan, 1])

        self.manager.register_drug_user(12345, "A")
        self.manager.register_drug_user(12345, "C")
        self.manager.register_drug_user(12345, "D")

        v = self.manager.variable
        drug = self.manager.drug

        np.testing.assert_array_equal(
            (v("d1") & drug(12345)).data,
            np.array([0, 0, np.nan, 1])
        )

    def test_virtuals_including_drugs_atc(self):
        """Test the virtual variable interface for drugs using ATC code."""
        self.manager.set_samples(list("ABCD"))
        self.manager.add_phenotype(
            name="d1", variable_type="discrete"
        )
        self.manager.add_data("d1", [0, 1, np.nan, 1])

        # 27417 is bisoprolol, a beta-blocker (C07)
        self.manager.register_drug_user(27417, "A")
        self.manager.register_drug_user(27417, "C")
        self.manager.register_drug_user(27417, "D")

        v = self.manager.variable
        drug = self.manager.drug

        np.testing.assert_array_equal(
            (v("d1") & drug("C07")).data,
            np.array([0, 0, np.nan, 1])
        )
