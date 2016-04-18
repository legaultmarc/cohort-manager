import unittest
import shutil

import numpy as np
import psycopg2

from ..core import CohortManager
from ..drugs.chembl import ChEMBL


NO_CHEMBL_MESSAGE = "Local ChEMBL database not installed."
try:
    chembl = ChEMBL()
    CHEMBL_INSTALLED = True
except psycopg2.OperationalError:
    CHEMBL_INSTALLED = False


def _build_manager():
    return CohortManager("_TestManager")


class TestManagerDrugs(unittest.TestCase):
    def setUp(self):
        self.tearDown()
        self.manager = _build_manager()

    def tearDown(self):
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

    @unittest.skipIf(not CHEMBL_INSTALLED, NO_CHEMBL_MESSAGE)
    def test_get_drug_users_atc(self):
        pass
