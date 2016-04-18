import unittest
import shutil

import numpy as np

from ..core import CohortManager


def _build_manager():
    return CohortManager("_TestManager")


class TestManager(unittest.TestCase):
    def setUp(self):
        self.tearDown()
        self.manager = _build_manager()

    def tearDown(self):
        try:
            shutil.rmtree("_TestManager")
        except Exception:
            pass

    def test_set_samples_str(self):
        self.manager.set_samples(["a", "b", "c"])
        self.assertTrue(
            np.all(self.manager.get_samples() == np.array(["a", "b", "c"]))
        )

    def test_set_samples_np(self):
        self.manager.set_samples(np.array(["a", "b", "c"], dtype=np.string_))
        self.assertTrue(
            np.all(self.manager.get_samples() == np.array(["a", "b", "c"]))
        )

    def test_get_samples(self):
        arr = np.array(["a", "b", "c", "d"], dtype=np.string_)
        self.manager.set_samples(arr)
        self.assertTrue(self.manager.n == 4)
        self.assertTrue(
            np.all(self.manager.get_samples() == arr.astype(str))
        )

    def test_add_phenotype(self):
        # Insert.
        self.manager.add_phenotype(
            name="TestPhenotype",
            variable_type="discrete",
        )

        # Get back.
        out = self.manager.get_phenotype("TestPhenotype")
        self.assertEqual(out["name"], "TestPhenotype")
        self.assertEqual(out["variable_type"], "discrete")

    def test_add_phenotype_full_data(self):
        # Insert parent.
        self.manager.add_phenotype(name="parent", variable_type="discrete")

        # Insert code.
        self.manager.add_codes([("c", 0, "a"), ("c", 1, "b")])

        # Add phenotype.
        params = {
            "name": "TestPhenotype",
            "icd10": "C07",
            "parent": "parent",
            "variable_type": "factor",
            "crf_page": 3,
            "question": "What is your favorite animal?",
            "code_name": "c",
        }
        self.manager.add_phenotype(**params)

        # Get back.
        for k, v in self.manager.get_phenotype("TestPhenotype").items():
            self.assertEqual(params[k], v)

    def test_add_phenotype_missing_param(self):
        with self.assertRaises(TypeError):
            self.manager.add_phenotype(name="test")

    def test_add_phenotype_extra_param(self):
        with self.assertRaises(TypeError):
            self.manager.add_phenotype(name="test", variable_type="discrete",
                                       bad="bad")

    def test_add_phenotype_bad_type(self):
        with self.assertRaises(TypeError):
            self.manager.add_phenotype(name="test", variable_type="potato")

    def test_add_code(self):
        self.manager.add_code("a_code", 0, "value1")
        self.manager.add_code("a_code", 1, "value2")
        for code in self.manager.get_code("a_code"):
            if code["key"] == 0:
                self.assertEqual(code["value"], "value1")
            elif code["key"] == 1:
                self.assertEqual(code["value"], "value2")
            else:
                raise Exception("Unexpected code entry found.")

    def test_add_codes(self):
        self.manager.add_codes([
            ("a_code", 0, "value1"),
            ("a_code", 1, "value2"),
        ])
        for code in self.manager.get_code("a_code"):
            if code["key"] == 0:
                self.assertEqual(code["value"], "value1")
            elif code["key"] == 1:
                self.assertEqual(code["value"], "value2")
            else:
                raise Exception("Unexpected code entry found.")
