import unittest
import shutil

import numpy as np

from ..core import CohortManager, FrozenCohortError, UnknownSamplesError


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
        with self.assertRaises(TypeError):
            self.manager.add_phenotype(variable_type="discrete")

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

    def test_restore_manager(self):
        self.manager.set_samples(list("ABC"))
        self.manager.close()
        self.manager2 = CohortManager("_TestManager")
        self.assertTrue(np.all(
            self.manager2.get_samples() == np.array(["A", "B", "C"])
        ))
        self.manager2.close()

    def test_context_manager(self):
        self.manager.close()
        with CohortManager("_TestManager") as manager:
            self.assertTrue(type(manager) is CohortManager)

    def test_manager_persistent_keys(self):
        self.manager["foo"] = "bar"
        self.manager.close()
        with CohortManager("_TestManager") as manager:
            self.assertEqual(manager["foo"], "bar")
            manager["foo"] = "update"
            self.assertEqual(manager["foo"], "update")

    def test_manager_n_property(self):
        self.assertEqual(self.manager.n, None)
        self.manager.set_samples(["A", "B", "C"])
        self.assertEqual(self.manager.n, 3)

    def test_update_phenotype(self):
        self.test_add_data()
        self.manager.update_phenotype("age", crf_page=3)
        phen = self.manager.get_phenotype("age")
        self.assertEqual(phen["crf_page"], 3)

    def test_update_phenotype_unset(self):
        self.manager.set_samples(list("ABCDEF"))
        self.manager.add_phenotype(
            name="age",
            variable_type="continuous",
            crf_page=3
        )
        self.manager.add_data("age", [10, 21, 30, 39, 82, 39])
        self.manager.update_phenotype("age", crf_page=None)
        self.assertEqual(
            self.manager.get_phenotype("age")["crf_page"],
            None
        )

    def test_reset_samples_error(self):
        self.manager.set_samples(list("ABCD"))
        with self.assertRaises(FrozenCohortError):
            self.manager.set_samples(list("ABCDE"))

    def test_add_data(self):
        """Sets the sample order and creates a simple 'age' phenotype with
        bound data.
        """
        self.manager.set_samples(list("ABCDEF"))
        self.manager.add_phenotype(name="age", variable_type="continuous")
        self.manager.add_data("age", [10, 21, 30, 39, 82, 39])


    def test_add_data_na_phenotype(self):
        """Add data to an unexisting phenotype."""
        self.test_add_data()
        with self.assertRaises(ValueError):
            self.manager.add_data("test", [1, 2, 3, 4, 5, 6])

    def test_add_data_na_phenotype(self):
        """Add data before setting sample order."""
        self.manager.add_phenotype(name="test", variable_type="continuous")
        with self.assertRaises(UnknownSamplesError):
            self.manager.add_data("test", [1, 2, 3, 4, 5, 6])

    def test_add_data_bad_number(self):
        """Add data with an incorrect shape."""
        self.test_add_data()

        # Extra.
        with self.assertRaises(ValueError):
            self.manager.add_data("age", [1, 2, 3, 4, 5, 6, 7])

        # Missing.
        with self.assertRaises(ValueError):
            self.manager.add_data("age", [1, 2, 3])

    def test_add_data_overwrite(self):
        """Add data two times."""
        self.test_add_data()
        with self.assertRaises(ValueError):
            self.manager.add_data("age", [10, 21, 30, 39, 82, 39])
