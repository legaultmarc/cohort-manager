import unittest.mock
import unittest
import shutil

import numpy as np
import pandas as pd

from . import datasets
from .. import core
from .. import backend
from .. import types


def _build_manager():
    return core.CohortManager("_TestManager")


class TestManager(unittest.TestCase):
    def setUp(self):
        self.manager = _build_manager()

    def tearDown(self):
        if not self.manager.closed:
            self.manager.close()

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
        self.manager.set_samples(np.array(["a", "b", "c"]))
        self.assertTrue(
            np.all(self.manager.get_samples() == np.array(["a", "b", "c"]))
        )

    def test_get_samples(self):
        """Test the get_samples method."""
        arr = np.array(["a", "b", "c", "d"])
        self.manager.set_samples(arr)
        self.assertTrue(self.manager.n == 4)
        self.assertTrue(
            np.all(self.manager.get_samples() == arr.astype(str))
        )

    def test_get_samples_unset(self):
        """Test get_samples before setting (returns None)."""
        self.assertEqual(self.manager.get_samples(), None)

    def test_add_phenotype(self):
        """Test adding a phenotypes of different types."""
        phenotypes = [("p1", "discrete"), ("p2", "continuous"),
                      ("p3", "factor")]
        for name, _type in phenotypes:
            # Insert.
            kwargs = {
                "name": name,
                "variable_type": _type,
            }
            if _type == "factor":
                kwargs["code_name"] = "c"

            self.manager.add_phenotype(**kwargs)

            # Get back.
            out = self.manager.get_phenotype(name)
            self.assertEqual(out["name"], name)
            self.assertEqual(out["variable_type"], _type)

            if _type == "factor":
                self.assertEqual(out["code_name"], "c")

    def test_add_dummy_phenotype(self):
        """Tests adding a dummy phenotype."""
        # Inserting a dummy phenotype
        self.manager.add_dummy_phenotype("dummy_pheno")

        # Checking the variable_type
        out = self.manager.get_phenotype("dummy_pheno")
        self.assertEqual("dummy_pheno", out["name"])
        self.assertEqual("dummy", out["variable_type"])

    def test_get_phenotype(self):
        """Test get missing phenotype."""
        with self.assertRaises(KeyError):
            self.manager.get_phenotype("test")

    def test_add_phenotype_full_data(self):
        # Insert parent.
        self.manager.add_phenotype(name="parent", variable_type="discrete")

        # Insert code.
        self.manager.add_codes([("c", 0, "a"), ("c", 1, "b")])

        # Add phenotype.
        params = {
            "name": "TestPhenotype",
            "snomed": "22298006",
            "parent": "parent",
            "variable_type": "factor",
            "description": "Favorite animal",
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

    def test_delete_phenotypes(self):
        """Tests when deleting a phenotype."""
        self.manager.set_samples(list("ABCDEF"))
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")
        self.manager.add_phenotype(name="phenotype3",
                                   variable_type="continuous")
        self.manager.add_data("phenotype1", range(6))

        self.manager.delete("phenotype1")
        self.assertFalse(self.manager.is_valid_phenotype("phenotype1"))

    def test_delete_invalid_genotypes(self):
        """Tests when deleting an invalid phenotype."""
        with self.assertRaises(ValueError):
            self.manager.delete("phenotype1")

    def test_delete_dummy_phenotypes(self):
        """Tests when deleting a dummy phenotype."""
        self.manager.add_dummy_phenotype("dummy1")
        self.manager.delete("dummy1")

        self.assertFalse(self.manager.is_valid_phenotype("dummy1"))
        self.assertFalse(self.manager.is_dummy_phenotype("dummy1"))

    def test_is_valid_phenotype(self):
        """Tests if the phenotype is valid."""
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_dummy_phenotype("dummy1")

        self.assertTrue(self.manager.is_valid_phenotype("phenotype1"))
        self.assertTrue(self.manager.is_valid_phenotype("dummy1"))
        self.assertFalse(self.manager.is_valid_phenotype("phenotype2"))

    def test_is_dummy_phenotype(self):
        """Tests if the phenotype is a dummy one."""
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_dummy_phenotype("dummy1")

        self.assertTrue(self.manager.is_dummy_phenotype("dummy1"))
        self.assertFalse(self.manager.is_dummy_phenotype("phenotype1"))

    def test_add_code(self):
        self.manager.add_code("a_code", 0, "value1")
        self.manager.add_code("a_code", 1, "value2")
        for key, value in self.manager.get_code("a_code"):
            if key == 0:
                self.assertEqual(value, "value1")
            elif key == 1:
                self.assertEqual(value, "value2")
            else:
                raise Exception("Unexpected code entry found.")

    def test_add_codes(self):
        self.manager.add_codes([
            ("a_code", 0, "value1"),
            ("a_code", 1, "value2"),
        ])
        for key, value in self.manager.get_code("a_code"):
            if key == 0:
                self.assertEqual(value, "value1")
            elif key == 1:
                self.assertEqual(value, "value2")
            else:
                raise Exception("Unexpected code entry found.")

    def test_restore_manager(self):
        self.manager.set_samples(list("ABC"))
        self.manager.close()
        self.manager2 = core.CohortManager("_TestManager")
        self.assertTrue(np.all(
            self.manager2.get_samples() == np.array(["A", "B", "C"])
        ))
        self.manager2.close()

    def test_context_manager(self):
        self.manager.close()
        with core.CohortManager("_TestManager") as manager:
            self.assertTrue(type(manager) is core.CohortManager)

    def test_manager_persistent_keys(self):
        self.manager["foo"] = "bar"
        self.manager.close()
        with core.CohortManager("_TestManager") as manager:
            self.assertEqual(manager["foo"], "bar")
            manager["foo"] = "update"
            self.assertEqual(manager["foo"], "update")

    def test_manager_n_property(self):
        self.assertEqual(self.manager.n, None)
        self.manager.set_samples(["A", "B", "C"])
        self.assertEqual(self.manager.n, 3)

    def test_update_phenotype(self):
        self.test_add_data()
        self.manager.update_phenotype("age", description="test")
        phen = self.manager.get_phenotype("age")
        self.assertEqual(phen["description"], "test")

    def test_update_phenotype_unset(self):
        self.manager.set_samples(list("ABCDEF"))
        self.manager.add_phenotype(
            name="age",
            variable_type="continuous",
            description="apple"
        )
        self.manager.add_data("age", [10, 21, 30, 39, 82, 39])
        self.manager.update_phenotype("age", description=None)
        self.assertEqual(
            self.manager.get_phenotype("age")["description"],
            None
        )

    def test_reset_samples_error(self):
        self.manager.set_samples(list("ABCD"))
        with self.assertRaises(backend.FrozenCohortError):
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

    def test_add_data_no_samples(self):
        """Add data before setting sample order."""
        self.manager.add_phenotype(name="test", variable_type="continuous")
        with self.assertRaises(backend.UnknownSamplesError):
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

    def test_check_discrete(self):
        """Test that checks for discrete outcomes get applied."""
        self.manager.set_samples(list("abcdef"))
        self.manager.add_phenotypes([
            dict(name="phenotype1", variable_type="discrete"),
            dict(name="phenotype2", variable_type="discrete"),
            dict(name="phenotype3", variable_type="discrete"),
        ])

        v = np.array([0, np.nan, 1, 0, 1, np.nan])
        self.manager.add_data("phenotype1", v)
        np.testing.assert_array_equal(
            v, self.manager.get_data("phenotype1")
        )

        v = [0, 1, 0, 1, 0, 0]
        self.manager.add_data("phenotype2", v)
        np.testing.assert_array_equal(
            np.array(v, dtype=float),
            self.manager.get_data("phenotype2")
        )

        # This should raise a ValueError because it's not only 0 and 1.
        with self.assertRaises(types.InvalidValues):
            self.manager.add_data("phenotype3", [0, 1, 1, 2, 0, 0])

        # Same thing but with more heterogeneity.
        with self.assertRaises(types.InvalidValues):
            self.manager.add_data("phenotype3", [1, 2, 3, 4, 5, 6])

    def test_check_data_continuous(self):
        """Check the QC checks for continuous data."""
        redundant = np.random.binomial(1, 0.4, 100)
        frequent_outlier = np.random.random(100)
        for _ in range(20):
            frequent_outlier[np.random.randint(0, 100)] = -9

        with self.assertRaises(types.InvalidValues) as cm:
            types.Continuous.check(
                redundant, _raise=True
            )
        self.assertEqual(
            cm.exception.args[0],
            "There is a lot of redundancy in the values for this continuous "
            "variable. Perhaps it should be modeled as a factor or another "
            "variable type."
        )

        with self.assertRaises(types.InvalidValues) as cm:
            types.Continuous.check(frequent_outlier, _raise=True)
        self.assertEqual(
            cm.exception.args[0],
            "The value '-9.0' is commonly found in the tails of the "
            "distribution. This could be because of bad coding of missing "
            "values."
        )

    def test_check_data_continuous_large(self):
        self.manager.set_samples(["s{}".format(i + 1) for i in range(6000)])
        self.manager.add_phenotype(name="phen1", variable_type="continuous")
        v = np.random.random(6000)
        v[3] = np.nan
        v[1024] = np.nan
        self.manager.add_data("phen1", v)
        np.testing.assert_array_equal(v, self.manager.get_data("phen1"))

    def test_check_data_factor(self):
        """Check correct factor data."""
        self.manager.add_code("code1", 1, "Sunny")
        self.manager.add_code("code1", 2, "Rain")
        self.manager.add_phenotype(
            name="phenotype1",
            code_name="code1",
            variable_type="factor"
        )
        self.manager.set_samples(list("abcdef"))
        self.manager.add_data("phenotype1", [1, 2, 1, 2, 1, 2])

    def test_check_data_factor_bad(self):
        """Check incorrect factor data."""
        self.manager.add_code("code1", 1, "Sunny")
        self.manager.add_code("code1", 2, "Rain")
        self.manager.add_phenotype(
            name="phenotype1",
            code_name="code1",
            variable_type="factor"
        )
        self.manager.set_samples(list("abcdef"))
        with self.assertRaises(types.InvalidValues):
            self.manager.add_data("phenotype1", [1, 2, 1, 0, 1, 1])

    def test_check_data_factor_nocode(self):
        """Check bad code name."""
        self.manager.add_phenotype(name="phenotype1", code_name="code1",
                                   variable_type="factor")
        self.manager.set_samples(list("abcdef"))
        with self.assertRaises(core.CohortDataError):
            self.manager.add_data("phenotype1", [1, 2, 1, 0, 1, 1])

    def test_contingency(self):
        """Test the contingency table."""
        self.manager.set_samples(list("abcdefghij"))
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="discrete")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")

        self.manager.add_data("phenotype1", [0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
        self.manager.add_data("phenotype2", [0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        df = self.manager.contingency("phenotype2", "phenotype1")
        self.assertEqual(
            df.loc["phenotype2 - control", "phenotype1 - control"], 3
        )

        self.assertEqual(
            df.loc["phenotype2 - case", "phenotype1 - control"], 1
        )

        self.assertEqual(
            df.loc["phenotype2 - control", "phenotype1 - case"], 4
        )

        self.assertEqual(
            df.loc["phenotype2 - case", "phenotype1 - case"], 2
        )

    def test_contingency_with_missing(self):
        """Test the contingency table with missing values."""
        self.manager.set_samples(list("abcdefghij"))
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="discrete")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")

        self.manager.add_data(
            "phenotype1",
            [np.nan, 0, 0, np.nan, 1, 0, np.nan, 1, 1, np.nan]
        )
        self.manager.add_data(
            "phenotype2",
            [np.nan, 0, 0, np.nan, np.nan, np.nan, 0, 0, 0, 0]
        )

        df = self.manager.contingency("phenotype2", "phenotype1")
        np.testing.assert_array_equal(
            df.values,
            np.array([[2, 1, 1], [2, 2, 2], [0, 0, 0]])
        )

    def test_contingency_factor(self):
        """Test the contingency table with factor variable."""
        self.manager.set_samples(list("abcdefghij"))
        self.manager.add_codes([("c", 1, "sunny"), ("c", 2, "rainy"),
                                ("c", 3, "windy")])
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="factor",
                                   code_name="c")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")

        m = np.nan
        self.manager.add_data("phenotype1", [m, m, 1, 1, 1, 1, 2, 2, 3, 3])
        self.manager.add_data("phenotype2", [m, 0, m, 0, 0, 1, 0, 1, m, 1])

        answers = {
            ("phenotype1 - missing", "phenotype2 - missing"): 1,
            ("phenotype1 - sunny", "phenotype2 - missing"): 1,
            ("phenotype1 - rainy", "phenotype2 - missing"): 0,
            ("phenotype1 - windy", "phenotype2 - missing"): 1,
            ("phenotype1 - missing", "phenotype2 - control"): 1,
            ("phenotype1 - sunny", "phenotype2 - control"): 2,
            ("phenotype1 - rainy", "phenotype2 - control"): 1,
            ("phenotype1 - windy", "phenotype2 - control"): 0,
            ("phenotype1 - missing", "phenotype2 - case"): 0,
            ("phenotype1 - sunny", "phenotype2 - case"): 1,
            ("phenotype1 - rainy", "phenotype2 - case"): 1,
            ("phenotype1 - windy", "phenotype2 - case"): 1,
        }

        df = self.manager.contingency("phenotype1", "phenotype2")
        for tu, ans in answers.items():
            row, col = tu
            self.assertEqual(df.loc[row, col], ans)

    def test_contingency_bad_type(self):
        """Test contingency on bad variable type."""
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")

        with self.assertRaises(ValueError):
            self.manager.contingency("phenotype1", "phenotype2")

    def test_get_n_pheno(self):
        """Get the number of phenotypes in the manager."""
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")
        self.manager.add_phenotype(name="phenotype3",
                                   variable_type="continuous")
        self.assertEqual(self.manager.get_number_phenotypes(), 3)

    def test_get_phenotypes(self):
        """Test getting a list of phenotype names."""
        self.manager.add_phenotype(name="phenotype1",
                                   variable_type="continuous")
        self.manager.add_phenotype(name="phenotype2",
                                   variable_type="discrete")
        self.manager.add_phenotype(name="phenotype3",
                                   variable_type="continuous")
        self.manager.add_dummy_phenotype("dummy_1")

        # Testing without dummies
        li = self.manager.get_phenotypes_list()
        self.assertEqual(len(li), 3)
        for phen in ("phenotype1", "phenotype2", "phenotype3"):
            self.assertTrue(phen in li)

        # Testing with dummies
        li = self.manager.get_phenotypes_list(dummy=True)
        self.assertEqual(len(li), 4)
        for phen in ("phenotype1", "phenotype2", "phenotype3", "dummy_1"):
            self.assertTrue(phen in li)

    def test_get_data(self):
        """Test the data getter when there is not data."""
        with self.assertRaises(KeyError):
            self.manager.get_data("test")

    def test_get_factor_data(self):
        """Test the decoding of factor variables."""
        self.manager.set_samples(list("abcdefghi"))
        self.manager.add_codes([("c", 1, "male"), ("c", 2, "female")])
        self.manager.add_phenotype(name="gender", variable_type="factor",
                                   code_name="c")

        nans = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=bool)
        self.manager.add_data("gender", [1, 2, 1, 1, 2, np.nan, 2, 1, np.nan])

        v = self.manager.get_data("gender")
        ans = pd.Series(["male", "female", "male", "male", "female", np.nan,
                         "female", "male", np.nan]).astype("category")

        self.assertTrue(
            np.all(v.iloc[~nans] == ans.iloc[~nans])
        )

    def test_get_factor_data_with_code(self):
        """Test the decoding of factor variables build with pandas code."""
        self.manager.set_samples(list("abcdefghi"))
        self.manager.add_codes([("c", 0, "male"), ("c", 1, "female")])
        self.manager.add_phenotype(name="gender", variable_type="factor",
                                   code_name="c")

        nans = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=bool)
        self.manager.add_data("gender", [0, 1, 0, 0, 1, np.nan, 1, 0, np.nan])

        v = self.manager.get_data("gender")

        ans = pd.Series(["male", "female", "male", "male", "female", np.nan,
                         "female", "male", np.nan])
        ans = ans.astype("category",
                         categories=["male", "female"])

        self.assertTrue(
            np.all(v.iloc[~nans] == ans.iloc[~nans])
        )

    def test_discrete_reclassify(self):
        """Test the dynamic reclassification with respect to the hierarchy.

        A -> B -> C

        Observed:
           a b c d e f g h i
        A: 0 1 0 0 1 N 1 0 1
        B: N 1 N N 0 N 0 N 1
        C: N 0 N N N N N N 1

        Reality:
        A: 0 1 0 0 1 N 1 0 1
        B: 0 1 0 0 0 N 0 0 1
        C: 0 0 0 0 0 N 0 0 1

        """
        self.manager.set_samples(list("abcdefghi"))
        self.manager.add_phenotypes([
            dict(name="A", variable_type="discrete"),
            dict(name="B", variable_type="discrete", parent="A"),
            dict(name="C", variable_type="discrete", parent="B"),
        ])

        self.manager.add_data("A", [0, 1, 0, 0, 1, np.nan, 1, 0, 1])
        self.manager.add_data("B", [np.nan, 1, np.nan, np.nan, 0, np.nan, 0,
                                    np.nan, 1])
        self.manager.add_data("C", [np.nan, 0, np.nan, np.nan, np.nan, np.nan,
                                    np.nan, np.nan, 1])

        expected = {
            "A": np.array([0, 1, 0, 0, 1, np.nan, 1, 0, 1]),
            "B": np.array([0, 1, 0, 0, 0, np.nan, 0, 0, 1]),
            "C": np.array([0, 0, 0, 0, 0, np.nan, 0, 0, 1]),
        }
        for phenotype, answer in expected.items():
            np.testing.assert_array_equal(
                self.manager.get_data(phenotype), answer
            )

    def test_get_number_missing(self):
        """Test the get_number_missing core API function."""
        datasets.fill_hierarchy(self.manager)
        answers = {
            "grandparent_discrete": 1,
            "parent_discrete": 1,
            "child_date": 2,
            "child_discrete": 2,
            "child_continuous": 3,
        }
        for k in datasets.hierarchy:
            self.assertEqual(answers[k], self.manager.get_number_missing(k))

    def test_get_code_names(self):
        """Test getting a list of code names."""
        self.manager.add_codes([
            ("a", 1, "apple"), ("a", 2, "orange"),
            ("b", 1, "day"), ("b", 2, "night"),
            ("c", 0, "mon"), ("c", 1, "tue"), ("c", 2, "wed")
        ])
        self.assertEqual(self.manager.get_code_names(), {"a", "b", "c"})
