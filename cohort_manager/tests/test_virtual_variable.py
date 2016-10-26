import shutil
import unittest
import operator

import numpy as np

from .. import core


NAN_RATE = 0.2


def _discrete_vector(n, p=0.5, nans_p=NAN_RATE):
    v = (np.random.rand(n) <= p).astype(float)
    v[np.random.rand(n) <= nans_p] = np.nan
    return v


def _normal_vector(n, mu, sigma, nans_p=NAN_RATE):
    v = np.random.normal(mu, sigma, n)
    v[np.random.rand(n) <= nans_p] = np.nan
    return v


def _uniform_vector(n, nans_p=NAN_RATE):
    v = np.random.random(n)
    v[np.random.rand(n) <= nans_p] = np.nan
    return v


def _nan_comparison(x, y, op):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    else:
        return op(x, y)


_nan_comparison = np.vectorize(_nan_comparison, excluded=[2], otypes=[float])


class TestVariable(unittest.TestCase):
    def setUp(self):
        self.manager = core.CohortManager("_TestManager")
        self.data = {}

    def _generate_data(self):
        self.manager.set_samples(["s{}".format(i + 1) for i in range(100)])

        # Discrete variables.
        self.manager.add_phenotype(
            name="d1", variable_type="discrete"
        )
        self.data["d1"] = _discrete_vector(100)
        self.manager.add_data("d1", self.data["d1"])

        self.data["d2"] = _discrete_vector(100)
        self.manager.add_phenotype(
            name="d2", variable_type="discrete"
        )
        self.manager.add_data("d2", self.data["d2"])

        # Continuous variables.
        self.data["c1"] = _normal_vector(100, 3, 2)
        self.manager.add_phenotype(
            name="c1", variable_type="continuous"
        )
        self.manager.add_data("c1", self.data["c1"])

        self.data["c2"] = _uniform_vector(100)
        self.manager.add_phenotype(
            name="c2", variable_type="continuous"
        )
        self.manager.add_data("c2", self.data["c2"])

    def _truth_table(self):
        self.manager.set_samples(list("abcdefghi"))
        self.manager.add_phenotype(name="test1", variable_type="discrete")
        self.manager.add_phenotype(name="test2", variable_type="discrete")

        self.manager.add_data(
            "test1", [0, 0, 0, 1, 1, 1, np.nan, np.nan, np.nan]
        )

        self.manager.add_data(
            "test2", [0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan]
        )

    def tearDown(self):
        if not self.manager.closed:
            self.manager.close()

        try:
            shutil.rmtree("_TestManager")
        except Exception:
            pass

    def test_or(self):
        """Test | operator."""
        self._truth_table()
        v = self.manager.variable
        np.testing.assert_array_equal(
            (v("test1") | v("test2")).data,
            np.array([0, 1, np.nan, 1, 1, 1, np.nan, 1, np.nan])
        )

    def test_and(self):
        """Test & operator."""
        self._truth_table()
        v = self.manager.variable
        np.testing.assert_array_equal(
            (v("test1") & v("test2")).data,
            np.array([0, 0, 0, 0, 1, np.nan, 0, np.nan, np.nan])
        )

    def test_xor(self):
        """Test ^ (XOR) operator."""
        self._truth_table()
        v = self.manager.variable
        np.testing.assert_array_equal(
            (v("test1") ^ v("test2")).data,
            np.array([0, 1, np.nan, 1, 0, np.nan, np.nan, np.nan, np.nan])
        )

    def test_add_sub(self):
        """Test +, -, *, / operators with _Variable and constants."""
        self._generate_data()
        v = self.manager.variable

        for op in (operator.add, operator.sub, operator.truediv, operator.mul):
            # Constant.
            np.testing.assert_array_equal(
                op(v("c1"), 3).data,
                op(self.data["c1"], 3)
            )

            np.testing.assert_array_equal(
                op(3, v("c1")).data,
                op(self.data["c1"], 3)
            )

            # Other vector.
            np.testing.assert_array_equal(
                op(v("c1"), v("c2")).data,
                op(self.data["c1"], self.data["c2"])
            )

            np.testing.assert_array_equal(
                op(v("c2"), v("c1")).data,
                op(self.data["c2"], self.data["c1"])
            )

    def test_operation_chains(self):
        """Test a sequence of operations."""
        self._generate_data()
        v = self.manager.variable

        c1 = self.data["c1"]
        c2 = self.data["c2"]
        ans = c1 * 2 * (c2 + 3) ** 2 - c1 / 5

        np.testing.assert_array_equal(
            ans,
            (v("c1") * 2 * (v("c2") + 3) ** 2 - v("c1") / 5).data
        )

    def test_comparators(self):
        """Test comparators."""
        self._generate_data()
        v = self.manager.variable

        c1 = self.data["c1"]
        c2 = self.data["c2"]

        np.testing.assert_array_equal(
            _nan_comparison(c1, 0.3, operator.lt),
            (v("c1") < 0.3).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(c1, (0.1 * c2), operator.gt),
            (v("c1") > (0.1 * v("c2"))).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(c2, 0.3, operator.ge),
            (v("c2") >= 0.3).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(c2, 0.3, operator.le),
            (v("c2") <= 0.3).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(0.3, c2, operator.le),
            (0.3 <= v("c2")).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(1, self.data["d1"], operator.eq),
            (v("d1") == 1.0).data
        )

        np.testing.assert_array_equal(
            _nan_comparison(1, self.data["d1"], operator.ne),
            (v("d1") != 1.0).data
        )

    def test_bad_pow(self):
        """Test power with another variable (not authorized)."""
        self._generate_data()
        v = self.manager.variable
        with self.assertRaises(TypeError):
            v("c1") ** v("c2")

    def test_invert(self):
        """Test inverting a variable."""
        self._generate_data()
        v = self.manager.variable

        ans = np.empty(100, dtype=float)
        d1 = self.data["d1"]
        for i in range(d1.shape[0]):
            if np.isnan(d1[i]):
                ans[i] = np.nan
            else:
                ans[i] = 0 if (d1[i] == 1) else 1

        np.testing.assert_array_equal(
            ans,
            (~v("d1")).data
        )

    def test_invert_not_discrete(self):
        """Test inverting a continuous variable."""
        self._generate_data()
        v = self.manager.variable
        with self.assertRaises(TypeError):
            ~v("c1")

    def test_log(self):
        """Test the log function."""
        self._generate_data()
        v = self.manager.variable
        c1 = self.data["c1"]
        np.testing.assert_array_equal(np.log(c1), v("c1").log().data)

    def test_mean(self):
        """Test the mean function."""
        self._generate_data()
        v = self.manager.variable
        c1 = self.data["c1"]
        np.testing.assert_array_equal(np.nanmean(c1), v("c1").mean().data)

    def test_std(self):
        """Test the std function."""
        self._generate_data()
        v = self.manager.variable
        c1 = self.data["c1"]
        np.testing.assert_array_equal(np.nanstd(c1), v("c1").std().data)

    def test_z_score(self):
        """Test computing the z-score."""
        self._generate_data()
        v = self.manager.variable
        c1 = self.data["c1"]

        z = (c1 - np.nanmean(c1)) / np.nanstd(c1)
        np.testing.assert_array_equal(
            z,
            (v("c1") - v("c1").mean()) / v("c1").std()
        )

    def test_special_functions_discrete(self):
        """Test computing special functions on discrete variables
        (not allowed).
        """
        self._generate_data()
        v = self.manager.variable

        with self.assertRaises(TypeError):
            v("d1").log()

        with self.assertRaises(TypeError):
            v("d1").mean()

        with self.assertRaises(TypeError):
            v("d1").std()
