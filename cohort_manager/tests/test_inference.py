import unittest
import random

import numpy as np
import scipy.stats

from .. import inference


class TestInference(unittest.TestCase):
    def test_estimate_num_distinct(self):
        """Test the estimation of the number of distinct elements."""
        n = inference.estimate_num_distinct(np.array([1, 1, 2, 3, 1]))
        self.assertEqual(n, 3)

        # Generate a bool vector.
        v = _generate_discrete()

        self.assertEqual(inference.estimate_num_distinct(v), 2)

        v = np.array(["a", "b", "a", "c", "b", "c"])
        self.assertEqual(inference.estimate_num_distinct(v), 3)

    def test_infer_primitive_type(self):
        """Test inference of primitive types."""
        cases = {
            "": "null",
            "1956": "past_year",
            "3": "positive_integer",
            "-3": "negative_integer",
            "0": "zero",
            "3.1415": "real",
            "2015-02-22": "date",
            "wassup12": "string"
        }

        for case, _type in cases.items():
            self.assertEqual(inference.infer_primitive_type(case), _type)

    def test_infer_type(self):
        v = [str(random.randint(1, 1000) / 10) for i in range(10)]
        cases = {
            tuple(v): "continuous",
            ("1", "0", "NA", "0", "1", "0", "1", "0"): "discrete",
            ("2", "NA", "5", "", "5", "2", "NA", "5"): "discrete",
            ("1990", "2004", "1958", "", "1789", "2000"): "year",
            ("1", "2", "3", "4", "5", "6", "7"): "unique_key",
            ("", "C", "A", "T", "A", "T", "D", "NA", "E", "F"): "text",
        }

        for case, _type in cases.items():
            self.assertEqual(inference.infer_type(list(case)), _type)

    def test_mutual_information_discrete_discrete(self):
        """Test the mutual information between pairs of discrete variables."""

        # Contingency for the joint distribution.
        #
        # +---+---+---+---+---+
        # |   | 0 | 1 | 2 | 3 |
        # +---+---+---+---+---+
        # | 0 | 0 | 4 | 1 | 0 | 5
        # +---+---+---+---+---+
        # | 1 | 1 | 0 | 0 | 8 | 9
        # +---+---+---+---+---+
        # | 2 | 0 | 11| 1 | 0 | 12
        # +---+---+---+---+---+
        # | 3 | 21| 1 | 2 | 1 | 25
        # +---+---+---+---+---+
        #       22  16  4   9   51

        joint = np.array([
            [0, 4, 1, 0],
            [1, 0, 0, 8],
            [0, 11, 1, 0],
            [21, 1, 2, 1],
        ])

        acc = 0
        x = []
        y = []
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                p_xy = joint[i, j] / 51
                p_x = np.sum(joint[i, :]) / 51
                p_y = np.sum(joint[:, j]) / 51

                if p_xy != 0:
                    acc += p_xy * np.log2(p_xy / (p_x * p_y))

                # Build the dataset.
                x.extend([i] * joint[i, j])
                y.extend([j] * joint[i, j])

        x = np.array(x)
        y = np.array(y)

        self.assertEqual(
            acc,
            inference.mutual_information(x, "discrete", y, "discrete",
                                         normalized=False)
        )

    def test_mutual_information_discrete_continuous(self):
        "Test estimation of MI between a continuous and a discrete variable."""
        params = np.array([
            # mu, sigma, n
            [2, 3],
            [3, 2.5],
            [5, 1]
        ])
        n = np.array([1000, 4000, 5000])

        # We have three overlapping continuous distributions for the different
        # values of the discrete variable.
        continuous_a = np.random.normal(*params[0], n[0])
        continuous_b = np.random.normal(*params[1], n[1])
        continuous_c = np.random.normal(*params[2], n[2])
        continuous = np.hstack((continuous_a, continuous_b, continuous_c))

        # p(y|0) ~ N(2, 3)
        # p(y|1) ~ N(3, 2.5)
        # p(y|2) ~ N(5, 1)
        # p(y) ~ \sum_x p(x)p(y|x)
        discrete_a = np.zeros_like(continuous_a)
        discrete_b = np.zeros_like(continuous_b) + 1
        discrete_c = np.zeros_like(continuous_c) + 2
        discrete = np.hstack((discrete_a, discrete_b, discrete_c))

        def _get_mu_sigma(x):
            return params[x]

        def _p(x):
            return n[x] / np.sum(n)

        # Compute the theoretical MI.
        acc = 0
        xs = [0, 1, 2]
        for x in xs:
            def _f(y):
                mu, sigma = _get_mu_sigma(x)

                # This is P(y | x)
                condi_p_y = scipy.stats.norm.pdf(y, mu, sigma)

                # p(y)=\sum_xp(x)p(y|x)
                p_y = 0
                for x2 in xs:
                    mu, sigma = _get_mu_sigma(x2)
                    p_y += _p(x2) * scipy.stats.norm.pdf(y, mu, sigma)

                if condi_p_y == 0 or p_y == 0:
                    return 0

                return _p(x) * condi_p_y * np.log2(condi_p_y / p_y)

            acc += scipy.integrate.quad(_f, -np.inf, np.inf)[0]

        theoretical_mi = max(0, acc)

        percent_error = abs(
            theoretical_mi -
            inference.mutual_information(discrete, "discrete",
                                         continuous, "continuous",
                                         normalized=False)
        ) / theoretical_mi

        # This is highly liberal. I don't know if it is possible to get a
        # better estimate.
        self.assertTrue(percent_error < 0.5)

    def test_mutual_information_side_effect(self):
        """Test that mutual information computation doesn't have a side effect.

        """
        d1 = _generate_discrete()
        d2 = _generate_discrete()

        # Introduce some dependancy.
        c1 = np.zeros_like(d1)
        c1[d1 == 0] = 3 + 0.1 * np.random.random(np.sum(d1 == 0))
        c1[d1 == 1] = 2.3 + 0.1 * np.random.random(np.sum(d1 == 1))
        c1[np.isnan(d1)] = 4 + 0.1 * np.random.random(np.sum(np.isnan(d1)))
        c1[np.random.permutation(np.arange(5500))[:220]] = np.nan

        c2 = np.random.normal(0, 1, 5500)
        c2[np.random.permutation(np.arange(5500))[:300]] = np.nan

        # Copy the original vectors.
        copy_d1 = d1.copy()
        copy_d2 = d2.copy()
        copy_c1 = c1.copy()
        copy_c2 = c2.copy()

        # Test all the possible kinds of MI and check that it did not change
        # the vectors.

        # Discrete discrete.
        inference.mutual_information(d1, "discrete", d2, "discrete")
        np.testing.assert_array_equal(copy_d1, d1)
        np.testing.assert_array_equal(copy_d2, d2)

        # Continuous discrete.
        inference.mutual_information(c1, "continuous", d1, "discrete")
        np.testing.assert_array_equal(copy_c1, c1)
        np.testing.assert_array_equal(copy_d1, d1)

        # Continuous continuous.
        inference.mutual_information(c1, "continuous", c2, "continuous")
        np.testing.assert_array_equal(copy_c1, c1)
        np.testing.assert_array_equal(copy_c2, c2)


def _generate_discrete(n=5500, p=0.5, missing_rate=0.05):
    n_missing = int(missing_rate * n)
    v = np.random.binomial(1, p, n).astype(float)
    v[np.random.permutation(np.arange(n))[:n_missing]] = np.nan
    return v
