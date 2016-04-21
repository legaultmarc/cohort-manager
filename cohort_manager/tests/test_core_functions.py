import shutil
import unittest

import numpy as np

from .. import core


class TestManager(unittest.TestCase):
    def test_vector_map_int(self):
        """Test remapping of int vector to float with missing."""
        data = np.array([1, 1, 2, 1, 2, 1, 2, 2, 1, -9, 1, 2, -9])
        remap = core.vector_map(data, [(2, 0.0), (-9, np.nan)])
        np.testing.assert_array_equal(
            remap,
            np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, np.nan, 1, 0, np.nan])
        )

    def test_vector_map_float(self):
        """Test remapping of float vector to int (not accepted)."""
        data = np.array([1, 1, 2, 1, 2, 2, 1, 1], dtype=float)
        with self.assertRaises(TypeError):
            core.vector_map(data, {2: 1, 1: 0})

    def test_vector_map_float_2_float(self):
        """Test remapping of with NaN in key (not accepted)."""
        data = np.array([1, np.nan, 2, 1, 2, np.nan, 1, 1])
        with self.assertRaises(TypeError):
            core.vector_map(data, {np.nan: -9.0, 2: 1.0, 1: 0.0})

    def test_vector_map_str(self):
        """Test remapping to str."""
        data = np.array([1, 2, 2, 1, 2])
        np.testing.assert_array_equal(
            core.vector_map(data, [(1, "a"), (2, "b")]),
            np.array(["a", "b", "b", "a", "b"])
        )

    def test_vector_map_from_str(self):
        """Test remapping from str (not accepted)."""
        data = np.array(["a", "b", "b", "a", "a"])
        with self.assertRaises(TypeError):
            core.vector_map(data, {"a": 1, "b": 2}),

    def test_vector_map_transitive(self):
        """Test transitive remapping."""
        data = np.array([0, 1, -9, 1, 0, -9, 1, 0, 1])
        np.testing.assert_array_equal(
            core.vector_map(data, [(-9, 0), (0, 2), (1, 2)]),
            np.array([2, 2, 0, 2, 2, 0, 2, 2, 2])
        )

    def test_vector_map_ambiguous_target(self):
        """Test vector remapping with ambiguous target dtype."""
        data = np.array([0, 1, 2, 3, 1, 0])
        with self.assertRaises(TypeError):
            core.vector_map(data, [(0, -9.0), (1, 1)])

    def test_permutation(self):
        """Test the Permutation class (core logic)."""
        manager = core.CohortManager("_TestManager")
        manager.set_samples(list("abcdef"))
        manager.add_phenotype(name="p1", variable_type="continuous")
        manager.add_data("p1", list(range(6)))

        v1 = manager.get_data("p1")
        perm = core.Permutation(manager, ["f", "d", "e", "b", "c", "a"])
        v2 = perm.get_data("p1")

        perm = core.Permutation(manager, ["a", "c"], allow_subset=True)
        v3 = perm.get_data("p1")

        shutil.rmtree("_TestManager")

        np.testing.assert_array_equal(
            v1,
            #         a  b  c  d  e  f
            np.array([0, 1, 2, 3, 4, 5])
        )

        np.testing.assert_array_equal(
            v2,
            np.array([5, 3, 4, 1, 2, 0])
        )

        np.testing.assert_array_equal(
            v3,
            np.array([0, 2])
        )

    def test_permutation_subset(self):
        """Test implicit subsetting in Permutation (not accepted)."""
        manager = core.CohortManager("_TestManager")
        manager.set_samples(list("abcdef"))
        manager.add_phenotype(name="p1", variable_type="continuous")
        manager.add_data("p1", list(range(6)))

        with self.assertRaises(ValueError):
            # Missing "e" and not subset=True
            core.Permutation(manager, ["f", "d", "b", "c", "a"])

        shutil.rmtree("_TestManager")

    def test_permutation_extra(self):
        """Test unknown samples in Permutation (not accepted)."""
        manager = core.CohortManager("_TestManager")
        manager.set_samples(list("abcdef"))
        manager.add_phenotype(name="p1", variable_type="continuous")
        manager.add_data("p1", list(range(6)))

        with self.assertRaises(ValueError):
            # z is unknown.
            core.Permutation(manager, ["f", "d", "z", "c", "a"])

        shutil.rmtree("_TestManager")
