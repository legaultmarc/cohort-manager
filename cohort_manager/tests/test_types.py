import unittest

import numpy as np

from .. import types

from . import test_inference


class TestTypes(unittest.TestCase):
    def test_estimate_num_distinct(self):
        """Test the estimation of the number of distinct elements."""
        n = types.estimate_num_distinct(np.array([1, 1, 2, 3, 1]))
        self.assertEqual(n, 3)

        # Generate a bool vector.
        v = test_inference._generate_discrete()

        self.assertEqual(types.estimate_num_distinct(v), 2)

        v = np.array(["a", "b", "a", "c", "b", "c"])
        self.assertEqual(types.estimate_num_distinct(v), 3)
