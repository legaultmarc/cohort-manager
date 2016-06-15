import unittest

import numpy as np
import pandas as pd

from .. import types

from . import test_inference
from . import datasets


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

    def test_checks_pass(self):
        """Test that valid type checks pass."""
        for t, values in datasets.types.items():
            if t == "factor":
                mapping = [(1, "a"), (2, "b"), (3, "c")]
                types.type_str(t).check(values, mapping)
            else:
                types.type_str(t).check(values)

    def test_checks_fail(self):
        """Test that bad type checks fail."""
        # discrete
        for bad_type in ("continuous", "integer", "positiveinteger",
                         "negativeinteger", "year", "date", "pastdate"):
            with self.assertRaises(types.InvalidValues):
                types.Discrete.check(datasets.types[bad_type])

        # continuous
        for bad_type in ("date", "pastdate"):
            with self.assertRaises(types.InvalidValues):
                types.Continuous.check(datasets.types[bad_type])

        # integer
        for bad_type in ("date", "pastdate", "continuous"):
            with self.assertRaises(types.InvalidValues):
                types.Integer.check(datasets.types[bad_type])

        # positiveint and year
        for bad_type in ("continuous", "integer", "negativeinteger", "date",
                         "pastdate"):
            with self.assertRaises(types.InvalidValues):
                types.PositiveInteger.check(datasets.types[bad_type])
                types.Year.check(datasets.types[bad_type])

        # negativeint
        for bad_type in ("continuous", "integer", "positiveinteger", "date",
                         "pastdate"):
            with self.assertRaises(types.InvalidValues):
                types.NegativeInteger.check(datasets.types[bad_type])

        # factor
        mapping = [(1, "a"), (2, "b"), (3, "c")]
        for bad_type in ("continuous", "discrete", "integer",
                         "positiveinteger", "negativeinteger", "year",
                         "date"):
            with self.assertRaises(types.InvalidValues):
                types.Factor.check(datasets.types[bad_type], mapping)

        # date
        for bad_type in ("continuous", "discrete", "integer",
                         "positiveinteger", "negativeinteger", "year",
                         "factor"):
            with self.assertRaises(types.InvalidValues):
                types.Date.check(datasets.types[bad_type])

        # pastdate
        for bad_type in ("continuous", "discrete", "integer",
                         "positiveinteger", "negativeinteger", "year",
                         "factor", "date"):
            with self.assertRaises(types.InvalidValues):
                types.PastDate.check(datasets.types[bad_type])

    def test_subtype_of(self):
        """Test the subtype hierarchy checks."""
        # Test direct relationship.
        self.assertTrue(types.Discrete.subtype_of(types.Type))
        self.assertTrue(types.Continuous.subtype_of(types.Type))
        self.assertTrue(types.Integer.subtype_of(types.Continuous))
        self.assertTrue(types.PositiveInteger.subtype_of(types.Integer))
        self.assertTrue(types.Year.subtype_of(types.PositiveInteger))
        self.assertTrue(types.NegativeInteger.subtype_of(types.Integer))
        self.assertTrue(types.Factor.subtype_of(types.Type))
        self.assertTrue(types.Date.subtype_of(types.Type))
        self.assertTrue(types.PastDate.subtype_of(types.Date))

        # Test transitive relationships.
        self.assertTrue(types.PositiveInteger.subtype_of(types.Continuous))
        self.assertTrue(types.NegativeInteger.subtype_of(types.Continuous))
        self.assertTrue(types.Year.subtype_of(types.Continuous))

    def test_date_encode(self):
        """Test the date encoding."""
        values = list(datasets.types["date"])
        expected = []
        for d in values:
            # Parse a datetime object.
            date = types.Date._parse_date(d)

            # Compute the expected representation.
            expected.append(int(date.strftime("%Y%m%d")))

        expected = np.array(expected, dtype=np.float)
        np.testing.assert_array_equal(expected, types.Date.encode(values))

    def test_date_decode(self):
        """Test the date decoding."""
        values = list(datasets.types["date"])
        encoded = types.Date.encode(values)
        decoded = types.Date.decode(encoded)

        answer = pd.to_datetime(
            pd.Series([types.Date._parse_date(d) for d in values])
        )
        pd.util.testing.assert_series_equal(answer, decoded)

    def test_type_str(self):
        """Test the string to Type conversion."""
        answers = {
            "discrete": types.Discrete,
            "continuous": types.Continuous,
            "integer": types.Integer,
            "positiveinteger": types.PositiveInteger,
            "year": types.Year,
            "negativeinteger": types.NegativeInteger,
            "factor": types.Factor,
            "date": types.Date,
            "pastdate": types.PastDate,
        }

        for s, _type in answers.items():
            self.assertTrue(types.type_str(s) is _type)
