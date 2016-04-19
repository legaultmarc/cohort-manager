import unittest

from . import test_suite

unittest.TextTestRunner(verbosity=2).run(test_suite)
