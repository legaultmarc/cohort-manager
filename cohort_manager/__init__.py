try:
    from .version import cohort_manager_version as __version__
except ImportError:
    __version__ = None


def test(verbosity=1):
    import unittest
    from .tests import test_suite

    unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
