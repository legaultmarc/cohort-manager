import unittest
import logging

logging.disable(logging.CRITICAL)
test_suite = unittest.defaultTestLoader.discover(__name__)
