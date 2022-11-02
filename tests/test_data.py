import unittest

from rockley.utils.data import test


class TestData(unittest.TestCase):
    def test_example(self):
        self.assertEqual(test(), 42)
        self.assertEqual(0, 0)
        self.assertAlmostEqual(0.001, 0, places=2)
        self.assertNotAlmostEqual(0.001, 0, places=3)
        self.assertTrue(True)
        self.assertFalse(False)
