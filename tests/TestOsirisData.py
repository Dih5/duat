#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestOsirisData.py: Tests for the `osiris.data` module.
"""

import unittest

from duat.data import *


class TestOsirisData(unittest.TestCase):
    def test_data(self):
        self.assertAlmostEqual(26.9815386, molar_mass("Al"))
        self.assertAlmostEqual(26.9815386, molar_mass(13))
        self.assertAlmostEqual(7.830946949704344e+23, full_ionization_density("Al", 13))
        self.assertAlmostEqual(449.54810753739446, full_ionization_density("Al", 13) / critical_density())


if __name__ == "__main__":
    unittest.main()
