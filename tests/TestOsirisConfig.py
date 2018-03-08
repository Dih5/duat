#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestOsirisConfig.py: Tests for the `osiris.config` module.
"""

import unittest

from duat.config import *


class TestOsirisConfig(unittest.TestCase):
    def test_val_to_fortran(self):
        self.assertEqual(".true.", val_to_fortran(True))
        self.assertEqual(".false.", val_to_fortran(False))
        self.assertEqual('""', val_to_fortran(""))
        self.assertEqual("2", val_to_fortran(2))
        self.assertEqual("-2", val_to_fortran(-2))
        self.assertEqual("2", val_to_fortran(2.))
        self.assertEqual("2.1", val_to_fortran(2.1))
        self.assertEqual("0.3", val_to_fortran(.3))
        self.assertEqual("2", val_to_fortran(np.array([2.0])[0]))
        self.assertEqual("9.87435d-07", val_to_fortran(.0000009874345))
        self.assertEqual("-9.87435d-07", val_to_fortran(-.0000009874345))

    def test_get_d(self):
        for d in [1, 2, 3]:
            self.assertEqual(d, ConfigFile(d).get_d())

    def test_None_assign_deletes(self):
        """Check a None assignment deletes parameters and sections"""
        sim = ConfigFile(1)
        self.assertFalse("diag_emf" in sim)
        sim["diag_emf"]["reports"] = ["e1"]  # Implicit creation
        self.assertTrue("diag_emf" in sim)
        self.assertTrue("reports" in sim["diag_emf"])
        sim["diag_emf"]["reports"] = None  # Remove a parameter
        self.assertFalse("reports" in sim["diag_emf"])
        sim["diag_emf"] = None  # Remove the section
        self.assertFalse("diag_emf" in sim)


if __name__ == "__main__":
    unittest.main()
