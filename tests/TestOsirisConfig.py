#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestOsirisConfig.py: Tests for the `osiris.config` module.
"""

import unittest

import re

from duat import config

import numpy as np


def _remove_comments(s):
    """Remove fortran comments from string"""
    return "\n".join(filter(lambda s2: not re.match("!--.*", s2), s.split("\n")))


class TestOsirisConfig(unittest.TestCase):
    def test_val_to_fortran(self):
        self.assertEqual(".true.", config._val_to_fortran(True))
        self.assertEqual(".false.", config._val_to_fortran(False))
        self.assertEqual('""', config._val_to_fortran(""))
        self.assertEqual("2", config._val_to_fortran(2))
        self.assertEqual("-2", config._val_to_fortran(-2))
        self.assertEqual("2", config._val_to_fortran(2.))
        self.assertEqual("2.1", config._val_to_fortran(2.1))
        self.assertEqual("0.3", config._val_to_fortran(.3))
        self.assertEqual("2", config._val_to_fortran(np.array([2.0])[0]))
        self.assertEqual("9.87435d-07", config._val_to_fortran(.0000009874345))
        self.assertEqual("-9.87435d-07", config._val_to_fortran(-.0000009874345))

    def test_get_d(self):
        for d in [1, 2, 3]:
            self.assertEqual(d, config.ConfigFile(d).get_d())

    def test_None_assign_deletes(self):
        """Check a None assignment deletes parameters and sections"""
        sim = config.ConfigFile(1)
        self.assertFalse("diag_emf" in sim)
        sim["diag_emf"]["reports"] = ["e1"]  # Implicit creation
        self.assertTrue("diag_emf" in sim)
        self.assertTrue("reports" in sim["diag_emf"])
        sim["diag_emf"]["reports"] = None  # Remove a parameter
        self.assertFalse("reports" in sim["diag_emf"])
        sim["diag_emf"] = None  # Remove the section
        self.assertFalse("diag_emf" in sim)

    def test_from_string(self):
        """Check the same config file is obtained from its fortran code"""
        s = config.ConfigFile(d=1, template="default").to_fortran()
        s2 = config.ConfigFile.from_string(s).to_fortran()
        # TODO: The order of the parameters in the output is not defined.
        # A comparison of ConfigFile instances should be implemented instead
        self.assertEqual(_remove_comments(s), _remove_comments(s2))


if __name__ == "__main__":
    unittest.main()
