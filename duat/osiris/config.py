# -*- coding: UTF-8 -*-
"""Model OSIRIS configuration files."""
from __future__ import print_function

import numpy as np
import sys

from ..common import ifd


def val_to_fortran(val):
    """
    Transform a value to fortran code.

    Args:
        val:

    Returns:

    """
    # Convert numpy types
    if isinstance(val, np.generic):
        val = val.item()

    t = type(val)
    if t is bool:
        return ".true." if val else ".false."
    elif t is int:
        return str(val)
    elif t is float:
        return ("%.6g" % val).replace("e", "d")
    elif t is str:
        return '"' + val + '"'
    raise TypeError("Unknown type: " + t)


def par_to_fortran(name, val):
    """
    Transform a parameter to fortran code.

    Args:
        name (str): The name of the parameter.
        val: The value of the parameter.

    Returns: A string which assigns the value to the parameter

    """
    # Convert numpy types
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, np.ndarray):
        val = val.tolist()

    t = type(val)
    if t in [bool, int, float, str]:
        return name + " = " + val_to_fortran(val) + ","
    elif t in [list, tuple]:  # 1D or 2D or 3D

        if type(val[0]) in [list, tuple]:  # 2D or 3D
            if type(val[0][0]) in [list, tuple]:  # 3D
                s = ""
                for i, row in enumerate(val):
                    for j, row_row in enumerate(row):
                        s += name + "(1:" + str(len(row_row)) + ", " + str(i + 1) + ", " + str(
                            j + 1) + ") = " + ", ".join(
                            list(map(val_to_fortran, row_row))) + ",\n  "
                return s
            else:  # 2D
                s = ""
                for i, row in enumerate(val):
                    s += name + "(1:" + str(len(row)) + ", " + str(i + 1) + ") = " + ", ".join(
                        list(map(val_to_fortran, row))) + ",\n  "
                return s
        else:  # 1D
            l = len(val)
            return name + "(1:" + str(l) + ") = " + ", ".join(list(map(val_to_fortran, val))) + ","
    raise TypeError


class MetaSection:
    """
    Something behaving like a set of sections
    """

    def __str__(self):
        return self.to_fortran()

    def to_fortran(self):
        raise NotImplementedError()


class Section(MetaSection):
    """
    The class defining a configuration block
    """

    def __init__(self, name, param=None):
        self.name = name
        if param:
            self.pars = param.copy()
        else:
            self.pars = {}

    def __getitem__(self, ind):
        return self.pars[ind]

    def __setitem__(self, key, value):
        self.pars[key] = value

    def set_par(self, name, val):
        """Add or update the value of a parameter"""
        self.pars[name] = val

    def set_pars(self, **kwargs):
        """Add or update some parameters using keyword arguments"""
        for key in kwargs:
            self.pars[key] = kwargs[key]

    def to_fortran(self):
        s = self.name + "\n{\n"
        for p in self.pars:
            s += "  " + par_to_fortran(p, self.pars[p]) + "\n"
        s += "}\n"
        return s


# TODO: Add a class derived from ConfigSection with prefixed parameters.


class SectionList(MetaSection):
    """
    Class defining a list of sections in a numerical order.

    Here 'section' refers to any subclass of `MetaSection`.
    """

    def __init__(self, label=None, lst=None):
        self.label = label if label else ""
        self.lst = lst if lst else []

    def __getitem__(self, ind):
        return self.lst[ind]

    def append_section(self, section):
        self.lst.append(section)

    def to_fortran(self):
        s = ("!---" + self.label + "\n") if self.label else ""
        for sec in self.lst:
            s += sec.to_fortran() + "\n"
        return s


class SectionOrdered(MetaSection):
    """
    Class defining a set of sections that must be outputted in a particular order given by a keyword related to them.

    Here 'section' refers to any subclass of `MetaSection`.
    """

    def __init__(self, label=None, order=None, fixed=True, types=None):
        self.label = label if label else ""
        if not order:
            if fixed:
                print("A ConfigSectionOrdered instance with no order defined cannot be fixed.", file=sys.stderr)
            self.fixed = False
            self.order = []
        else:
            self.fixed = fixed
            self.order = order

        self.types = types if types else {}
        self.subsections = {}

    def __getitem__(self, name):
        if isinstance(name, int):
            name = self.order[name]

        if name in self.subsections:
            return self.subsections[name]
        elif name in self.types:
            self.subsections[name] = self.types[name](name)
            return self.subsections[name]
        else:
            print(self.types)
            raise ValueError("Subsection %s cannot be implicitly created due to unknown type." % name)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.order[key]
        self.set_section(key, value)

    def set_section(self, name, section=None):
        """ Add or replace a section"""
        if not section:  # No section was given
            section = Section(name) if "_list" not in name else SectionList(label=name)
        if isinstance(section, str):  # A string was given (abuse of notation)
            section = Section(section) if "_list" not in name else SectionList(label=section)
        if (name in self.subsections) or (name in self.order):
            self.subsections[name] = section
        else:
            if self.fixed:
                raise ValueError("Section %s not expected here. Valid names are %s" % (name, self.order))
            else:
                self.subsections[name] = section
                self.order.append(name)

    def to_fortran(self):
        s = ("!---" + self.label + "\n") if self.label else ""
        for name in self.order:
            if name in self.subsections:
                s += self.subsections[name].to_fortran() + "\n"
        return s


# *****************************************
# Classes describing the semantic structure
# *****************************************


class Species(SectionOrdered):
    """
    Set of sections defining a species
    """

    # Keep the default order as a class variable, but copy to the instance to allow modification
    order = ["species", "profile", "spe_bound", "diag_species"]
    types = {"species": Section, "profile": Section, "spe_bound": Section, "diag_species": Section}

    def __init__(self, d, num=None):
        label = "Configuration for species " + str(num) if num else ""
        SectionOrdered.__init__(self, label=label, order=Species.order, fixed=True, types=Species.types)

        self.set_section("species",
                         Section("species",
                                 {"num_par_max": 2048, "rqm": -1.0, "num_par_x": [2] * d, "vth": [0.1] * 3,
                                        "vfl": [0.0, 0.0, 0.6], "den_min": 1.0e-5, "num_dgam": 0, "dgam": 0}))

        default_profile = ifd(d, {"fx": [[1., 1., 1., 1., 1., 1.]],
                                  "x": [[0., 0.9999, 1.000, 2.000, 2.001, 10000.]]},
                              # 2d
                              {"fx": [[1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.]],
                               "x": [[0., 0.9999, 1.000, 2.000, 2.001, 10000.],
                                     [0., 0.9999, 1.000, 2.000, 2.001, 10000.]]},
                              # 3d
                              {"fx": [[1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.],
                                      [1., 1., 1., 1., 1., 1.]],
                               "x": [[0., 0.9999, 1.000, 2.000, 2.001, 10000.],
                                     [0., 0.9999, 1.000, 2.000, 2.001, 10000.],
                                     [0., 0.9999, 1.000, 2.000, 2.001, 10000.]]}
                              )
        self.set_section("profile", Section("profile", default_profile))

        self.set_section("spe_bound",
                         Section("spe_bound",
                                 {"type": ifd(d, [[0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]])}))

        self.set_section("diag_species", Section("diag_species"))


class ConfigFile(SectionOrdered):
    """
    Set of Sections defining an input file
    """
    order = ["simulation", "node_conf", "grid", "time_step", "restart", "space", "time", "el_mag_fld", "emf_bound",
             "smooth", "diag_emf", "particles", "species_list", "cathode_list", "neutral_list", "neutral_mov_ions_list",
             "zpulse_list", "current", "smooth_current"]

    # FIXME: There are two "smooths" in the config file definition. I have renamed the second to smooth_current, but the output form must change

    types = {"simulation": Section, "node_conf": Section, "grid": Section, "time_step": Section, "restart": Section,
             "space": Section, "time": Section, "el_mag_fld": Section, "emf_bound": Section, "smooth": Section,
             "diag_emf": Section, "particles": Section, "species_list": SectionList, "cathode_list": SectionList,
             "neutral_list": SectionList, "neutral_mov_ions_list": SectionList, "zpulse_list": SectionList,
             "current": Section, "smooth_current": Section}

    def __init__(self, d):
        """
        Create a default d-dimensional config file

        Args:
            d(int): The number of dimensions (1, 2 or 3)
        """
        SectionOrdered.__init__(self, order=ConfigFile.order, fixed=True, types=ConfigFile.types)

        if d not in [1, 2, 3]:
            raise TypeError("Invalid dimension")

        self["node_conf"] = Section("node_conf", {"node_number": [1] * d, "if_periodic": [True] * d})

        self["grid"] = Section("grid",
                               {"coordinates": "cartesian", "nx_p": ifd(d, [1024], [32, 32], [10, 10, 10])})

        self["time_step"] = Section("time_step", {"dt": ifd(d, 0.07, 0.07, 0.05), "ndump": 10})

        self["space"] = Section("space", {"xmin": [0.0] * d, "xmax": ifd(d, [102.4], [3.2, 3.2], [1.0, 1.0, 1.0]),
                                                "if_move": [False] * d})

        self["time"] = Section("time", {"tmin": 0.0, "tmax": 7.0})

        self["emf_bound"] = Section("emf_bound",
                                    {"type": ifd(d, [[0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]])})

        self["particles"] = Section("particles", {"num_species": 2})

        self["species_list"] = SectionList(label="Species configuration")
        for i in [1, 2]:
            self["species_list"].append_section(Species(d, i))

        # self["pulse_sequence"] = ConfigSection("pulse_sequence", {"num_pulses": 0})

    def write(self, path):
        """
        Save the config file to the specified path.

        Args:
            path: the path of the output file.

        Raises:

        """
        with open(path, "w") as f:
            f.write(self.to_fortran())

    def get_d(self):
        """
        Get the dimension of the configuration filed

        Returns:
            (int): The dimension according to the mandatory xmax parameter in the space section.
        """
        x_max = self["space"]["xmax"]
        return len(x_max) if isinstance(x_max, list) else 1
