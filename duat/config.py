# -*- coding: UTF-8 -*-
"""Model OSIRIS configuration files."""

import copy
from itertools import product

import numpy as np

from duat.common import ifd, logger


def val_to_fortran(val):
    """
    Transform a value to Fortran code.

    Args:
        val: The value to translate.

    Returns: 
        str: The Fortran code.

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

    Returns:
        str: A string which assigns the value to the parameter.

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
    """Something behaving like a set of sections."""

    def __str__(self):
        return self.to_fortran()

    def to_fortran(self):
        raise NotImplementedError()

    @classmethod
    def get_structure(cls, offset=0):
        """
        Get a string representing the structure of the class.

        This methods returns a string with lines in the format "- index (type)", where index is the one used to access
        a subitem and type is the class of that item.

        Args:
            offset: Two spaces times this number will be used to indent the returned string. This is used to generate a
                    multi-level description.

        Returns:
            (str): A representation of the structure of the class.

        """
        return ""


class Section(MetaSection):
    """The class defining a configuration block."""

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


# TODO: Add a class derived from Section with prefixed parameters.


class SectionList(MetaSection):
    """
    Class defining a list of sections in a numerical order.

    Here 'section' refers to any subclass of `MetaSection`.
    
    """

    default_type = Section

    @classmethod
    def get_structure(cls, offset=0):
        s = ""
        t = Section if cls.default_type is None else cls.default_type
        s += " " * (offset * 2) + "- 0 (%s)\n" % (t.__name__,)
        s += t.get_structure(offset + 1)
        s += " " * (offset * 2) + "- 1 ...\n"
        return s

    def __init__(self, label=None, lst=None, default_type=None):
        """
        
        Args:
            label (str): a label inserted as a comment when generating fortran code.
            lst (list of MetaSection): a preexisting list of sections conforming this one. 
            default_type (str or class): the default type for implicit creation of content. If a str, a Section with the
                                         provided name will be created. Otherwise, a instance of the provided class with
                                         no parameters will be created.
        """
        self.label = label if label else ""
        self.lst = lst if lst else []
        if default_type is not None:
            self.default_type = default_type
        else:
            self.default_type = SectionList.default_type

    def __getitem__(self, ind):
        if ind < len(self.lst):
            return self.lst[ind]
        if self.default_type == Section:
            raise ValueError("A subsection cannot be implicitly added to the list due to generic default type.")
        if ind > len(self.lst):
            logger.warning("Implicitly creating more than one section in a list.")
        for i in range(len(self.lst), ind + 1):
            if isinstance(self.default_type, str):
                self.append_section(Section(self.default_type))
            else:
                # Note default_type is not Section here, no reason to expect incorrect call arguments in general.
                # Code analyzers may warn though.
                self.append_section(self.default_type())
        return self.lst[ind]

    def __setitem__(self, ind, value):
        if ind < len(self.lst):
            self.lst[ind] = value
        elif ind == len(self.lst):
            self.lst.append(value)
        else:
            if self.default_type is None:
                raise ValueError("A subsection cannot be implicitly added to the list due to unknown type.")
            logger.warning("Implicitly added subsection(s).")
            for i in range(len(self.lst), ind):
                if isinstance(self.default_type, str):
                    self.append_section(Section(self.default_type))
                else:
                    # Note default_type is not Section here, no reason to expect incorrect call arguments in general.
                    # Code analyzers may warn though.
                    self.append_section(self.default_type())
            self.lst.append(value)

    def __iter__(self):
        for x in self.lst:
            yield x

    def __len__(self):
        return len(self.lst)

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

    order = None
    types = None

    @classmethod
    def get_structure(cls, offset=0):
        s = ""
        for x in cls.order:
            t = cls.types[x]
            s += " " * (offset * 2) + "- %s (%s)\n" % (x, t.__name__)

            s += t.get_structure(offset + 1)

        return s

    def __init__(self, label=None, order=None, fixed=True, types=None):
        self.label = label if label else ""
        if not order:
            if fixed:
                logger.warning("A ConfigSectionOrdered instance with no order defined cannot be fixed.")
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
            raise ValueError("Subsection %s cannot be implicitly created due to unknown type." % name)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.order[key]
        self.set_section(key, value)

    def __iter__(self):
        for x in self.order:
            yield x

    def set_section(self, name, section=None):
        """ Add or replace a section."""
        if section is None:  # No section was given
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
    """Set of sections defining a species."""

    # Keep the default order as a class variable, but copy to the instance to allow modification
    order = ["species", "profile", "spe_bound", "diag_species"]
    types = {"species": Section, "profile": Section, "spe_bound": Section, "diag_species": Section}

    def __init__(self, label=None, dim=None):
        if isinstance(label, int):
            label = "Configuration for species " + str(label)
        SectionOrdered.__init__(self, label=label, order=Species.order, fixed=True, types=Species.types)

        # TODO: Move initialization to ConfigFileClass
        if dim is not None:
            self.set_section("species",
                             Section("species",
                                     {"num_par_max": 2048, "rqm": -1.0, "num_par_x": [2] * dim, "vth": [0.0] * 3,
                                      "vfl": [0.0] * 3}))

            default_profile = ifd(dim, {"fx": [[1., 1., 1., 1., 1., 1.]],
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
                                     {"type": ifd(dim, [[0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]])}))

            self.set_section("diag_species", Section("diag_species"))


class Cathode(SectionOrdered):
    """Set of sections defining a cathode."""

    # Keep the default order as a class variable, but copy to the instance to allow modification
    order = ["cathode", "species", "spe_bound", "diag_species"]
    types = {"cathode": Section, "species": Section, "spe_bound": Section, "diag_species": Section}

    def __init__(self, label=None):
        if isinstance(label, int):
            label = "Configuration for cathode " + str(label)
        SectionOrdered.__init__(self, label=label, order=Cathode.order, fixed=True, types=Cathode.types)


class Neutral(SectionOrdered):
    """Set of sections defining a neutral."""

    # Keep the default order as a class variable, but copy to the instance to allow modification
    order = ["neutral", "profile", "diag_neutral", "species", "spe_bound", "diag_species"]
    types = {"neutral": Section, "profile": Section, "diag_neutral": Section, "species": Section, "spe_bound": Section,
             "diag_species": Section}

    def __init__(self, label=None):
        if isinstance(label, int):
            label = "Configuration for neutral " + str(label)
        SectionOrdered.__init__(self, label=label, order=Neutral.order, fixed=True, types=Neutral.types)


# Types of lists of sections

class SpeciesList(SectionList):
    """List of sections defining the species"""
    default_type = Species

    def __init__(self, label="species"):
        SectionList.__init__(self, label=label)


class CathodeList(SectionList):
    """List of sections defining the cathodes"""
    default_type = Cathode

    def __init__(self, label="cathodes"):
        SectionList.__init__(self, label=label)


class NeutralList(SectionList):
    """List of sections defining the neutrals"""
    default_type = Neutral

    def __init__(self, label="neutrals"):
        SectionList.__init__(self, label=label)


class NeutralMovIonsList(SectionList):
    """List of sections defining the neutral moving ions"""

    def __init__(self, label="neutral moving ions"):
        # TODO: Write me
        raise NotImplementedError("Neutral moving ions are not yet implemented")
        # SectionList.__init__(self, label=label, default_type="neutral_mov_ions")


class ZpulseList(SectionList):
    """List of sections defining the laser pulses"""

    def __init__(self, label="zpulses"):
        SectionList.__init__(self, label=label, default_type="zpulse")


# Node sections


class SmoothCurrent(Section):
    def __init__(self, name):
        # Overwrite name with smooth
        Section.__init__(self, "smooth")


# Global input file model


class ConfigFile(SectionOrdered):
    """
    Set of Sections defining an input file.
    """
    order = ["simulation", "node_conf", "grid", "time_step", "restart", "space", "time", "el_mag_fld", "emf_bound",
             "smooth", "diag_emf", "particles", "species_list", "cathode_list", "neutral_list", "neutral_mov_ions_list",
             "collisions", "zpulse_list", "current", "smooth_current"]

    # TODO: Add antenna

    types = {"simulation": Section, "node_conf": Section, "grid": Section, "time_step": Section, "restart": Section,
             "space": Section, "time": Section, "el_mag_fld": Section, "emf_bound": Section, "smooth": Section,
             "diag_emf": Section, "particles": Section, "species_list": SpeciesList, "cathode_list": SpeciesList,
             "neutral_list": NeutralList, "neutral_mov_ions_list": NeutralMovIonsList, "collisions": Section,
             "zpulse_list": ZpulseList, "current": Section, "smooth_current": SmoothCurrent}

    def __init__(self, d):
        """
        Create a default d-dimensional config file.

        Args:
            d(int): The number of dimensions (1, 2 or 3).
            
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

        self["species_list"] = SpeciesList(label="Species configuration")

        for i in [1, 2]:
            self["species_list"].append_section(Species(i, dim=d))

    def _update_particles(self):
        """Update the particles Section with the currently set data"""
        self["particles"]["num_species"] = len(self["species_list"]) if "species_list" in self.subsections else 0
        self["particles"]["num_cathode"] = len(self["cathode_list"]) if "cathode_list" in self.subsections else 0
        self["particles"]["num_neutral"] = len(self["neutral_list"]) if "neutral_list" in self.subsections else 0
        self["particles"]["num_neutral_mov_ions"] = len(
            self["neutral_mov_ions_list"]) if "neutral_mov_ions_list" in self.subsections else 0

    def to_fortran(self):
        self._update_particles()
        return SectionOrdered.to_fortran(self)

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
        Get the dimension of the configuration filed.

        Returns:
            int: The dimension according to the mandatory xmax parameter in the space section.
            
        """
        x_max = self["space"]["xmax"]
        return len(x_max) if isinstance(x_max, list) else 1


class Variation:
    """
    Represents a variation of a set of parameters in config files.
    
    Each parameter varies in list of values. The configuration files produced by this class take into account all
    combinations of values, i.e., the parameter space is given by the cartesian product.
    
    """

    def __init__(self, *args, epilog=None):
        """
        Create a Variation with the given parameters and values.
        
        Args:
            *args (2-:obj:`tuple` of :obj:`list`): Each argument must be a 2-tuple whose first elements is a list of str
                               or int which identifies the parameter in its section and a list of the values the
                               parameter will take. The list can be None to perform no action while passing the
                               parameter to the epilog function (see below).
            epilog (callable): A function of two arguments that will be called with the simulation and the list of
                               parameters when a config is being generated. This can be used for advanced modification,
                               for example, to set two parameters to a related value (like two species temperature).
        """
        self.parameters = args
        self._par_names = ["[dummy]" if p[0] is None else p[0][-1] for p in self.parameters]
        self.len_list = [len(p[1]) for p in self.parameters]
        self.epilog = epilog
        if epilog is None and all(p[0] is None for p in self.parameters):
            logger.warning("Trivial Variation generated. Did you forget an epilog?")

    def __repr__(self):
        return "Variation<%s (%s)>" % (" x ".join(self._par_names), "x".join([str(x) for x in self.len_list]))

    def get_generator(self, config):
        """
        Get a generator that produces ConfigFile objects following the Variation.
        
        Args:
            config (ConfigFile): The configuration where the Variation will be applied.

        Returns:
            generator: A generator which provides the ConfigFile instances.

        """
        paths = [p[0] for p in self.parameters]
        values = [p[1] for p in self.parameters]

        def gen():
            for value in product(*values):  # Value is multidimensional in general
                c = copy.deepcopy(config)
                for i, path in enumerate(paths):  # For the i-th thing to change
                    if path:  # If path is not None (non dummy parameter)
                        place = c  # Start from the root
                        for level in path[:-1]:  # ... access the leaf...
                            place = place[level]
                        place.set_par(path[-1], value[i])  # ... changing its value
                if self.epilog:
                    self.epilog(c, value)
                yield c

        return gen()

    def get_parameter_list(self):
        """
        Get a list with the parameter values in the same order that
        :func:`~duat.osiris.config.Variation.get_generator`.
        
        This method might be useful to post-process the results if the parameter space is simple.

        Returns:
            :obj:`list` of `tuple`: A list with the values of the parameters in the cartesian product order.

        """
        values = [p[1] for p in self.parameters]
        return list(product(*values))
