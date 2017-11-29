# -*- coding: UTF-8 -*-
"""Useful data for PIC simulations."""

# Data from periodictable package (public domain)
_molar_mass_dict = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                    'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                    'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                    'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                    'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                    'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                    'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055,
                    'Pd': 106.42,
                    'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                    'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                    'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                    'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                    'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23,
                    'Ir': 192.217,
                    'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                    'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                    'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                    'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                    'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                    'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294}

# List by atomic number
_molar_mass_list = [x[1] for x in sorted(_molar_mass_dict.items(), key=lambda t: t[1])]

# List of elements (obtained el cheapo way)
_element_names = [x[0] for x in sorted(_molar_mass_dict.items(), key=lambda t: t[1])]

# Added material data
_molar_mass_dict["water"] = 18.01528

# Data from periodictable package (public domain)
_element_densities = dict(
    H=(0.0708, "T=-252.87"),
    He=(0.122, "T=-268.93"),
    Li=0.534,
    Be=1.848,
    B=2.34,
    C=(2.1, "1.9-2.3 (graphite)"),
    N=(0.808, "T=-195.79"),
    O=(1.14, "T=-182.95"),
    F=(1.50, "T=-188.12"),
    Ne=(1.207, "T=-246.08"),
    Na=0.971,
    Mg=1.738,
    Al=2.6989,
    Si=(2.33, "T=25"),
    P=1.82,
    S=2.07,
    Cl=(1.56, "T=-33.6, 0.44 C above boiling point"),
    Ar=(1.40, "T=-185.85"),
    K=0.862,
    Ca=1.55,
    Sc=(2.989, "T=25"),
    Ti=4.54,
    V=(6.11, "T=18.7"),
    Cr=(7.19, "7.18-7.20"),
    Mn=(7.33, "7.21-7.44"),
    Fe=7.874,
    Co=8.9,
    Ni=(8.902, "T=25"),
    Cu=8.96,
    Zn=(7.133, "T=25"),
    Ga=(5.904, "T=29.6"),
    Ge=(5.323, "T=25"),
    As=5.73,
    Se=4.79,
    Br=3.12,
    Kr=(2.16, "T=-153.22"),
    Rb=1.532,
    Sr=2.54,
    Y=(4.469, "T=25"),
    Zr=6.506,
    Nb=8.57,
    Mo=10.22,
    Tc=(11.50, "calculated"),
    Ru=12.41,
    Rh=12.41,
    Pd=12.02,
    Ag=10.50,
    Cd=8.65,
    In=7.31,
    Sn=7.31,
    Sb=6.691,
    Te=6.24,
    I=4.93,
    Xe=(3.52, "T=-108.12"),
    Cs=1.873,
    Ba=3.5,
    La=(6.145, "T=25"),
    Ce=(6.770, "T=25"),
    Pr=6.773,
    Nd=(7.008, "T=25"),
    Pm=(7.264, "T=25"),
    Sm=(7.520, "T=25"),
    Eu=(5.244, "T=25"),
    Gd=(7.901, "T=25"),
    Tb=8.230,
    Dy=(8.551, "T=25"),
    Ho=(8.795, "T=25"),
    Er=(9.066, "T=25"),
    Tm=(9.321, "T=25"),
    Yb=6.966,
    Lu=(9.841, "T=25"),
    Hf=13.31,
    Ta=16.654,
    W=19.3,
    Re=21.02,
    Os=22.57,
    Ir=(22.42, "T=17"),
    Pt=21.45,
    Au=(19.3, "approximate"),
    Hg=13.546,
    Tl=11.85,
    Pb=11.35,
    Bi=9.747,
    Po=9.32,
    Th=11.72,
    Pa=(15.37, "calculated"),
    U=(18.95, "approximate"),
    Np=20.25,
    Pu=(19.84, "T=25"),
    Am=13.67,
    Cm=(13.51, "calculated"),
    Bk=(14, "estimated"),
    water=1,
)


def molar_mass(material):
    """
    Get the molar mass of a material (g/mol).
    
    Data is taken from the periodictable package (public domain).
    
    Args:
        material (str or int): Name of the material (e.g., 'Al' or 'water') or atomic number 

    Returns:
        The molar mass (g/mol).

    """
    if isinstance(material, int):
        material = _element_names[material - 1]

    return _molar_mass_dict[material]


def density(material):
    """
    Get a density (under certain 'normal' conditions) of a material (g/cm^3).

    Data is taken from the periodictable package (public domain). The user should check if the returned density fits
    its application
    
    Args:
        material (str or int): Name of the material (e.g., 'Al' or 'water') or atomic number.

    Returns:
        The density (g/cm^3).

    """
    if isinstance(material, int):
        material = _element_names[material - 1]
    rho = _element_densities[material]
    if isinstance(rho, tuple):
        rho = rho[0]
    return rho


def full_ionization_density(material, z=1):
    """
    Get the full ionization density of a Z (e.g. "Al") when z electrons per atom are ionized.
    
    Args:
        material (str or int): Name of the material (e.g., 'Al' or 'water') or atomic number.
        z (int): Number of electrons ionized per atom (a global factor).

    Returns:
        (float) Full ionization density (particles/cm^3)
        
    """
    return z * 6.022140E23 / molar_mass(material) * density(material)


def critical_density(wavelength=800):
    """
    Get the critical density for a laser with the given wavelength.
    
    Args:
        wavelength: Laser wavelength (in nm)

    Returns:
        (float) Critical density (particles/cm^3)
    """
    # From the SI formula
    # epsilon_0*electron mass/(electron charge)^2*(2*pi*c/(wavelength))^2/cm^-3
    return 1.11485422E27 / wavelength ** 2
