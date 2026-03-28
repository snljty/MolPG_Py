#! /usr/bin/env python3
# -*- Coding: UTF-8 -*-

r"""
This program written by python is aimming to
determine the point group of a molecule.

It reads a structure from a xyz file.
"""

import numpy as np
import os, sys
from scipy.spatial.distance import cdist
import inspect, IPython
# IPython.embed(header=f"Debug: at line {inspect.currentframe().f_lineno:d} of file {os.path.split(__file__)[-1]:s}:")
# from functools import reduce

ncoords = 3

np.set_printoptions(precision=7, suppress=True, formatter={"float": "{: 0.7f}".format})

elements_list = [None, 
 "H" , "He", "Li", "Be", "B" , "C" , "N" , "O" ,
 "F" , "Ne", "Na", "Mg", "Al", "Si", "P" , "S" ,
 "Cl", "Ar", "K" , "Ca", "Sc", "Ti", "V" , "Cr",
 "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
 "As", "Se", "Br", "Kr", "Rb", "Sr", "Y" , "Zr",
 "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
 "In", "Sn", "Sb", "Te", "I" , "Xe", "Cs", "Ba",
 "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
 "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
 "Ta", "W" , "Re", "Os", "Ir", "Pt", "Au", "Hg",
 "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
 "Ac", "Th", "Pa", "U" , "Np", "Pu", "Am", "Cm",
 "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
 "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
 "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

elements_dict = {elements_list[_]: _ for _ in range(1, len(elements_list))}

elements_average_weight = [None, 
      1.00794,      4.002602,     6.941,        9.012182, 
     10.811,       12.0107,      14.0067,      15.9994, 
     18.9984032,   20.1797,      22.98976928,  24.3050, 
     26.9815386,   28.0855,      30.973762,    32.065, 
     35.453,       39.948,       39.0983,      40.078, 
     44.955912,    47.867,       50.9415,      51.9961, 
     54.938045,    55.845,       58.933195,    58.6934, 
     63.546,       65.38,        69.723,       72.64, 
     74.92160,     78.96,        79.904,       83.798, 
     85.4678,      87.62,        88.90585,     91.224, 
     92.90638,     95.96,        98,          101.07, 
    102.90550,    106.42,       107.8682,     112.411, 
    114.818,      118.710,      121.760,      127.60, 
    126.90447,    131.293,      132.9054519,  137.327, 
    138.90547,    140.116,      140.90765,    144.242, 
    145,          150.36,       151.964,      157.25, 
    158.92535,    162.500,      164.93032,    167.259, 
    168.93421,    173.054,      174.9668,     178.49, 
    180.94788,    183.84,       186.207,      190.23, 
    192.217,      195.084,      196.966569,   200.59, 
    204.3833,     207.2,        208.98040,    209, 
    210,          222,          223,          226, 
    227,          232.03806,    231.03588,    238.02891, 
    237,          244,          243,          247, 
    247,          251,          252,          257, 
    258,          259,          262,          267, 
    268,          271,          272,          270, 
    276,          281,          280,          285, 
    284,          289,          288,          293, 
    294,          294]


class Molecule(object):
    r"""
this class contains basic information of a xyz file.
"""

    def __init__(self, ifilename: str|None=None):
        if ifilename is not None and ifilename:
            self.read(ifilename)

    def read(self, ifilename: str):
        suffix = os.path.splitext(ifilename)[1]
        if suffix == ".xyz":
            self.read_xyz(ifilename)
        elif suffix == ".gjf":
            self.read_gjf(ifilename)
        else:
            raise ValueError("Cannot understand file name suffix.")

    def resize(self):
        # note: this method does not change natoms.
        self.elements = np.empty((self.natoms,), dtype=np.dtype("<U2"))
        self.coordinates = np.empty((self.natoms, ncoords), dtype=np.double)
        self.atomic_numbers = np.empty((self.natoms,), dtype=int)
        self.atomic_weights = np.empty((self.natoms,), dtype=np.double)

    def read_xyz(self, ifilename: str):
        if os.path.splitext(ifilename)[1] != ".xyz":
            raise ValueError("The suffix of a .xyz file must be \".xyz\".")
        with open(ifilename) as ifile:
            self.natoms = int(ifile.readline())
            comment = ifile.readline()
            self.resize()
            for iatom in range(self.natoms):
                line = ifile.readline().split()
                self.elements[iatom] = line[0]
                self.atomic_numbers[iatom] = elements_dict[self.elements[iatom]]
                self.atomic_weights[iatom] = elements_average_weight[self.atomic_numbers[iatom]]
                self.coordinates[iatom, :] = np.array(line[1:4], dtype=np.double)

    def read_gjf(self, ifilename: str):
        if os.path.splitext(ifilename)[1] != ".gjf":
            raise ValueError("The suffix of a .gjf file must be \".gjf\".")
        with open(ifilename, "rb") as ifile:
            for line in ifile:
                if not line.strip(): break
            for line in ifile:
                if not line.strip(): break
            line = ifile.readline()
            charge, multiplicity = list(map(int, line.split()))
            file_pos = ifile.tell()
            self.natoms = 0
            while True:
                line = ifile.readline()
                if not line.strip(): break
                self.natoms += 1
            ifile.seek(file_pos)
            self.resize()
            for iatom in range(self.natoms):
                line = ifile.readline().split()
                self.elements[iatom] = line[0]
                self.atomic_numbers[iatom] = elements_dict[self.elements[iatom]]
                self.atomic_weights[iatom] = elements_average_weight[self.atomic_numbers[iatom]]
                self.coordinates[iatom, :] = np.array(line[1:4], dtype=np.double)

    def detect_point_group(self, tol: np.double=1.E-4) -> str:
        # quick return
        if not hasattr(self, "natoms") or not self.natoms:
            raise AttributeError("You should load a molecule first.")
        if self.natoms == 1:
            # spherical
            return "Kh"
        elif self.natoms == 2:
            # 2-atoms linear, C_\mathrm{\infty v} or D_\mathrm{\infty h}
            return "Dinfh" if self.elements[0] == self.elements[1] else "Cinfv"
        coords_centered = self.coordinates - self.coordinates.mean(axis=0)

        # get moments of inertia
        # moments_of_inertia_tensor = (self.atomic_weights[:, np.newaxis, np.newaxis] * 
        #     (np.sum(coords_centered ** 2, axis=1)[:, np.newaxis, np.newaxis] * np.eye(ncoords) - 
        #      coords_centered[:, :, np.newaxis] * coords_centered[:, np.newaxis, :])).sum(axis=0)
        moments_of_inertia_tensor = sum(self.atomic_weights[iatom] * 
            (np.dot(coords_centered[iatom], coords_centered[iatom]) * np.eye(3, dtype=np.double) - 
                np.outer(coords_centered[iatom], coords_centered[iatom])) for iatom in range(self.natoms))

        moments_of_inertia, principal_axes = np.linalg.eigh(moments_of_inertia_tensor)
        # moments_of_inertia = sorted(np.linalg.eigvals(moments_of_inertia_tensor)) # do not use this, it may return complex numbers

        # detect SEA
        atomic_numbers_to_compare = np.tile(self.atomic_numbers, (self.natoms, 1))
        distance_to_compare = cdist(coords_centered, coords_centered)

        for iatom in range(self.natoms):
            sort_method = np.argsort(distance_to_compare[iatom])
            distance_to_compare[iatom] = distance_to_compare[iatom][sort_method]
            atomic_numbers_to_compare[iatom] = atomic_numbers_to_compare[iatom][sort_method]

        touched = np.zeros((self.natoms,), dtype=bool)
        SEAs = [] # symmetry equavalent atoms

        for iatom in range(self.natoms):
            if touched[iatom]: continue
            touched[iatom] = True
            SEAs.append([iatom,])

            for jatom in range(self.natoms):
                if touched[jatom]: continue
                if self.elements[iatom] != self.elements[jatom]: continue
                if not np.all(atomic_numbers_to_compare[iatom] == atomic_numbers_to_compare[jatom]): continue
                if np.any(np.abs(distance_to_compare[iatom][1:] - distance_to_compare[jatom][1:]) > tol): continue

                touched[jatom] = True
                SEAs[-1].append(jatom)

        # print([[_ + 1 for _ in SEA_group] for SEA_group in SEAs])

        # print(moments_of_inertia)

        if moments_of_inertia[0] <= tol and moments_of_inertia[2] - moments_of_inertia[1] <= tol:
            # linear, I_A = 0, I_B = I_C
            print("linear")
        elif moments_of_inertia[1] - moments_of_inertia[0] <= tol and moments_of_inertia[2] - moments_of_inertia[1] <= tol:
            # more than one main-axes where n > 2, I_A = I_B = I_C, a.k.a. "spherial-like"
            print("spherial-like")
        elif moments_of_inertia[1] - moments_of_inertia[0] <= tol or moments_of_inertia[2] - moments_of_inertia[1] <= tol:
            # symmetric, I_A = I_B \ne I_C or I_A \ne I_B = I_C
            print("symmetric")
        else:
            # asymmetric, I_A \ne I_B \ne I_C
            print("asymmetric")

        return "undetected"

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc - 1 != 1:
        raise ValueError(f"Usage: {sys.argv[0]:s} xxx.xyz")
    molname = sys.argv[1]
    mol = Molecule(molname)
    print(mol.detect_point_group())

