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

ncoords: int = 3
coord_x: int
coord_y: int
coord_z: int
coord_x, coord_y, coord_z = range(ncoords)

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

    def write(self, ofilename: str):
        suffix = os.path.splitext(ofilename)[1]
        if suffix == ".xyz":
            self.write_xyz(ofilename)
        elif suffix == ".gjf":
            self.write_gjf(ofilename)
        else:
            raise ValueError("Cannot understand file name suffix.")

    def resize(self):
        # note: this method does not change natoms.
        self.elements = np.empty((self.natoms,), dtype=np.dtype("<U2"))
        self.coordinates = np.empty((self.natoms, ncoords), dtype=np.double)
        self.atomic_numbers = np.empty((self.natoms,), dtype=int)
        self.atomic_weights = np.empty((self.natoms,), dtype=np.double)
        self.new_coordinates = np.zeros((self.natoms, ncoords), dtype=np.double)

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
            else:
                raise IOError("Cannot read title.")
            for line in ifile:
                if not line.strip(): break
            else:
                raise IOError("Cannot read charge and multiplicity.")
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
            line = ifile.readline()
            if not line: raise IOError("Cannot read final blank line.")

    def write_xyz(self, ofilename: str):
        if os.path.splitext(ofilename)[1] != ".xyz":
            raise ValueError("The suffix of a .xyz file must be \".xyz\".")
        with open(ofilename, "w") as ofile:
            print("{:5d}".format(self.natoms), file=ofile)
            print(os.path.splitext(ofilename)[0], file=ofile)
            for iatom in range(self.natoms):
                print(f" {self.elements[iatom]:<2s}" + ("    {:13.8f}" * 3).format(*self.coordinates[iatom]), file=ofile)

    def write_gjf(self, ofilename: str):
        if os.path.splitext(ofilename)[1] != ".gjf":
            raise ValueError("The suffix of a .gjf file must be \".gjf\".")
        with open(ofilename, "w") as ofile:
            print(f"%chk={os.path.splitext(ofilename)[0]:s}.chk", file=ofile)
            print("#P B3LYP/6-31G* EmpiricalDispersion=GD3BJ 5D", file=ofile)
            print("\n{:s}\n".format(os.path.splitext(ofilename)[0]), file=ofile)
            print(" {:d} {:d}".format(0, 1), file=ofile)
            for iatom in range(self.natoms):
                print(f" {self.elements[iatom]:<2s}" + ("    {:13.8f}" * 3).format(*self.coordinates[iatom]), file=ofile)
            print(file=ofile)

    def use_new_coordinates(self):
        self.coordinates[:, :] = self.new_coordinates[:, :]

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
        if np.linalg.det(principal_axes) < 0.: principal_axes[:, 0] = - principal_axes[:, 0]
        coords_centered @= principal_axes # rotate principal axes to x y z
        print(moments_of_inertia)

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
        coords_operated = coords_centered.copy()

        def is_sym_okay() -> bool:
            touched.fill(False)
            sym_okay = True
            for SEA_group in SEAs:
                for iatom in SEA_group:
                    for jatom in SEA_group:
                        if np.all(np.abs(coords_operated[iatom] - coords_centered[jatom]) <= tol):
                            touched[iatom] = True
                            break
                    else:
                        sym_okay = False
                if not sym_okay: break
            else:
                sym_okay = True
            return sym_okay

        inertia_tol = tol * 2. * np.linalg.norm(coords_centered, axis=1) @ self.atomic_weights.T
        if moments_of_inertia[0] <= inertia_tol and moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # linear, I_A = 0, I_B = I_C
            # print("linear")
            # only need to check symmetry center
            # Cinfv or Dinfh

            # the molecule is on axis x

            # coords_operated[:, :] = - coords_centered
            # return "Dinfh" if is_sym_okay() else "Cinfv"

            for SEA_group in SEAs:
                if len(SEA_group) == 2:
                    if np.abs(coords_centered[SEA_group[0], coord_x] + coords_centered[SEA_group[1], coord_x]) / 2. > tol:
                        sym_okay = False
                        break
                elif len(SEA_group) == 1:
                    if np.abs(coords_centered[SEA_group[0], coord_x]) > tol:
                        sym_okay = False
                        break
                else:
                    raise ValueError("This should never happen.")
            else:
                sym_okay = True
            return "Dinfh" if sym_okay else "Cinfv"

        elif moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol and moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # more than one main-axes where n > 2, I_A = I_B = I_C, a.k.a. "spherial-like"
            # T, Td, Th, O, Oh, I, Ih
            print("spherial-like")

        elif moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol or moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # symmetric, I_A = I_B \ne I_C or I_A \ne I_B = I_C
            # Dnd for n>= 2, (Cn, Cnh, Cnv, Dn, Dnh) for n > 2, Cni for (n > 1 and n is odd, a.k.a. S(2n)) and Sn for 4 | n
            print("symmetric")

            # rotate the principal axis corresponding to the unequivalent moment of inertia to x axis
            if moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol:
                swp = - coords_centered[:, coord_x]
                coords_centered[:, coord_x] = coords_centered[:, coord_z]
                coords_centered[:, coord_z] = swp
                del swp

            # detect the main axis n>2
            # first find the maximum possible Cn axis
            max_Cn_try: int = 2
            for SEA_group in SEAs:
                if len(SEA_group) <= max_Cn_try: continue
                # at least 2 elements in compare_x
                compare_x = np.empty((len(SEA_group),), dtype=np.double)
                for i, iatom in enumerate(SEA_group):
                    compare_x[i] = coords_centered[iatom, coord_x]
                compare_x.sort()
                max_Cn_try_current: int = 1
                max_Cn_try_list: list = []
                for i in range(1, len(compare_x)):
                    if compare_x[i] - compare_x[i - 1] <= tol:
                        max_Cn_try_current += 1
                    else:
                        max_Cn_try_list.append(max_Cn_try_current)
                        max_Cn_try_current = 1
                max_Cn_try_list.append(max_Cn_try_current)
                for max_Cn_try_current in max_Cn_try_list:
                    if max_Cn_try_current > max_Cn_try: max_Cn_try = max_Cn_try_current
            
            def rotate_around_x_by_n(n: int):
                angle = 2. * np.pi / np.double(n)
                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)
                rot_mat = np.array([[cos_theta, - sin_theta], [sin_theta, cos_theta]], dtype=np.double)
                coords_operated[:, coord_x] = coords_centered[:, coord_x]
                coords_operated[:, coord_y:] = coords_centered[:, coord_y:] @ rot_mat.T

            # find the major Cn
            for major_Cn in range(max_Cn_try, 1, -1):
                rotate_around_x_by_n(major_Cn)
                if is_sym_okay(): break
            else:
                raise RuntimeError("This should never happen.");

            # find available C2
            found_minor_C2: bool = False
            axis_point: np.ndarray
            axis_point_norm: np.double
            # first, check centers of two SEAs
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            for SEA_group in SEAs:
                if len(SEA_group) < 2: continue
                for iatom in SEA_group:
                    for jatom in SEA_group:
                        if iatom == jatom: continue
                        axis_point = (coords_centered[iatom, coord_y:] + coords_centered[jatom, coord_y:]) / 2.
                        axis_point_norm = np.linalg.norm(axis_point)
                        if axis_point_norm  <= tol: continue
                        axis_point /= axis_point_norm
                        # test a C2 through origin point and center of iatom and jatom
                        projection = (coords_centered[:, coord_y:] @ axis_point[:, np.newaxis]) * axis_point
                        coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                        if is_sym_okay():
                            found_minor_C2 = True
                            break
                    if found_minor_C2: break
                if found_minor_C2: break

            if not found_minor_C2:
                # second, check C2 through each atom
                for iatom in range(self.natoms):
                    axis_point_norm = np.linalg.norm(coords_centered[iatom, coord_y:])
                    if axis_point_norm <= tol: continue
                    axis_point = coords_centered[iatom, coord_y:] / axis_point_norm
                    # test a C2 through origin point and iatom
                    projection = (coords_centered[:, coord_y:] @ axis_point[:, np.newaxis]) * axis_point
                    coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                    if is_sym_okay():
                        found_minor_C2 = True
                        break

            if found_minor_C2:
                # rotate the found C2 to y axis
                rot_mat = np.array([[axis_point[0], axis_point[1]], [- axis_point[1], axis_point[0]]])
                coords_centered[:, coord_y:] @= rot_mat.T

            print(found_minor_C2)

        else:
            # asymmetric, I_A \ne I_B \ne I_C
            # D2, D2h, C2, C2h, C2v, C1, Ci, Cs

            has_x_C2: bool
            has_y_C2: bool
            has_z_C2: bool
            has_xOy_mirror: bool
            has_yOz_mirror: bool
            has_zOx_mirror: bool
            has_sym_center: bool

            coords_operated[:, coord_x] =   coords_centered[:, coord_x]
            coords_operated[:, coord_y] = - coords_centered[:, coord_y]
            coords_operated[:, coord_z] = - coords_centered[:, coord_z]
            has_x_C2 = is_sym_okay()
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            coords_operated[:, coord_y] =   coords_centered[:, coord_y]
            has_y_C2 = is_sym_okay()
            coords_operated[:, coord_y] = - coords_centered[:, coord_y]
            coords_operated[:, coord_z] =   coords_centered[:, coord_z]
            has_z_C2 = is_sym_okay()

            # there can only be 0, 1 or 3 C2 in this situation
            if has_x_C2 and has_y_C2 and has_z_C2:
                # D2, D2h
                coords_operated[:, coord_x] =   coords_centered[:, coord_x]
                has_zOx_mirror = is_sym_okay()
                coords_operated[:, coord_x] = - coords_centered[:, coord_x]
                coords_operated[:, coord_y] =   coords_centered[:, coord_y]
                has_yOz_mirror = is_sym_okay()
                coords_operated[:, coord_x] =   coords_centered[:, coord_x]
                coords_operated[:, coord_z] = - coords_centered[:, coord_z]
                has_xOy_mirror = is_sym_okay()
                return "D2h" if has_xOy_mirror and has_yOz_mirror and has_zOx_mirror else "D2"

            # C2, C2h, C2v, C1, Ci, Cs
            # rotate the C2 axis to x axis
            if has_z_C2:
                swp = - coords_centered[:, coord_x]
                coords_centered[:, coord_x] = coords_centered[:, coord_z]
                coords_centered[:, coord_z] = swp
                has_z_C2 = False
                has_x_C2 = True
                del swp
            elif has_y_C2:
                swp = - coords_centered[:, coord_y]
                coords_centered[:, coord_y] = coords_centered[:, coord_x]
                coords_centered[:, coord_x] = swp
                has_y_C2 = False
                has_x_C2 = True
                del swp

            if has_x_C2:
                # C2, C2h, C2v
                coords_operated[:, coord_x] = - coords_centered[:, coord_x]
                coords_operated[:, coord_y] =   coords_centered[:, coord_y]
                coords_operated[:, coord_z] =   coords_centered[:, coord_z]
                has_yOz_mirror = is_sym_okay()
                if has_yOz_mirror: return "C2h"
                # C2, C2v
                coords_operated[:, coord_x] =   coords_centered[:, coord_x]
                coords_operated[:, coord_z] = - coords_centered[:, coord_z]
                has_xOy_mirror = is_sym_okay()
                return "C2v" if has_xOy_mirror else "C2"

            # C1, Ci, Cs
            coords_operated[:, coord_x] =   coords_centered[:, coord_x]
            coords_operated[:, coord_y] = - coords_centered[:, coord_y]
            coords_operated[:, coord_z] =   coords_centered[:, coord_z]
            has_zOx_mirror = is_sym_okay()
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            coords_operated[:, coord_y] =   coords_centered[:, coord_y]
            has_yOz_mirror = is_sym_okay()
            coords_operated[:, coord_x] =   coords_centered[:, coord_x]
            coords_operated[:, coord_z] = - coords_centered[:, coord_z]
            has_xOy_mirror = is_sym_okay()
            if has_xOy_mirror or has_yOz_mirror or has_zOx_mirror: return "Cs"

            # C1, Ci
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            coords_operated[:, coord_y] = - coords_centered[:, coord_y]
            has_sym_center = is_sym_okay()
            return "Ci" if has_sym_center else "C1"

        return "undetected"

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc - 1 != 1:
        raise ValueError(f"Usage: {sys.argv[0]:s} xxx.xyz")
    molname = sys.argv[1]
    mol = Molecule(molname)
    print(mol.detect_point_group())

