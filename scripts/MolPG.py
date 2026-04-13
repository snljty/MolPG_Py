#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-

r"""
This program written by python is aimming to
determine the point group of a molecule.

It reads a structure from a xyz file.
"""

__version__ = "1.1.2"

import numpy as np
import os, sys
from scipy.spatial.distance import cdist
# import inspect, IPython
# IPython.embed(header=f"Debug: at line {inspect.currentframe().f_lineno:d} of file {os.path.split(__file__)[-1]:s}:")

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

    def detect_point_group(self, tol: np.double=1.E-4) -> tuple[str, int]:
        # align method of new_coordinates:
        # the geometry center is located at origin point of Cartesian axes.
        # for geometry with no more than one rotation axis Cn with n > 2, 
        # the major axis (the highest n), if any is placed on x axis, 
        # if there are minor C2, one of them is placed on y axis,
        # otherwise if there are \sigma_v, one of them is placed on y axis. 
        # for spherical-like geometry, T-series places 3 C2 on Cartesian axes., 
        # O-series places 3 C4 on Cartesian axes, and I-series has 15 C2, 
        # find 3 of them orthogonal to each other, place them on Cartesian axes.

        # order 0 for infinity
        # quick return
        if not hasattr(self, "natoms") or not self.natoms:
            raise AttributeError("You should load a molecule first.")
        if self.natoms == 1:
            # spherical
            self.new_coordinates[:, :] = 0.
            return "Kh", 0
        elif self.natoms == 2:
            # 2-atoms linear, C_\mathrm{\infty v} or D_\mathrm{\infty h}
            self.new_coordinates[:, :] = 0.
            bond_length: np.double = np.linalg.norm(self.coordinates[1] - self.coordinates[0])
            self.new_coordinates[0, coord_x] = - bond_length / 2.
            self.new_coordinates[1, coord_x] = bond_length / 2.
            return ("Dinfh", 0) if self.elements[0] == self.elements[1] else ("Cinfv", 0)
        coords_centered = self.coordinates - self.coordinates.mean(axis=0)

        # get moments of inertia
        # moments_of_inertia_tensor = (self.atomic_weights[:, np.newaxis, np.newaxis] * 
        #     (np.sum(coords_centered ** 2, axis=1)[:, np.newaxis, np.newaxis] * np.eye(ncoords) - 
        #      coords_centered[:, :, np.newaxis] * coords_centered[:, np.newaxis, :])).sum(axis=0)
        moments_of_inertia_tensor = sum(self.atomic_weights[iatom] * 
            (np.dot(coords_centered[iatom], coords_centered[iatom]) * np.eye(3, dtype=np.double) - 
                np.outer(coords_centered[iatom], coords_centered[iatom])) for iatom in range(self.natoms))

        inertia_tol = tol * 2. * np.linalg.norm(coords_centered, axis=1) @ self.atomic_weights.T
        moments_of_inertia, principal_axes = np.linalg.eigh(moments_of_inertia_tensor)
        if np.linalg.det(principal_axes) < 0.: principal_axes[:, 0] = - principal_axes[:, 0]
        if (np.abs((moments_of_inertia_tensor - np.diag(np.diagonal(moments_of_inertia_tensor)))) > inertia_tol).any():
            coords_centered @= principal_axes # rotate principal axes to x y z
        self.new_coordinates[:, :] = coords_centered[:, :]

        # detect SEA (symmetry equavalent atoms)
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

        if moments_of_inertia[0] <= inertia_tol and moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # linear, I_A = 0, I_B = I_C
            # print("linear")
            # only need to check symmetry center
            # Cinfv or Dinfh

            # the molecule is on axis x

            # coords_operated[:, :] = - coords_centered
            # return ("Dinfh", 0) if is_sym_okay() else ("Cinfv", 0)

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
                    raise RuntimeError("This should never happen.")
            else:
                sym_okay = True
            if sym_okay:
                # C\infty, \infty \sigma_v, \infty C2, S\infty (including i and \sigma_h)
                return "Dinfh", 0
            else:
                # C\infty, \infty \sigma_v
                return "Cinfv", 0

        elif moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol and moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # more than one main-axes where n > 2, I_A = I_B = I_C, a.k.a. "spherical-like"
            # T, Td, Th, O, Oh, I, Ih
            def rotate_half_around_axis(axis: np.ndarray):
                # assume axis is already normalized
                # \cos\pi = -1, \sin\pi = 0
                # R = \begin{bmatrix}
                # \cos\theta + (1-\cos\theta){n_x}^2 & (1-\cos\theta)n_xn_y-\sin\theta n_z & (1-\cos\theta)n_xn_z+\sin\theta n_y \\
                # (1-\cos\theta)n_yn_x+\sin\theta n_z & \cos\theta + (1-\cos\theta){n_y}^2 & (1-\cos\theta)n_yn_z-\sin\theta n_x \\
                # (1-\cos\theta)n_zn_x-\sin\theta n_y & (1-\cos\theta)n_zn_y+\sin\theta n_x & \cos\theta + (1-\cos\theta){n_z}^2 \\
                # \end{bmatrix}
                # Rodrigues' rotation formula:
                # R = I - K(n)\sin\theta + (1-\cos\theta)(nn^\top-I), where K(n) is the asymmetrical cross matrix:
                # K(n) = \begin{bmatrix}
                # 0 & - n_z & n_y \\
                # n_z & 0 & - n_x \\
                # - n_y & n_x & 0 \\
                # \end{bmatrix}
                rot_mat: np.ndarray = 2. * np.outer(axis, axis) - np.identity(ncoords, dtype=np.double)
                coords_operated[:, :] = coords_centered @ rot_mat.T

            def rotate_quarter_around_axis(axis: np.ndarray):
                # assume axis is already normalized
                # \cos\frac\pi2 = 0, \sin\frac\pi2 = 1
                cross_mat: np.ndarray = np.array([[0., - axis[coord_z], axis[coord_y]], 
                                                  [axis[coord_z], 0., - axis[coord_x]], 
                                                  [- axis[coord_y], axis[coord_x], 0.]], dtype=np.double)
                rot_mat: np.ndarray = np.outer(axis, axis) + cross_mat
                coords_operated[:, :] = coords_centered @ rot_mat.T

            def flip_against_plane(normal_axis: np.ndarray):
                # assume normal_axis is already normalized
                # \sigma_"h" = i @ C2
                flip_mat: np.ndarray = np.identity(ncoords, dtype=np.double) - 2. * np.outer(normal_axis, normal_axis)
                coords_operated[:, :] = coords_centered @ flip_mat.T

            def rotate_around_axis(axis: np.ndarray, order: int, times: int=1):
                # assume axis is already normalized
                theta = 2. * np.pi / np.double(order) * np.double(times)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                cross_mat: np.ndarray = np.array([[0., - axis[coord_z], axis[coord_y]], 
                                                  [axis[coord_z], 0., - axis[coord_x]], 
                                                  [- axis[coord_y], axis[coord_x], 0.]], dtype=np.double)
                rot_mat: np.ndarray = np.identity(ncoords, dtype=np.double) + cross_mat * sin_theta + \
                    (1. - cos_theta) * (np.outer(axis, axis) - np.identity(ncoords, dtype=np.double))
                coords_operated[:, :] = coords_centered @ rot_mat.T

            # T series have 3 C2, O series have 6 individual C2 and 3 C4, I series have 15 C2
            all_C2: np.ndarray = np.zeros((15, ncoords), dtype=np.double)
            num_C2_found: int = 0

            axis_point: np.ndarray
            axis_point_norm: np.double
            # first, check C2 through center of two SEAs
            for SEA_group in SEAs:
                if len(SEA_group) < 2: continue
                for iatom in SEA_group:
                    for jatom in SEA_group:
                        if iatom >= jatom: continue
                        axis_point = (coords_centered[iatom] + coords_centered[jatom]) / 2.
                        axis_point_norm = np.linalg.norm(axis_point)
                        if axis_point_norm  <= tol: continue
                        axis_point /= axis_point_norm
                        for C2_index in range(num_C2_found):
                            if np.linalg.norm(axis_point - all_C2[C2_index]) <= tol or \
                               np.linalg.norm(axis_point + all_C2[C2_index]) <= tol: break
                        else:
                            rotate_half_around_axis(axis_point)
                            if is_sym_okay():
                                if num_C2_found >= 15: raise RuntimeError("This should never happen.")
                                all_C2[num_C2_found] = axis_point
                                num_C2_found += 1

            # second, check C2 through each atom
            for iatom in range(self.natoms):
                axis_point_norm = np.linalg.norm(coords_centered[iatom])
                if axis_point_norm <= tol: continue
                axis_point = coords_centered[iatom] / axis_point_norm
                for C2_index in range(num_C2_found):
                    if np.linalg.norm(axis_point - all_C2[C2_index]) <= tol or \
                        np.linalg.norm(axis_point + all_C2[C2_index]) <= tol: break
                else:
                    rotate_half_around_axis(axis_point)
                    if is_sym_okay():
                        if num_C2_found >= 15: raise RuntimeError("This should never happen.")
                        all_C2[num_C2_found] = axis_point
                        num_C2_found += 1

            if num_C2_found == 3:
                # T, Td, Th
                flip_against_plane(all_C2[0])
                has_sigma_h: bool = is_sym_okay()
                # since two C2 are orthogonal, if there is a \sigma_d divides them, there must be another perpendicular to that.
                mirror_point = (all_C2[0] + all_C2[1]) / np.sqrt(2.)
                flip_against_plane(mirror_point)
                has_sigma_d: bool = is_sym_okay()
                rot_mat: np.ndarray = all_C2[:3].copy()
                if np.linalg.det(rot_mat) < 0.: rot_mat[coord_x] = - rot_mat[coord_x]
                coords_centered @= rot_mat.T
                self.new_coordinates[:, :] = coords_centered
                # aligned coordinates:
                # 3 C2(1):
                #     [1, 0, 0]
                #     [0, 1, 0]
                #     [0, 0, 1]
                # 4 C3(1,2):
                #     [1,  1,  1] / sqrt(3)
                #     [1,  1, -1] / sqrt(3)
                #     [1, -1,  1] / sqrt(3)
                #     [1, -1, -1] / sqrt(3)
                #
                # 1 i                  (Th)
                # 3 sigma_h            (Th): perpendicular to C2
                # 4 S6(1,5)= - I3(1,5) (Th): superposition with C3
                #
                # 6 sigma_d             (Td): divides any two C2s,  contains the third C2
                # 3 S4(1,3) = - I4(1,3) (Td): superposition with C2
                return ("Th", 24) if has_sigma_h else ("Td", 24) if has_sigma_d else ("T", 12)

            elif num_C2_found == 9:
                # O, Oh
                C4_index: np.ndarray = np.zeros((3,), dtype=int)
                num_C4_found: int = 0
                for C2_index in range(9):
                    rotate_quarter_around_axis(all_C2[C2_index])
                    if is_sym_okay():
                        C4_index[num_C4_found] = C2_index
                        num_C4_found += 1
                if num_C4_found != 3: raise RuntimeError("This should never happen.")
                flip_against_plane(all_C2[C4_index[0]])
                has_sigma_h: bool = is_sym_okay()
                rot_mat: np.ndarray = np.array([all_C2[C4_index[0]], all_C2[C4_index[1]], all_C2[C4_index[2]]], dtype=np.double)
                if np.linalg.det(rot_mat) < 0.: rot_mat[coord_x] = - rot_mat[coord_x]
                coords_centered @= rot_mat.T
                self.new_coordinates[:, :] = coords_centered
                # aligned coordinates:
                # 6 C2(1):
                #     [ 0,  1,  1] / sqrt(2)
                #     [ 0,  1, -1] / sqrt(2)
                #     [ 1,  0,  1] / sqrt(2)
                #     [-1,  0,  1] / sqrt(2)
                #     [ 1,  1,  0] / sqrt(2)
                #     [ 1, -1,  0] / sqrt(2)
                # 4 C3(1,2):
                #     [1,  1,  1] / sqrt(3)
                #     [1,  1, -1] / sqrt(3)
                #     [1, -1,  1] / sqrt(3)
                #     [1, -1, -1] / sqrt(3)
                # 3 C4(1,2,3) (C4(2) = C2(1), not the C2(1) above):
                #     [1, 0, 0]
                #     [0, 1, 0]
                #     [0, 0, 1]
                #
                # 1 i                 (Oh)
                # 6 sigma_d           (Oh): divides any two C4s, contains the third C4
                # 4 S6(1,5) = I3(1,5) (Oh): superposition with C3
                # 3 sigma_h           (Oh): perpendicular to C4
                # 3 S4(1,3) = I4(1,3) (Oh): superposition with C4
                return ("Oh", 48) if has_sigma_h else ("O", 24)

            elif num_C2_found == 15:
                # I, Ih
                flip_against_plane(all_C2[0])
                has_sigma: bool = is_sym_okay()
                C2_use_index: np.ndarray = np.zeros((3,), dtype=int)
                for C2_use_index[1] in range(1, 15):
                    if np.abs(np.dot(all_C2[C2_use_index[1]], all_C2[0])) <= tol: break
                else:
                    raise RuntimeError("This should never happen.")
                for C2_use_index[2] in range(1, 15):
                    if C2_use_index[2] == C2_use_index[1]: continue
                    if np.abs(np.dot(all_C2[C2_use_index[2]], all_C2[0])) <= tol and \
                        np.abs(np.dot(all_C2[C2_use_index[2]], all_C2[C2_use_index[1]])) <= tol: break
                else:
                    raise RuntimeError("This should never happen.")
                rot_mat: np.ndarray = np.array([all_C2[C2_use_index[0]], all_C2[C2_use_index[1]], all_C2[C2_use_index[2]]], dtype=np.double)
                if np.linalg.det(rot_mat) < 0.: rot_mat[coord_x] = - rot_mat[coord_x]
                coords_centered @= rot_mat.T
                self.new_coordinates[:, :] = coords_centered
                # for fullerene C60 (soccer-ball-like), there are 12 pentagons and 20 hexagons. 
                # each pentagon has 5 hexagon neighbor, 
                # each hexagon has alternate distributing three pentagons and three hexagons.
                # each C5 passes through centers of two opposite pentagons.
                # each C3 passes through centers of two opposite hexagons.
                # each C2 passes through the midpoints of two opposite common edges between a hexagonal rings and its adjacent hexagonal rings.
                # let \phi=\frac{\sqrt{5}+1}{2}
                # aligned coordinates:
                # 15 C2(1) : x, y, z axes, [1, \pm\phi, \pm(1-\phi)] / 2 and their cyclic permutations
                # 10 C3(1,2) : [1, \pm(1+\phi), 0] / sqrt(3\phi+3) and their cyclic permutations and [1, \pm1, \pm1] / sqrt(3)
                # 6  C5(1,2,3,4) : [1, 0, \pm\phi] / sqrt(\phi+2) and theri cyclic permutations
                # i                            (Ih)
                # 6 S10(1,3,7,9) = I5(7,1,9,3) (Ih) : superposition with C5
                # 10 S6(1,5) = I3(1,5)         (Ih) : superposition with C3
                # 15 sigma                     (Ih) : perpendicular to C2
                return ("Ih", 120) if has_sigma else ("I", 60)

            else:
                raise RuntimeError("This should never happen.")

        elif moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol or moments_of_inertia[2] - moments_of_inertia[1] <= inertia_tol:
            # symmetric, I_A = I_B \ne I_C or I_A \ne I_B = I_C
            # Dnd for n >= 2, (Cn, Cnh, Cnv, Dn, Dnh) for n > 2, Cni for (n > 1 and n is odd, a.k.a. S(2n)) and Sn for 4 | n

            # rotate the principal axis corresponding to the unequivalent moment of inertia to x axis
            if moments_of_inertia[1] - moments_of_inertia[0] <= inertia_tol:
                coords_centered[:, coord_x], coords_centered[:, coord_z] = coords_centered[:, coord_z].copy(), - coords_centered[:, coord_x].copy()

            # detect the main axis n > 2
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
                raise RuntimeError("This should never happen.")

            # find available C2
            has_minor_C2: bool = False
            axis_point: np.ndarray
            axis_point_norm: np.double
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            projection: np.ndarray
            # first, check centers of two SEAs
            for SEA_group in SEAs:
                if len(SEA_group) < 2: continue
                for iatom in SEA_group:
                    for jatom in SEA_group:
                        if iatom >= jatom: continue
                        axis_point = (coords_centered[iatom, coord_y:] + coords_centered[jatom, coord_y:]) / 2.
                        axis_point_norm = np.linalg.norm(axis_point)
                        if axis_point_norm  <= tol: continue
                        axis_point /= axis_point_norm
                        # test a C2 through origin point and center of iatom and jatom
                        projection = np.outer(coords_centered[:, coord_y:] @ axis_point, axis_point)
                        coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                        if is_sym_okay():
                            has_minor_C2 = True
                            break
                    if has_minor_C2: break
                if has_minor_C2: break

            if not has_minor_C2:
                # second, check C2 through each atom
                for iatom in range(self.natoms):
                    axis_point_norm = np.linalg.norm(coords_centered[iatom, coord_y:])
                    if axis_point_norm <= tol: continue
                    axis_point = coords_centered[iatom, coord_y:] / axis_point_norm
                    # test a C2 through origin point and iatom
                    projection = np.outer(coords_centered[:, coord_y:] @ axis_point, axis_point)
                    coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                    if is_sym_okay():
                        has_minor_C2 = True
                        break

            if has_minor_C2:
                # rotate the found C2 to y axis
                rot_mat = np.array([[axis_point[0], axis_point[1]], [- axis_point[1], axis_point[0]]])
                coords_centered[:, coord_y:] @= rot_mat.T
                self.new_coordinates[:, :] = coords_centered[:, :]

            if major_Cn == 2:
                # D2d, S4
                if has_minor_C2:
                    # S4(1,3), major C2=S4(2), 2 minor C2, 2 \sigma_d
                    return "D2d", 8
                else:
                    # S4(1,3), C2
                    return "S4", 4

            # find sigma_h
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            coords_operated[:, coord_y:] = coords_centered[:, coord_y:]
            has_sigma_h: bool = is_sym_okay()

            if has_minor_C2:
                # Dn, Dnh, Dnd for n > 2
                if has_sigma_h:
                    # Cn, n C2, n \sigma_v, n (\sigma_h@Cn(1,...,n)) 
                    # the last one is (Sm(k), k is odd, m is a divisor of n, including \sigma_h=\sigma_h@Cn(n)), 
                    # for even n, i is included in above.)
                    return "D{:d}h".format(major_Cn), major_Cn * 4
                # Dn, Dnd for n > 2
                # if exists sigma_d, it divides two minor C2. one minor C2 is already on axis y.
                coords_operated[:, coord_x] = coords_centered[:, coord_x]
                angle = np.pi / np.double(major_Cn) / 2.
                axis_point = np.array([np.cos(angle), np.sin(angle)])
                projection = (coords_centered[:, coord_y:] @ axis_point[:, np.newaxis]) * axis_point
                coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                has_sigma_d: bool = is_sym_okay()
                if has_sigma_d:
                    # Cn, n C2, n \sigma_d, S{2n}(1,3,...,2n-1). for odd n there is S{2n}(n)=i
                    # we can prove that the combination of a minor C2 and its nearest clockwise \sigma_d
                    # is actually S{2n}(1), and since S{2n}(2)=Cn(1), there combinations are all S{2n}.
                    return "D{:d}d".format(major_Cn), major_Cn * 4
                else:
                    # Cn, n C2
                    return "D{:d}".format(major_Cn), major_Cn * 2

            # (Cn, Cnv, Cnh) for n > 2, Cni for odd i > 1, S4n for n > 1
            coords_operated[:, :] = - coords_centered
            has_sym_center: bool = is_sym_okay()
            # S{4n+2} = C{2n+1} + i (C{2n+1}i), S{2n+1} = C{2n+1} + sigma_h (C{2n+1}h)
            # if there is S4n, there must be C2n, and S4n does not contain i or sigma_h
            if major_Cn % 2 != 0:
                if has_sym_center and has_sigma_h: raise RuntimeError("This should never happen.")
                if has_sym_center:
                    # Cn, (i@Cn(1,...,n))=In(k=1,3,...,2n-1)=S{2n}(mod(2k+n,2n)). i=In(n)
                    return "C{:d}i".format(major_Cn), major_Cn * 2
                if has_sigma_h:
                    # Cn, (\sigma_h@Cn(1,...,n))=Sn(k=1,3,...,2n-1)=I{2n}(mod(2k+n,2n)). \sigma_h=Sn(n)
                    return "C{:d}h".format(major_Cn), major_Cn * 2 # only odd Cnh here
            else:
                if has_sigma_h:
                    if not has_sym_center: raise RuntimeError("This should never happen.")
                    # the symmetry elements can be written as:
                    # Cn, (\sigma_h@Cn(1,...,n)). i=\sigma_h@Cn(n/2). \sigma_h=\sigma_h@Cn(n)
                    # let g be the largest power of two that is a common devisor of n and k, 
                    # then if k/g is odd, \sigma_h@Cn(k)=S{n/g}(k/g), otherwise S{n/g}(k/g+n/g)
                    # or the symmetry elements can be written as:
                    # Cn, (i@Cn(1,...,n)). i=i@Cn(). \sigma_h=@Cn(n/2)
                    # let g be the largest power of two that is a common devisor of n and k, 
                    # then if k/g is odd, i@Cn(k)=I{n/g}(k/g), otherwise I{n/g}(k/g+n/g)
                    return "C{:d}h".format(major_Cn), major_Cn * 2 # only even Cnh here
                # if major_Cn is odd, then Sn(n)=\sigma_h, there must be \sigma_h
                # if major_Cn is an odd multiple of two, then Sn(n/2)=\sigma_d@C2=i, there must be i
                # in other words, if there is Sm but neigher \sigma_h nor i, m must be a multiple of 4, 
                # and in this case the maximum rotation axis of Sm is C{m/2}, hence only needs to test C{2n}.
                S_order: int = major_Cn * 2
                rotate_around_x_by_n(S_order)
                coords_operated[:, coord_x] = - coords_centered[:, coord_x]
                has_Sn: bool = is_sym_okay()
                if has_Sn:
                    # S{n}(k) for odd 1<=k<n, C{n/2}(k) for 1<=k<= n/2
                    return "S{:d}".format(S_order), S_order

            # Cn or Cnv for n > 2
            # find available sigma v
            has_sigma_v: bool = False
            mirror_point: np.ndarray
            mirror_point_norm: np.double
            coords_operated[:, coord_x] = coords_centered[:, coord_x]
            # first, check centers of two SEAs
            for SEA_group in SEAs:
                if len(SEA_group) < 2: continue
                for iatom in SEA_group:
                    for jatom in SEA_group:
                        if iatom >= jatom: continue
                        mirror_point = (coords_centered[iatom, coord_y:] + coords_centered[jatom, coord_y:]) / 2.
                        mirror_point_norm = np.linalg.norm(mirror_point)
                        if mirror_point_norm  <= tol: continue
                        mirror_point /= mirror_point_norm
                        # test a C2 through origin point and center of iatom and jatom
                        projection = np.outer(coords_centered[:, coord_y:] @ mirror_point, mirror_point)
                        coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                        if is_sym_okay():
                            has_sigma_v = True
                            break
                    if has_sigma_v: break
                if has_sigma_v: break

            if not has_sigma_v:
                # second, check C2 through each atom
                for iatom in range(self.natoms):
                    mirror_point_norm = np.linalg.norm(coords_centered[iatom, coord_y:])
                    if mirror_point_norm <= tol: continue
                    mirror_point = coords_centered[iatom, coord_y:] / mirror_point_norm
                    # test a C2 through origin point and iatom
                    projection = np.outer(coords_centered[:, coord_y:] @ mirror_point, mirror_point)
                    coords_operated[:, coord_y:] = 2. * projection - coords_centered[:, coord_y:]
                    if is_sym_okay():
                        has_sigma_v = True
                        break

            if has_sigma_v:
                # rotate the found sigma_v to y axis
                rot_mat = np.array([[mirror_point[0], mirror_point[1]], [- mirror_point[1], mirror_point[0]]])
                coords_centered[:, coord_y:] @= rot_mat.T
                self.new_coordinates[:, :] = coords_centered[:, :]
                # Cn, n \sigma_v
                return "C{:d}v".format(major_Cn), major_Cn * 2
            else:
                # Cn
                return "C{:d}".format(major_Cn), major_Cn

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
                if has_xOy_mirror and has_yOz_mirror and has_zOx_mirror:
                    # 3 C2, 3 \sigma_h, i
                    return "D2h", 8
                else:
                    # 3 C2
                    return "D2", 4

            # C2, C2h, C2v, C1, Ci, Cs
            # rotate the C2 axis to x axis
            if has_z_C2:
                coords_centered[:, coord_x], coords_centered[:, coord_z] = coords_centered[:, coord_z].copy(), - coords_centered[:, coord_x].copy()
                has_z_C2 = False
                has_x_C2 = True
            elif has_y_C2:
                coords_centered[:, coord_x], coords_centered[:, coord_y] = - coords_centered[:, coord_y].copy(), coords_centered[:, coord_x].copy()
                has_y_C2 = False
                has_x_C2 = True

            if has_x_C2:
                # C2, C2h, C2v
                coords_operated[:, coord_x] = - coords_centered[:, coord_x]
                coords_operated[:, coord_y] =   coords_centered[:, coord_y]
                coords_operated[:, coord_z] =   coords_centered[:, coord_z]
                has_yOz_mirror = is_sym_okay()
                if has_yOz_mirror:
                    # C2, \sigma_h, i
                    return "C2h", 4
                # C2, C2v
                coords_operated[:, coord_x] =   coords_centered[:, coord_x]
                coords_operated[:, coord_z] = - coords_centered[:, coord_z]
                has_xOy_mirror = is_sym_okay()
                if has_xOy_mirror:
                    # C2, 2 \sigma_v
                    return "C2v", 4
                else:
                    # C2
                    return "C2", 2

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
            if has_xOy_mirror or has_yOz_mirror or has_zOx_mirror:
                # \sigma
                return "Cs", 2

            # C1, Ci
            coords_operated[:, coord_x] = - coords_centered[:, coord_x]
            coords_operated[:, coord_y] = - coords_centered[:, coord_y]
            has_sym_center = is_sym_okay()
            if has_sym_center:
                # i
                return "Ci", 2
            else:
                # nothing except E
                return "C1", 1

        return "undetected", 1

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc - 1 != 1 and argc - 1 != 2:
        raise ValueError(f"Usage: {sys.argv[0]:s} xxx.xyz|xxx.gjf [tolerance]")
    molname = sys.argv[1]
    mol = Molecule(molname)
    if argc - 1 == 2:
        tol = np.double(sys.argv[2])
        point_group, order = mol.detect_point_group(tol)
    else:
        point_group, order = mol.detect_point_group()
    print(point_group)
    # if order:
    #     print(f"order = {order:d}")
    # else:
    #     print(R"order = \infty")
    # mol.use_new_coordinates()
    # mol.write_gjf("new.gjf")

