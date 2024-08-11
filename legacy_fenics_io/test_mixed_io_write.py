##################################################
# This script is intended to test parallel I/O for legacy FEniCS
##################################################

import os
from mpi4py import MPI
import dolfin as dl
import ufl
from utils import root_print

# MPI setup.
COMM = MPI.COMM_WORLD
rank = COMM.rank
nproc = COMM.size

# General setup.
OUTPUT_DIR = "output"
MESHFILE = os.path.join(OUTPUT_DIR, "box.xdmf")
H5FILE = os.path.join(OUTPUT_DIR, "test_mixed_h5.h5")
NAME_2_READ = "test"

SEP = 80*"#"
root_print(COMM, SEP)
root_print(COMM, f"There are {nproc} processes in total.")
root_print(COMM, f"The mesh file will be written to {MESHFILE}.")
root_print(COMM, f"The HDF5 file will be written to {H5FILE}.")

# mesh = dl.UnitCubeMesh(COMM, 10, 10, 10)
mesh = dl.UnitSquareMesh(COMM, 10, 10)

# write the mesh to file.
with dl.XDMFFile(COMM, MESHFILE) as fid:
    fid.write(mesh)

Vh = dl.FunctionSpace(mesh, "CG", 1)
mixed_element = ufl.MixedElement([Vh.ufl_element(), Vh.ufl_element()])
Vhm = dl.FunctionSpace(mesh, mixed_element)

assigner = dl.FunctionAssigner(Vhm, [Vh, Vh])

expr0 = dl.Expression("x[0]", degree=2)
u0 = dl.interpolate(expr0, Vh)
expr1 = dl.Expression("x[1]", degree=2)
u1 = dl.interpolate(expr1, Vh)

u = dl.Function(Vhm)
assigner.assign(u, [u0, u1])

# write solution to HDF5.
with dl.HDF5File(COMM, H5FILE, "w") as fid:
    fid.write(mesh, "mesh")
    fid.write(u, NAME_2_READ)
    
root_print(COMM, SEP)
