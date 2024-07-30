##################################################
# This script is intended to test parallel I/O for legacy FEniCS
##################################################

from mpi4py import MPI
import dolfin as dl
from utils import root_print

# MPI setup.
COMM = MPI.COMM_WORLD
rank = COMM.rank
nproc = COMM.size

# General setup.
XDMFFILE = "test_xdmf.xdmf"
H5FILE = "test_h5.h5"
NAME_2_READ = "test"

SEP = 80*"#"
root_print(COMM, SEP)
root_print(COMM, f"There are {nproc} processes in total.")
root_print(COMM, f"The XDMF file will be written to {XDMFFILE}.")
root_print(COMM, f"The HDF5 file will be written to {H5FILE}.")

mesh = dl.UnitSquareMesh(COMM, 10, 10)
Vh = dl.FunctionSpace(mesh, "CG", 1)

uexpr = dl.Expression("x[0]", degree=2)
u = dl.interpolate(uexpr, Vh)

u2expr = dl.Expression("x[1]", degree=2)
u2 = dl.interpolate(u2expr, Vh)

# write solution to XDMF.
with dl.XDMFFile(COMM, XDMFFILE) as fid:
    fid.write(mesh)
    # Perhaps difficult
    fid.write_checkpoint(u, NAME_2_READ, 0, dl.XDMFFile.Encoding.HDF5, True)
    fid.write_checkpoint(u2, NAME_2_READ, 1, dl.XDMFFile.Encoding.HDF5, True)

# write solution to HDF5.
with dl.HDF5File(COMM, H5FILE, "w") as fid:
    fid.write(mesh, "mesh")
    fid.write(u, NAME_2_READ)
    fid.write(u2, f"{NAME_2_READ}_2")
    
root_print(COMM, SEP)
