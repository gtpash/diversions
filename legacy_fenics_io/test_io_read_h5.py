##################################################
# This script is intended to test parallel I/O for legacy FEniCS
##################################################

import os
import argparse
import dolfin as dl
from mpi4py import MPI  # MUST be imported AFTER dolfin
import ufl
import numpy as np
from utils import root_print

def compute_l2_error(u, u_true):
    l2_error = np.sqrt(dl.assemble(ufl.inner(u - u_true, u - u_true)*ufl.dx))
    return l2_error

def main(args):
    # MPI setup.
    
    COMM = MPI.COMM_WORLD
    # COMM = dl.MPI.comm_world  # may be better to use the built in
    rank = COMM.rank
    nproc = COMM.size

    # General setup.
    OUTPUT_DIR = "output"
    MESHFILE = os.path.join(OUTPUT_DIR, "box.xdmf")
    H5FILE = os.path.join(OUTPUT_DIR, "test_h5.h5")
    H5FILE_1 = os.path.join(OUTPUT_DIR, "test_mult_1_h5.h5")
    H5FILE_2 = os.path.join(OUTPUT_DIR, "test_mult_2_h5.h5")
    NAME_2_READ = "test"

    SEP = 80*"#"
    root_print(COMM, SEP)
    root_print(COMM, f"There are {nproc} processes in total.")

    if args.read_mesh:
        mesh = dl.Mesh()
        with dl.HDF5File(COMM, H5FILE, "r") as fid:
            fid.read(mesh, "mesh", False)
    else:
        mesh = dl.Mesh()
        with dl.XDMFFile(COMM, MESHFILE) as fid:
            fid.read(mesh)
        # mesh = dl.UnitSquareMesh(COMM, 10, 10)
    
    # set up function space and true solutions.
    Vh = dl.FunctionSpace(mesh, "CG", 1)

    u1true_expr = dl.Expression("x[0]", degree=2)
    u1true = dl.interpolate(u1true_expr, Vh)

    u2true_expr = dl.Expression("x[1]", degree=2)
    u2true = dl.interpolate(u2true_expr, Vh)

    uh5 = dl.Function(Vh)
    u2h5 = dl.Function(Vh)

    # Load the solution from HDF5.
    with dl.HDF5File(COMM, H5FILE, "r") as fid:
        fid.read(uh5, NAME_2_READ)
        fid.read(u2h5, f"{NAME_2_READ}_2")

    l2_error = compute_l2_error(uh5, u1true)
    assert l2_error < 1e-12, "HDF5 read back of first checkpoint failed."

    l2_error = compute_l2_error(u2h5, u2true)
    assert l2_error < 1e-12, "HDF5 read back of second checkpoint failed."

    root_print(COMM, "Read back from HDF5 was: SUCCESSFUL.")
    
    # Load the solution back from separate HDF5 files, using only the mesh from the first.
    uh5.vector().zero()
    u2h5.vector().zero()
    
    with dl.HDF5File(COMM, H5FILE_1, "r") as fid:
        fid.read(uh5, NAME_2_READ)
        
    with dl.HDF5File(COMM, H5FILE_2, "r") as fid:
        fid.read(u2h5, f"{NAME_2_READ}_2")
    
    l2_error = compute_l2_error(uh5, u1true)
    assert l2_error < 1e-12, "HDF5 read back of first checkpoint failed."

    l2_error = compute_l2_error(u2h5, u2true)
    assert l2_error < 1e-12, "HDF5 read back of second checkpoint failed."

    root_print(COMM, "Read back from separate HDF5 was: SUCCESSFUL.")
    root_print(COMM, SEP)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reading from XDMF and HDF5.")
    
    parser.add_argument("--read_mesh", action=argparse.BooleanOptionalAction, default=False, help="Read the mesh from the file?")
    
    args = parser.parse_args()
    
    main(args)