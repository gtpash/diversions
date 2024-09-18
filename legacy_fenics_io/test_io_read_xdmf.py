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
    XDMFFILE = os.path.join(OUTPUT_DIR, "test_xdmf.xdmf")
    NAME_2_READ = "test"

    SEP = 80*"#"
    root_print(COMM, SEP)
    root_print(COMM, f"There are {nproc} processes in total.")

    if args.read_mesh:
        mesh = dl.Mesh()
        with dl.XDMFFile(COMM, XDMFFILE) as fid:
            fid.read(mesh)
    else:
        mesh = dl.Mesh()
        with dl.XDMFFile(COMM, MESHFILE) as fid:
            fid.read(mesh)
        # mesh = dl.UnitSquareMesh(COMM, 10, 10)
        # mesh = dl.UnitCubeMesh(COMM, 10, 10, 10)
    
    # set up function space and true solutions.
    Vh = dl.FunctionSpace(mesh, "CG", 1)

    u1true_expr = dl.Expression("x[0]", degree=2)
    u1true = dl.interpolate(u1true_expr, Vh)

    u2true_expr = dl.Expression("x[1]", degree=2)
    u2true = dl.interpolate(u2true_expr, Vh)
    
    uxdmf = dl.Function(Vh)
    u2xdmf = dl.Function(Vh)
    
    # Load the solution from XDMF.
    with dl.XDMFFile(COMM, XDMFFILE) as fid:
        fid.read(mesh)
        fid.read_checkpoint(uxdmf, NAME_2_READ, 0)
        fid.read_checkpoint(u2xdmf, NAME_2_READ, 1)

    l2_error = compute_l2_error(uxdmf, u1true)
    assert l2_error < 1e-12, "XDMF read back of first checkpoint failed."

    l2_error = compute_l2_error(u2xdmf, u2true)
    assert l2_error < 1e-12, "XDMF read back of second checkpoint failed."

    root_print(COMM, "Read back from XDMF was: SUCCESSFUL.")
    root_print(COMM, SEP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reading from XDMF and HDF5.")
    
    parser.add_argument("--read_mesh", action=argparse.BooleanOptionalAction, default=False, help="Read the mesh from the file?")
    
    args = parser.parse_args()
    
    main(args)