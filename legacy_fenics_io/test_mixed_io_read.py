##################################################
# This script is intended to test parallel I/O for legacy FEniCS
##################################################

import os
import argparse
from mpi4py import MPI
import dolfin as dl
import ufl
import numpy as np
from utils import root_print

def compute_l2_error(u, u_true):
    l2_error = np.sqrt(dl.assemble(ufl.inner(u - u_true, u - u_true)*ufl.dx))
    return l2_error

def main(args):
# MPI setup.
    COMM = MPI.COMM_WORLD
    rank = COMM.rank
    nproc = COMM.size

    # General setup.
    OUTPUT_DIR = "output"
    H5FILE = os.path.join(OUTPUT_DIR, "test_mixed_h5.h5")
    MESHFILE = os.path.join(OUTPUT_DIR, "box.xdmf")
    NAME_2_READ = "test"

    SEP = 80*"#"
    root_print(COMM, SEP)
    root_print(COMM, f"There are {nproc} processes in total.")

    if args.read_mesh:
        mesh = dl.Mesh(COMM)
        with dl.HDF5File(COMM, H5FILE, "r") as fid:
            fid.read(mesh, "mesh", False)
    else:
        # by default, read the mesh from the XDMF file.
        # This may fail, depending on the mesh partitioner.
        
        mesh = dl.Mesh(COMM)
        with dl.XDMFFile(COMM, MESHFILE) as fid:
            fid.read(mesh)
        
        # mesh = dl.UnitSquareMesh(COMM, 10, 10)
        # mesh = dl.UnitCubeMesh(COMM, 10, 10, 10)
    
    # set up function space and true solutions.
    Vh = dl.FunctionSpace(mesh, "CG", 1)
    mixed_element = ufl.MixedElement([Vh.ufl_element(), Vh.ufl_element()])
    Vhm = dl.FunctionSpace(mesh, mixed_element)

    u1true_expr = dl.Expression("x[0]", degree=2)
    u1true = dl.interpolate(u1true_expr, Vh)

    u2true_expr = dl.Expression("x[1]", degree=2)
    u2true = dl.interpolate(u2true_expr, Vh)
    
    assigner = dl.FunctionAssigner(Vhm, [Vh, Vh])
    
    utrue = dl.Function(Vhm)
    assigner.assign(utrue, [u1true, u2true])

    uh5 = dl.Function(Vhm)

    # Load the solution from HDF5.
    with dl.HDF5File(COMM, H5FILE, "r") as fid:
        fid.read(uh5, NAME_2_READ)

    # plot the first component of the read back solution.
    if args.plot:
        import matplotlib.pyplot as plt
        dl.plot(uh5.sub(0))
        plt.savefig(os.path.join(OUTPUT_DIR, "first_component_read_back.png"))

    l2_error = compute_l2_error(uh5, utrue)
    
    assert l2_error < 1e-12, "HDF5 read back of first checkpoint failed."

    root_print(COMM, "Read back from HDF5 was: SUCCESSFUL.")
    root_print(COMM, SEP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reading from XDMF and HDF5.")
    
    parser.add_argument("--read_mesh", action=argparse.BooleanOptionalAction, default=False, help="Read the mesh from the file?")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot the read back solution?")
    
    args = parser.parse_args()
    
    main(args)