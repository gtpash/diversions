import argparse
import dolfin as dl 
import numpy as np 
import matplotlib.pyplot as plt 
from mpi4py import MPI 


class DofOrdering:
    def __init__(self, Vh, filename):
        self.Vh = Vh 
        self.dof_order = dl.Function(Vh)
        self.comm = Vh.mesh().mpi_comm()
        self.filename = filename 

    def setup_on_process(self):
        dof_order_np = np.arange(self.Vh.dim())
        local_range = self.dof_order.vector().local_range()
        self.dof_order.vector().set_local(dof_order_np[local_range[0]:local_range[1]])
        self.dof_order.vector().apply("")

    def save(self):
        with dl.HDF5File(self.comm, self.filename, 'w') as fid:
            fid.write(self.dof_order, 'dof_order')

    def load(self):
        # self.dof_order.vector().zero()
        self.dof_order = dl.Function(self.Vh)
        with dl.HDF5File(self.comm, self.filename, 'r') as fid:
            fid.read(self.dof_order, 'dof_order')

    def as_function(self):
        return self.dof_order 

    def as_numpy(self): 
        return np.array(self.dof_order.vector().get_local(), dtype=int)



def save_function(comm, filename, fun):
    with dl.HDF5File(comm, filename, 'w') as fid:
        fid.write(fun, 'fun')

def load_function(comm, filename, fun):
    with dl.HDF5File(comm, filename, 'r') as fid:
        fid.read(fun, 'fun')

def plot_mixed_function(u, title=""):
    plt.figure(figsize=(12,6))
    plt.subplot(131)
    tri = dl.plot(u.split()[0].split()[0])
    cb = plt.colorbar(tri, shrink=0.6)
    plt.title(title)
    plt.subplot(132)
    tri = dl.plot(u.split()[0].split()[1])
    cb = plt.colorbar(tri, shrink=0.6)
    plt.title(title)
    plt.subplot(133)
    tri = dl.plot(u.split()[1])
    cb = plt.colorbar(tri, shrink=0.6)
    plt.title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, choices=['save', 'load'], default='save')
    parser.add_argument('--mesh_type', '-t', type=str, choices=['square', 'object'], default='object')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD 
    u_expr = dl.Expression(("sin(4*x[0]) * sin(4*x[1])", "cos(8*x[0]) * cos(8*x[1])", "sin(x[0]) * cos(x[1])"), degree=4)
    
    if args.mesh_type == 'square': 
        mesh = dl.UnitSquareMesh(20, 20)
    elif args.mesh_type == 'object': 
        mesh = dl.Mesh(comm) 
        with dl.XDMFFile(comm, "mesh.xdmf") as fid:
            fid.read(mesh)
    else:
        raise ValueError("mesh needs to be square of object")


    P2 = dl.VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = dl.FiniteElement("CG", mesh.ufl_cell(), 1)
    ME = dl.MixedElement([P2, P1])

    Vh = dl.FunctionSpace(mesh, ME)
    u = dl.Function(Vh)

    # Save dof ordering 
    save_dir = "function"

    if args.mode == "save":
        u.interpolate(u_expr)
        
        # First save the function as hdf5 
        save_function(comm, f"{save_dir}/u.h5", u)
        
        # Then save with gather on zero 
        u_np = u.vector().gather_on_zero()
        if comm.Get_rank() == 0:
            np.save(f"{save_dir}/u.npy", u_np)
        else:
            pass 
        
        # Save the dof ordering 
        dof_order = DofOrdering(Vh, f'{save_dir}/dof_order.h5')
        dof_order.setup_on_process()
        dof_order.save()

    else:
        load_function(comm, f"{save_dir}/u.h5", u)
        u_np = np.load(f"{save_dir}/u.npy")
            
        # Plot the true function loaded from hdf5 
        plot_mixed_function(u, "From hdf5")
        
        # Plot the function when directly setting local with numpy save 
        # Will be jumbled on different meshes
        u.vector().set_local(u_np)
        plot_mixed_function(u, "From numpy")

        # Load the dof ordering and reorder the indices according to this 
        # Then plot 
        dof_order = DofOrdering(Vh, f"{save_dir}/dof_order.h5")
        dof_order.load()
        dof_order_np = dof_order.as_numpy()
        u.vector().set_local(u_np[dof_order_np])
        plot_mixed_function(u, "From numpy, sorted")

        plt.show()


