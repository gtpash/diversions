# Parallel I/O in FEniCS

These scripts are intended to test the parallel I/O capabilities of legacy [`FEniCS`](https://fenicsproject.org/), that is version `2019.1.0`. To run the codes in this diretory, you will need a working environment with `dolfin`, `numpy`, and `mpi4py`.

## Lagrange Elements
Standard Lagrange elements can be tested by first running, for example
```
mpirun -np 7 python3 test_io_write.py
```
The XMDF read back may be tested by running:
```
mpirun -np 3 python3 test_io_read_xdmf.py
```
This may pass in serial, but likely fails in parallel. Alternatively, one may test HDF5 functionality similarly:
```
mpirun -np 3 python3 test_io_read_h5.py --read_mesh
```
HDF5 should succeed.

## Mixed Finite Elements
XMDF does not support mixed elements, so we instead just test HDF5. Generate the data
```
mpirun -np 3 python3 test_mixed_io_write.py
```
and read the data back in
```
mpirun -np 7 python3 test_mixed_io_read.py --read_mesh
```

### Notes
- It is important to use a `dl.FunctionAssigner` to assign the individual components of the mixed element appropriately by interpolating a known function on to each individual function space comprising the mixed space.
- These scripts also demonstrate the need to read the mesh back from the same file as the data. When reading the mesh from the saved `.xdmf` file, the mesh partitioner may be inconsistent and lead to garbage.
