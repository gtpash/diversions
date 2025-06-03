# serial io works from without dof ordering 
python save_mixed_mesh.py -m save  
python save_mixed_mesh.py -m load

# parallel io needs the additional saved dof_order 
mpirun -n 8 python save_mixed_mesh.py -m save  
python save_mixed_mesh.py -m load
