# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/ngp/NeRF/nerf_hybrid.py
PYTHONPATH='.' python plenoxels/main.py --config-path  plenoxels/configs/ngp/Rodin/nerf_hybrid.py
PYTHONPATH='.' mpiexec -n 1 python plenoxels/main.py --config-path  plenoxels/configs/ngp/RodinSubject/nerf_hybrid.py --ddp 1