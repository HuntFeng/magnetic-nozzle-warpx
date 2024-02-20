#!/bin/bash
echo -n "Select backend (type number): 
1) CPU (OMP)
2) GPU (CUDA)
"
read choice
# source the newly created virtualenv
if [[ $choice == "1" ]]; then
  source $SCRATCH/magnetic-nozzle-warpx/warpx.profile --cpu
  backend="OMP"
elif [[ $choice == "2" ]]; then
  source $SCRATCH/magnetic-nozzle-warpx/warpx.profile --gpu
  backend="CUDA"
else
  echo "Please enter a valid choice"
  exit 0
fi

# compile warpx
# enable python binding, openpmd (hdf5) output format
warpx=WarpX-23.11
echo "compile $warpx"
cmake -S $HOME/$warpx -B $HOME/$warpx/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=$backend \
  -DWARX_MPI=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=ON \
  -DWarpX_PYTHON=ON

echo "build warpx and do pip install"
cmake --build $HOME/$warpx/build --target pip_install -j 4

