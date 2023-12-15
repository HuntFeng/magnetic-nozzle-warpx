#!/bin/bash
# source the newly created virtualenv
source ~/.venvs/warpx/bin/activate

# compile warpx
# enable python binding, openpmd (hdf5) output format
warpx=WarpX-23.11
echo "compile $warpx"
cmake -S $HOME/$warpx -B $HOME/$warpx/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=OMP \
  -DWARX_MPI=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=ON \
  -DWarpX_PYTHON=ON

echo "build warpx and do pip install"
cmake --build $HOME/$warpx/build --target pip_install -j 4

