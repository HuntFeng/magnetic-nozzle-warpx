#!/bin/bash

# create virtualenv
read -p "want to create virtualenv for warpx? (y/n)" createvenv
if [ $createvenv == "y" ]
then 
  virtualenv --system-site-packages ~/.venvs/warpx
  source ~/.venvs/warpx/bin/activate
  echo "installing using $(which pip)"
  pip install cmake mpi4py tqdm matplotlib jupyter yt 
else
  source ~/.venvs/warpx/bin/activate
fi


# compile warpx
warpx=WarpX-23.11
echo "compile $warpx"
cmake -S $HOME/$warpx -B $HOME/$warpx/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=OMP \
  -DWARX_MPI=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=ON \
  -DWarpX_PYTHON=ON

echo "build warpx and do pip install"
# Setting this indicates to pywarpx that it should refer to an
# already built warpx library rather than compiling it for itself
# export PYWARPX_LIB_DIR=$HOME/$warpx/build/lib
# these env vars are for pywarpx building
# export WARPX_DIMS=RZ
# export WARPX_COMPUTE=OMP
# export WARPX_MPI=ON
# export WARPX_OPENPMD=OFF
# export WARPX_QED=OFF
echo | which pip
cmake --build $HOME/$warpx/build --target pip_install -j 4

