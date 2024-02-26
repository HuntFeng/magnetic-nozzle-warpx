#!/bin/bash
echo -n "Select backend (type number): 
1) CPU (OMP)
2) GPU (CUDA)
"
read choice

echo -n "Need to create virtualenv for python? (y/n)"

read build_venv

# source the newly created virtualenv
if [[ $choice == "1" ]]; then
  module purge
  module load NiaEnv/2019b intel/2020u4 intelmpi/2020u4 python/3.11.5 ffmpeg/3.4.2 hdf5/1.10.7
  module list
  backend="OMP"
elif [[ $choice == "2" ]]; then
  module purge
  module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 hdf5-mpi/1.14.2 python/3.11.5 mpi4py/3.1.4
  module list
  backend="CUDA"
else
  echo "Please enter a valid choice"
  exit 0
fi

if [[ $build_venv == "y" ]]; then
  echo "remove existing warpx venv"
  rm -rf $HOME/.venvs/warpx
  virtualenv --system-site-packages $HOME/.venvs/warpx
  source $HOME/.venvs/warpx/bin/activate
  pip install matplotlib tqdm jupyter
else
  source $HOME/.venvs/warpx/bin/activate
fi
echo "remove cache from $(which pip)"
python -m pip cache purge

# compile warpx
# enable python binding, openpmd (hdf5) output format
warpx=WarpX-23.11
echo "remove build cache"
rm -rf $HOME/$warpx/build
echo "compile $warpx"
cmake -S $HOME/$warpx -B $HOME/$warpx/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=$backend \
  -DWARX_MPI=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=ON \
  -DWarpX_PYTHON=ON

echo "build warpx and do pip install"
cmake --build $HOME/$warpx/build --target pip_install -j 8

