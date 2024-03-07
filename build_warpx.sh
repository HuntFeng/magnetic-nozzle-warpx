#!/bin/bash
echo -n "Select backend (type number): 
1) CPU (OMP)
2) GPU (CUDA)
"
read choice
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


echo -n "Create virtualenv (y/n)?"
read create_venv
echo -n "Virtualenv path: $HOME/.venvs/warpx (if not, type your path)?"
read venv_path
if [[ $venv_path == "" ]]; then
  venv_path=$HOME/.venvs/warpx
fi
if [[ $create_venv == "y" ]]; then
  echo "Remove existing warpx venv with the same path"
  rm -rf $venv_path
  virtualenv $venv_path
  source $venv_path/bin/activate
  pip install matplotlib tqdm jupyter h5py
  echo "Remove cache from $(which pip)"
  python -m pip cache purge
else
  source $venv_path/bin/activate
fi

# compile warpx
# enable python binding, openpmd (hdf5) output format
echo -n "WarpX path is $HOME/WarpX (if not, type your path)?"
read warpx_path
if [[ $warpx_path == "" ]]; then
  warpx_path=$HOME/WarpX
fi
echo "Remove build cache"
rm -rf $warpx_path/build
echo "Compile warpx with branch $(cd $warpx_path; git rev-parse --abbrev-ref HEAD)"
cmake -S $warpx_path -B $warpx_path/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=$backend \
  -DWARX_MPI=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=ON \
  -DWarpX_PYTHON=ON

echo "Build warpx and do pip install"
cmake --build $warpx_path/build --target pip_install -j 2

