# Magnetic Nozzle PIC Simulation in Cylindrical Coordinate

## Installing the dependencies

We are going to use virtualenv to install all things
### Load Niagara modules
```
module purge
module load NiaEnv/2019b gcc/8.3.0 intelmpi/2019u5 python/3.9.8
```

### Create virtualenv
```
virtualenv --system-site-packages $HOME/.venvs/warpx
source $HOME/.venvs/warpx/bin/activate
pip install cmake mpi4py numpy scipy tqdm matplotlib jupyter yt
```
- if `mpi4py` installation is not okay, remove the pip cache and reinstall it.
```
pip unstall mpi4py
pip cache remove mpi4py
pip install mpi4py
```

### Compile warpx
Clone the repo first, we are using version 23.11
```
https://github.com/ECP-WarpX/WarpX
```
Put this into a bash script and run it,
```
# compile warpx
# the -S and -B flag must in same line
# one option should also in the same line as cmake, otherwise the command fails sometime
echo "compile warpx"
cmake -S $HOME/warpx -B $HOME/warpx/build -DWarpX_DIMS=RZ \
  -DWarpX_COMPUTE=OMP \
  -DWARX_MPI=ON \
  -DWarpX_LIB=ON \
  -DWarpX_QED=OFF \
  -DWarpX_OPENPMD=OFF \
  -DWarpX_PYTHON=ON

echo "build warpx and do pip install"
# these env vars are for pywarpx building
export PYWARPX_LIB_DIR=$HOME/warpx/build/lib
export WARPX_DIMS=RZ
export WARPX_COMPUTE=OMP
export WARPX_MPI=ON
export WARPX_OPENPMD=OFF
export WARPX_QED=OFF
cmake --build $HOME/warpx/build --target pip_install -j 4
```

## Running

### Testing and debugging
```
mpirun -np <tasks> python picmi-input.py
```

### Use Slurm
Put this into a slurm script
```
#!/bin/bash
## Ask for 2 nodes on the cluster
#SBATCH --nodes=2
## Ask for a total of 32 MPI tasks
#SBATCH --ntasks=32
## Ste a run time limit
#SBATCH --time=0:30:00
# sends mail when process begins, and when it ends. Make sure you define your email 
#SBATCH --mail-type=end 
#SBATCH --mail-user=hunt.feng@usask.ca 

source $HOME/warpx.profile 
source $HOME/.venvs/warpx/bin/activate
cd $SCRATCH/magnetic-nozzle-warpx
mpirun -np <tasks> python <picmi-input>.py >& output.log
```
