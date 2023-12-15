# Magnetic Nozzle PIC Simulation in Cylindrical Coordinate

## Installing the dependencies

We are going to use virtualenv to install all things
### Load Niagara modules
```
module purge
module load NiaEnv/2019b intel/2020u4 intelmpi/2020u4 python/3.11.5 ffmpeg/3.4.2 hdf5/1.10.7
```
- hdf5 is to enable WarpX hdf5 output.
- ffmpeg is for making animes. If not planning to make animes, this is optional.
- `match...case...` syntax is used in `post_processing.py`, python version must be `>3.10`.

### Create virtualenv
```
virtualenv --system-site-packages $HOME/.venvs/warpx
source $HOME/.venvs/warpx/bin/activate
pip install cmake tqdm matplotlib jupyter
```
- cmake is required in the virtualenv since warpx will be compiled and build using cmake.

### Compile WarpX
Download (WarpX-23.11)[https://github.com/ECP-WarpX/WarpX] to Home directory and rename it to `WarpX-23.11`, then use the `build-warpx.sh` to build WarpX. It is configured to enable RZ coordinate, OMP, python binding (pywarpx), and openpmd (hdf5) output format.

## Running

### Testing and debugging
To run the simulation, simply do
```
mpirun -np <tasks> python run_simulation.py
```
- The script will create a folder with format `diagsyyyymmddHHMMSS`. The diagnostics are in this folder.
- If want to diagnose the wall time per step using the `post_processing.py`, set the warpx verbosity to 1 
```
picmi.Simulation(verbose=1)
```
the output the warpx standard output a `.log` file and place into the diagnostics folder.
```
mpirun -np <tasks> python run_simulation.py > output.log
# when simulation is done
mv output.log <diags_folder>
```

### Use Slurm
To submit a job to Slurm. simply use the slurm script `nozzle`,
```
sbatch nozzle
```
- Change the system variables as you wish, since every HPC has different directory stuctures.
- Again, if want to diagnose the wall time per step, remember to move the output log to the diags folder
```
mv output*.log <diags_folder>
```

## Post-Processing
The `post_processing.py` script has everything ready for you.
- To make animes, simply do
```
python post_processing.py <diags_folder>
```
- To draw graphs of certain fields at certain step, check the usage in `analysis.ipynb`.

## Files
- `params.py`: Record the parameters used in simulation. 
- `util.py`: Usefule physics formulas are in here.
- `solver.py`: Custom Poisson solvers.