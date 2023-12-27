module purge
module load anaconda3
module load MistEnv/2021a cuda/11.7.1 gcc/10.3.0 openmpi/4.1.1+ucx-1.10.0 cmake hdf5/1.10.7
module list
source activate warpx_gpu
