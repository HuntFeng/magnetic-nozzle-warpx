while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) 
        echo "-c | --cpu: Load cpu profile"
        echo "-g | --gpu: Load gpu profile"
        return
        ;;
        -c|--cpu)
        module purge
        module load NiaEnv/2019b intel/2020u4 intelmpi/2020u4 python/3.11.5 ffmpeg/3.4.2 hdf5/1.10.7
        module list
        source ~/.venvs/warpx/bin/activate
        return
        ;;
        -g|--gpu)
        module purge
        module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.2 hdf5-mpi/1.14.2 python/3.11.5 mpi4py/3.1.4
        module list
        # source ~/.venvs/warpx/bin/activate
        source ~/.venvs/warpx_test/bin/activate
        return
        ;;
        *)
        echo "Invalid option: $1"
        echo "-c | --cpu: Load cpu profile"
        echo "-g | --gpu: Load gpu profile"
        return 1
        ;;
    esac
done
    
