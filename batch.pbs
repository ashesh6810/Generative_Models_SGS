#!/bin/bash -l
### Job Name
#PBS -N 2DFT
### Charging account
#PBS -A URIC0004 
### Request two resource chunks, each with 4 CPUs, GPUs, MPI ranks, and 40 GB of memory
#PBS -l select=1:ncpus=1:ngpus=4:mem=100GB
### Specify that the GPUs will be V100s
#PBS -l gpu_type=v100
##PBS -l gpu_type=gp100
### Allow job to run up to 12 hour
#PBS -l walltime=24:00:00
### Route the job to the casper queue
#PBS -q casper
### Join output and error streams into single file
#PBS -j oe
### Send email on abort, begin and end
#PBS -m abe

module load python/3.7.9
module load ncarenv/1.3
module load gnu/9.1.0
module load ncarcompilers/0.5.0
module load netcdf/4.7.4
module load openmpi/4.1.0
ncar_pylib my_npl_clone3 
module load cuda

python -u -m RK4_Unet_implicit_energy_TL.py 



