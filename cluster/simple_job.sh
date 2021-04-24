#!/bin/sh
#SBATCH -J catnap #job name
#SBATCH --time=07-00:00:00 #requested time (DD-HH:MM:SS)
#SBATCH -p preempt
#SBATCH -N 1   #1 nodes
#SBATCH -n 1   
#SBATCH -c 2   #1 cpu cores per task
#SBATCH --gres=gpu:t4:4
#SBATCH --chdir=/cluster/home/skrieg01
#SBATCH --mem=8g
#SBATCH --output=MyJob.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=MyJob.%j.%N.err   #saving standard error to file, %j=JOBID, %N=NodeName

module load singularity/3.6.1
singularity exec --nv --writable-tmpfs image_latest.sif python conditional-growth/experiments/distance_traveled/optimize_sim.py
