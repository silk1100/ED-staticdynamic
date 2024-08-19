#!/bin/bash
#
# CREATED USING THE BIOHPC PORTAL on Fri Dec 01 2023 19:00:51 GMT-0600 (Central Standard Time)
#
# This file is batch script used to run commands on the BioHPC cluster.
# The script is submitted to the cluster using the SLURM `sbatch` command.
# Lines starting with # are comments, and will not be run.
# Lines starting with #SBATCH specify options for the scheduler.
# Lines that do not start with # or #SBATCH are commands that will run.

# Name for the job that will be visible in the job queue and accounting tools.
#SBATCH --job-name NewJob

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB       # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

# Memory (RAM) requirement/limit in MB.

# Time limit for the job in the format Days-H:M:S
# A job that reaches its time limit will be cancelled.
# Specify an accurate time limit for efficient scheduling so your job runs promptly.
#SBATCH -t 0-2:0:0

# The standard output and errors from commands will be written to these files.
# %j in the filename will be replace with the job number when it is submitted.
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Send an email when the job status changes, to the specfied address.
#SBATCH --mail-type ALL
#SBATCH --mail-user mohamed.ali@utsouthwestern.edu

module load python

# COMMAND GROUP 1
cd /home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src


# COMMAND GROUP 2
/home2/s223850/.conda/envs/ED/bin/python polars_scripts/data_prepare.py



# END OF SCRIPT
