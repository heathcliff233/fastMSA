#!/bin/bash
#SBATCH -J msacath
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=16 
#SBATCH -n 16
#SBATCH --gres=gpu:0
#SBATCH --get-user-env #SBATCH -e job-%j.err #SBATCH -o job-%j.out

CDIR=/share/hongliang/casp14-40-ur9018-db/
ODIR=/user/hongliang/mydpr/hmmer-batch/ur90/40/

time (/user/hongliang/mydpr/hmmer-batch/jack.sh $CDIR $ODIR)

echo Completed!

