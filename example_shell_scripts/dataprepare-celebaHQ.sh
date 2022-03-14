#!/bin/bash

#prepare virtual environment
source ../../ENV/bin/activate

#Variables
target='celebaHQ'
sources='celeba_noise'
lamb=1000.
theta=0.75
seed=-1
critters=100
num_crit=12
model='sndcgan'
bs=64
eval_freq=12
logfile='./experiment_history.pkl'
exp_name='celebaHQ'
data="$SLURM_TMPDIR"
num_workers=4
numsample=3000
cp=10
cp_folder='./results/celebaHQ/20210531212748/'

timestamp=`date '+%Y%m%d%H%M%S'`
results_folder="./results/$target/$timestamp/"
target_folder="../../projects/def-nachman2/$target"

cp "$target_folder/celebaHQ128.zip" $SLURM_TMPDIR/
unzip -q "$SLURM_TMPDIR/celebaHQ128.zip" -d "$SLURM_TMPDIR/"
mv "$SLURM_TMPDIR/celebaHQ128" "$SLURM_TMPDIR/celebaHQ"
mkdir "$SLURM_TMPDIR/celebaHQtest"
cp "$SLURM_TMPDIR/celebaHQ/test/pics/"* "$SLURM_TMPDIR/celebaHQtest/"


