#!/bin/bash

#Variables
target='celebaHQ'
data="$SLURM_TMPDIR"

target_folder="/home/tmilne5/projects/def-nachman2/$target"

cp "$target_folder/celebaHQ128.zip" $SLURM_TMPDIR/
unzip -q "$SLURM_TMPDIR/celebaHQ128.zip" -d "$SLURM_TMPDIR/"
mv "$SLURM_TMPDIR/celebaHQ128" "$SLURM_TMPDIR/celebaHQ"
mkdir "$SLURM_TMPDIR/celebaHQtest"
cp "$SLURM_TMPDIR/celebaHQ/test/pics/"* "$SLURM_TMPDIR/celebaHQtest/"


