#!/bin/bash

phase="evaluation"
architecture="d9"
dist_branch=True
include_bg=True
embedding_dim=8
losstype="long_loss"

train_dir="/work/scratch/wu/interweight/tfrecords/CVPPP2017/train"
validation=True
val_dir="/work/scratch/wu/interweight/tfrecords/CVPPP2017/val"
image_depth="uint8"
image_channels=3
model_dir="./model_dim"

lr=0.0001
batch_size=4
training_epoches=600


cd /work/scratch/wu/leafex/dim/repro

/home/students/wu/miniconda3/envs/tf15/bin/python /work/scratch/wu/leafex/dim/repro/main.py \
			--phase="$phase" \
			--architecture="$architecture" \
			--dist_branch="$dist_branch" \
			--include_bg="$include_bg" \
			--embedding_dim="$embedding_dim" \
        --losstype="$losstype" \
			--train_dir="$train_dir" \
			--validation="$validation" \
			--val_dir="$val_dir"\
			--image_depth="$image_depth" \
			--image_channels="$image_channels" \
			--model_dir="$model_dir" \
			--lr="$lr" \
			--batch_size="$batch_size" \
			--training_epoches="$training_epoches"
