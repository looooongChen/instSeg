import numpy as np
import os
import cv2
from random import shuffle
from tensorflow import keras
from data_loader import *
import argparse

from tqdm import tqdm
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D, StarDistData2D


parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='bbbc010', help='experiment dataset')
args = parser.parse_args()

if args.ds.lower() == 'bbbc010':
    sz = (448, 448)
elif args.ds.lower() == 'occ2014':
    sz = (320, 320)
else:
    sz = (512, 512)

model_dir = './model_stardist'+'_'+args.ds.lower()
epoches = 300

imgs, masks = load_data_train(args.ds.lower(), sz)

keys = list(imgs.keys())
keys.sort()
keys_train = [keys[i] for i in range(len(keys)) if i%10 != 9]
keys_val = [keys[i] for i in range(len(keys)) if i%10 == 9]

X_trn = [imgs[k] for k in keys_train]
Y_trn = [labeled_random_overlap(masks[k]) for k in keys_train]
X_val = [imgs[k] for k in keys_val]
Y_val = [labeled_random_overlap(masks[k]) for k in keys_val]

del imgs, masks

axis_norm = (0,1)
X_trn = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_trn)]
X_val = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_val)]
Y_trn = [fill_label_holes(y) for y in tqdm(Y_trn)]
Y_val = [fill_label_holes(y) for y in tqdm(Y_val)]

conf = Config2D(n_rays=32, grid=(2,2), use_gpu=True, n_channel_in=1)
# from csbdeep.utils.tf import limit_gpu_memory
# limit_gpu_memory(0.8)

model = StarDist2D(conf, name='stardist', basedir=model_dir)
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), epochs=epoches, steps_per_epoch=len(X_trn)//4)


