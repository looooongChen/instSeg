import instSeg
import numpy as np
import os
import cv2
from random import shuffle
from tensorflow import keras
from data_loader import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='ccdb6843', help='experiment dataset')
parser.add_argument('--net', default='unet2', help='experiment architecture')
args = parser.parse_args()


config = instSeg.ConfigParallel(image_channel=1)
if args.ds.lower() == 'bbbc010':
    config.H = 448
    config.W = 448
elif args.ds.lower() == 'occ2014':
    config.H = 320
    config.W = 320
else:
    config.H = 512
    config.W = 512
config.filters = 32
config.nstage = 5
if args.net.lower() == 'unet2':
    config.modules = ['foreground']
else:
    config.modules = ['foreground', 'contour']
config.loss['foreground'] = 'crossentropy' 
config.loss['contour'] = 'crossentropy' 
config.train_learning_rate = 1e-4
config.lr_decay_rate = 0.8
config.lr_decay_period = 10000
config.backbone = 'unet'
config.input_normalization = 'per-image'
config.net_normalization = None
config.dropout_rate = 0.2
config.weight_decay = 0
config.save_best_metric = 'dice'

# model_dir = './model_unet3_bbbc010'
model_dir = './model_'+args.net.lower()+'_'+args.ds.lower()
epoches = 150
val_split = 0.1

imgs, masks = load_data_train(args.ds.lower(), (config.W, config.H))

# get foreground mask
fgs = {k: np.sum(v, axis=0) > 0 for k, v in masks.items()}
masks = {k: np.moveaxis(np.array(v), 0, -1) for k, v in masks.items()}
if args.net.lower() == 'unet3':
    contours = {k: instSeg.utils.contour(np.expand_dims(v, axis=0), radius=1, process_disp=False) for k, v in masks.items()}

# get train-val split
keys = list(imgs.keys())
shuffle(keys)
split = int(0.9*len(keys))
keys_train = keys[:split]
keys_val = keys[split:]

X_train = np.expand_dims(np.array([imgs[k] for k in keys_train]), axis=-1)
fg_train =  np.expand_dims(np.array([fgs[k] for k in keys_train], np.uint8), axis=-1)
if args.net.lower() == 'unet3':
    contour_train = np.squeeze(np.array([contours[k] for k in keys_train], np.uint8), axis=1)
    ds_train = {'image': X_train, 'foreground': fg_train, 'contour': contour_train}
else:
    ds_train = {'image': X_train, 'foreground': fg_train}

X_val = np.expand_dims(np.array([imgs[k] for k in keys_val]), axis=-1)
fg_val =  np.expand_dims(np.array([fgs[k] for k in keys_val], np.uint8), axis=-1)
if args.net.lower() == 'unet3':
    contour_val = np.squeeze(np.array([contours[k] for k in keys_val], np.uint8), axis=1)
    ds_val = {'image': X_val, 'foreground': fg_val, 'contour': contour_val}
else:
    ds_val = {'image': X_val, 'foreground': fg_val}

print('training set: ')
for k, d in ds_train.items():
    print(k+': ', d.shape)
print('validation set: ')
for k, d in ds_train.items():
    print(k+': ', d.shape)

del imgs, fgs, masks
if args.net.lower() == 'unet3':
    del contours

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
model.train(ds_train, ds_val, batch_size=4, epochs=epoches, augmentation=False)

