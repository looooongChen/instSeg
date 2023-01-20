import instSeg
import numpy as np
import os
import cv2
from random import shuffle
from tensorflow import keras
from data_loader import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='bbbc010', help='experiment dataset')
# parser.add_argument('--net', default='unet2', help='experiment architecture')
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
config.modules = ['embedding']
config.loss['embedding'] = 'sparse_cos' 
config.neighbor_distance = 10
config.embedding_dim = 3
config.embedding_include_bg = False
config.train_learning_rate = 1e-4
config.lr_decay_rate = 0.9
config.lr_decay_period = 10000
config.backbone = 'unet'
config.input_normalization = 'per-image'
config.net_normalization = None
config.dropout_rate = 0
config.weight_decay = 0
config.save_best_metric = 'AP'

model_dir = './model_embeddingVis'+'_'+args.ds.lower()
epoches = 300

imgs, masks = load_data_train(args.ds.lower(), (config.W, config.H))

if args.ds.lower() == 'ccdb6843':
    masks = {k: labeled_non_overlap(m, overlap_label=0) for k, m in masks.items()}
    keys = list(imgs.keys())
    X_train = np.expand_dims(np.array([imgs[k] for k in keys]), axis=-1)
    y_train = np.expand_dims(np.array([masks[k] for k in keys]), axis=-1)
    ds_train = {'image': X_train, 'instance': y_train}
    del imgs, masks
else:
    keys = list(imgs.keys())
    imgs = [imgs[k] for k in keys]
    masks = [masks[k] for k in keys]
    ds_train = {'image': imgs, 'instance': masks}
# keys = list(imgs.keys())
# imgs = [imgs[k] for k in keys]
# masks = [masks[k] for k in keys]
# ds_train = {'image': imgs, 'instance': masks}

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
model.train(ds_train, batch_size=4, epochs=epoches, augmentation=False)

