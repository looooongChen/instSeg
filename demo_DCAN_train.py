import instSeg
from skimage.io import imread, imsave
import numpy as np
import os
import csv
import time
import cv2
from random import shuffle
from tensorflow import keras

config = instSeg.ConfigParallel()
config.loss_semantic = 'crossentropy' 
config.loss_contour = 'binary_crossentropy'
config.contour_radius = 2
config.train_learning_rate = 1e-5
config.lr_decay_rate = 0.9
config.lr_decay_period = 20000
config.train_batch_size = 2
config.backbone = 'uNet'
config.save_best_metric == 'mAP'
model_dir = './model_DCAN'

X_train, y_train = [], [] # list of img/gt path for training
X_val, y_val = [], [] # list of img/gt path for test

val_ds = {'image': np.array(list(map(imread,X_val))),
            'instance': np.expand_dims(np.array(list(map(imread,y_val))), axis=-1)}
train_ds = {'image': np.array(list(map(imread,X_train))),
            'instance': np.expand_dims(np.array(list(map(imread,y_train))), axis=-1)}

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
model.train(train_ds, val_ds, batch_size=4, epochs=300)

