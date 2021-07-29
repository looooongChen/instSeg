import instSeg
from skimage.io import imread, imsave
import numpy as np
import os
import csv
import time
import cv2
from random import shuffle
from tensorflow import keras
import csv

config = instSeg.ConfigParallel(image_channel=3)
config.modules = ['embedding', 'edt']
config.loss_embedding = 'cos' 
config.edt_loss = 'mse'
config.embedding_include_bg = True
config.train_learning_rate = 1e-4
config.lr_decay_rate = 0.9
config.lr_decay_period = 20000
config.backbone = 'uNet'

# X_train: input images, N x H x W x C
# y_train: output masks, N x H x W x 1
X_train = np.array(X_train[:-val_split])
y_train = np.expand_dims(np.array(y_train[:-val_split]), axis=-1)
ds_train = {'image': X_train, 'instance': y_train}

X_val = np.array(X_train[-val_split:])
y_val = np.expand_dims(np.array(y_train[-val_split:]), axis=-1)
ds_val = {'image': X_val, 'instance': y_val}

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
model.train(ds_train, ds_val, batch_size=2, epochs=epoches, augmentation=False)

