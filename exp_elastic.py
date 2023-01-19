import instSeg
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
from skimage.measure import label as relabel
import argparse
from PNS import read_pns_cyst, read_patches
import random
import csv
import os


scale = [16, 32, 64, 128, 256]
strength = [0.1, 0.2, 0.3, 0.4, 0.5]
# strength = [2,4,8,16,32,64]

model_dir = './models_elastic_test'

sz = 512
distance = 64

g_config = instSeg.Generator_Config()
g_config.img_sz = sz
g_config.obj_sz = distance # interger or 'random'
g_config.max_degree = 10
g_config.pattern = 'grid' # 'grid' or 'random'
g_config.rotation = False
g_config.translation = False
g_config.shape = 'circle'
g_config.obj_color = (254, 208, 73)
g_config.bg_color = (0,0,0)

g = instSeg.Pattern_Generator(g_config)

X_train, y_train = [], []
for idx in range(10):
    img = g.generate()
    X_train.append(img)
    mask = relabel(img[:,:,0] > 0)
    y_train.append(mask)

ds_train = {'image': X_train, 'instance': y_train}

for Sc in scale:
    for St in strength:

        model_dir_run = os.path.join(model_dir, "scale{}_strength{}".format(Sc, St))
    
        ## basic setting
        config = instSeg.Config(image_channel=3)
        config.H = sz
        config.W = sz
        config.backbone = 'unet'
        config.filters = 16
        config.nstage = 2
        config.modules = ['layered_embedding']
        config.embedding_dim = 8
        config.loss['layered_embedding'] = 'cos' 

        config.flip = False
        config.random_rotation = False
        config.random_shift = False
        config.random_blur = False
        config.random_saturation = False
        config.random_hue = False
        config.random_gamma = False

        config.elastic_deform = True
        config.deform_scale = Sc
        config.deform_strength = St

        config.summary_step = 1

        ## training data setting
        model = instSeg.Model(config=config, model_dir=model_dir_run)
        model.train(ds_train, None, batch_size=1, epochs=10, augmentation=True)






