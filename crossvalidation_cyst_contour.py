
import instSeg
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from instSeg.post_process import *
import os

config = instSeg.ConfigContour()
# config.loss_semantic = 'focal_loss'
run_name = 'cystJKI_contour_dice'
ds_dir = './ds_cystJKI'
base_dir = './'

#### cross validation ####
for fold in range(5):
    model = instSeg.InstSegContour(config=config, base_dir=base_dir, run_name=run_name + '_crossval_'+str(fold))
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    with open(os.path.join(ds_dir, 'crossval_partition.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if int(row[0]) == fold:
                X_val.append(os.path.join(ds_dir, row[1][2:]))
                Y_val.append(os.path.join(ds_dir, row[2][2:]))
            else:
                X_train.append(os.path.join(ds_dir, row[1][2:]))
                Y_train.append(os.path.join(ds_dir, row[2][2:]))

    train_data = {'image': list(map(cv2.imread,X_train)),
                  'instance': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_train))}
    val_data = {'image': list(map(cv2.imread,X_val)),
                'instance': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_val))}

    model.train(train_data, val_data, batch_size=4, epochs=250, augmentation=False)
