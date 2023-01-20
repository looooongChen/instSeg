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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val_part', default='0', help='experiment dataset')
# parser.add_argument('--net', default='unet2', help='experiment architecture')
args = parser.parse_args()

config = instSeg.ConfigParallel(image_channel=3)
config.backbone = 'uNet'
config.modules = ['embedding', 'edt']
config.loss['embedding'] = 'cos' 
config.loss['edt'] = 'mse' 
config.edt_in_ram = True
config.embedding_include_bg = True

config.input_normalization = 'per-image'
config.train_learning_rate = 1e-4
config.lr_decay_rate = 0.9
config.lr_decay_period = 10000
config.net_normalization = None
config.dropout_rate = 0

config.flip = True
config.random_rotation = True
config.random_crop = True
config.random_crop_range = (0.6, 0.8)
config.blur = True
config.blur_gamma = 2

model_dir = './model_Endometrium'
val_split = 0.1
epoches = 200

ds_dir = '/work/scratch/chen/Datasets/Endometrium/patches_gland'

# load dataset
files = []
with open(os.path.join(ds_dir, 'files.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        if int(row[0]) != int(args.val_part):
            files.append(row)
            print(row)

X_train, y_train = [], []
for row in files:
    img = cv2.imread(os.path.join(ds_dir, row[2]), cv2.IMREAD_UNCHANGED)
    X_train.append(img)
    gt = cv2.imread(os.path.join(ds_dir, row[3]), cv2.IMREAD_UNCHANGED)
    y_train.append(gt)

idx = list(range(len(X_train)))
shuffle(idx)
X_train = [X_train[i] for i in idx]
y_train = [y_train[i] for i in idx]

val_split_ = int(val_split*len(idx))

X_val = np.array(X_train[-val_split_:])
y_val = np.expand_dims(np.array(y_train[-val_split_:]), axis=-1)
ds_val = {'image': X_val, 'instance': y_val}
del X_val, y_val

X_train = np.array(X_train[:-val_split_])
y_train = np.expand_dims(np.array(y_train[:-val_split_]), axis=-1)
ds_train = {'image': X_train, 'instance': y_train}
del X_train, y_train
print(ds_train['image'].shape, ds_train['instance'].shape, ds_val['image'].shape, ds_val['instance'].shape)

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir+'_'+str(args.val_part))
model.train(ds_train, ds_val, batch_size=4, epochs=epoches, augmentation=True)

# from skimage.segmentation import mark_boundaries
# model_dir = './model_endometrium/model_Endometrium'+'_'+str(args.val_part)
# print(model_dir)
# model = instSeg.load_model(model_dir, load_best=True)

# test_dir = './testttttt'
# if not os.path.exists(test_dir):
#     os.makedirs(test_dir)

# for idx, img in enumerate(ds_val['image']):
#     print(idx, img.shape)
#     instance = model.predict(img, keep_size=True)[0]
#     vis = mark_boundaries(img, instance, color=(0, 1, 1))
#     cv2.imwrite(os.path.join(test_dir,str(idx)+'.tif'), (vis*255).astype(np.uint8))

