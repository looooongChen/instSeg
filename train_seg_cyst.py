
import instSeg
from skimage.io import imread, imsave
import numpy as np
import os
import csv
import time
import cv2

config = instSeg.ConfigCascade()
config.save_best_metric = 'loss'
config.modules = ['semantic']
config.loss_semantic = 'dice'
# config.loss_contour = 'dice'
config.backbone = 'uNet'
config.filters = 32
run_name = 'model_SEGdice_cyst'
base_dir = './'
splits = [0,1,2,3,4]

ds_csv = 'D:/instSeg/ds_cyst/cyst.csv'

# training 
# for s in splits:
#     # load dataset
#     with open(ds_csv) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=';')
#         X_train, y_train = [], []
#         X_val, y_val = [], []
#         for row in csv_reader:
#             if int(row[0]) == s:
#                 X_val.append(row[1])
#                 y_val.append(row[2])
#             else:
#                 X_train.append(row[1])
#                 y_train.append(row[2])
    
#     train_ds = {'image': np.array(list(map(imread,X_train))),
#                 'instance': np.expand_dims(np.array(list(map(imread,y_train))), axis=-1)}
#     val_ds = {'image': np.array(list(map(imread,X_val))),
#               'instance': np.expand_dims(np.array(list(map(imread,y_val))), axis=-1)}
#     # create model and train

#     model = instSeg.InstSegCascade(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
#     model.train(train_ds, val_ds, batch_size=4, epochs=300, augmentation=False)


# evalutation

eval_dir = 'eval'+run_name[5:]
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)


with open(ds_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    idx_spilt, img_path, gt_path = [], [], []
    for row in csv_reader:
        idx_spilt.append(int(row[0]))
        img_path.append(row[1])
        gt_path.append(row[2])

dice = []
for s in splits:
    
    model = instSeg.InstSegCascade(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
    model.load_weights(load_best=True)

    for i, ss in enumerate(idx_spilt):
        if ss != s:
            continue
        print("processing: ", img_path[i])
        # load image
        img = imread(img_path[i])
        gt = imread(gt_path[i])
        # process
        seg, raw = model.predict(img)
        # eval
        d = 2* np.sum((seg>0) * (gt>0))/(np.sum(seg>0)+np.sum(gt>0))
        dice.append(d)
        # vis instance
        vis = instSeg.vis.vis_semantic_area(img, seg, colors=['red'])
        imsave(os.path.join(eval_dir, 'example_{:04d}_seg.png'.format(i)), vis)

