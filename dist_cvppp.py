
import instSegV2
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from instSegV2.evaluation import eva_angle
import os
import csv

config = instSegV2.Config_Cascade(semantic=True, dist=False, embedding=True, module_order=['semantic', 'dist', 'embedding'])
run_name = 'cvppp_sft_embedding'
ds_dir = './ds_cvppp'
base_dir = './cvppp_crossval'

inner_angels = []
inter_angles = []
#### cross validation ####
with open(os.path.join(base_dir, run_name+'_crossval.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for fold in range(5):
        model = instSegV2.InstSeg_Cascade(config=config, base_dir=base_dir, run_name=run_name + '_crossval_'+str(fold))
        model.load_weights(load_best=True)
        X_test, Y_test = [], []
        with open(os.path.join(ds_dir, 'crossval_partition.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if int(row[0]) == fold:
                    X_test.append(os.path.join(ds_dir, row[1][2:]))
                    Y_test.append(os.path.join(ds_dir, row[2][2:]))
        # val_data = {'image': list(map(cv2.imread,X_val)),
        #             'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_val))}
        for i, f_img in enumerate(X_test):
            img = cv2.imread(f_img)
            gt = cv2.imread(Y_test[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt.astype(np.uint16), (config.image_size[1], config.image_size[0]), interpolation=cv2.INTER_NEAREST)
            pred_raw = model.predict_raw(img)
            inner_angel, inter_angle = eva_angle(pred_raw['embedding'], gt, config.neighbor_distance)
            print(inner_angel, inter_angle)
            writer.writerow([f_img, inner_angel, inter_angle])
            inner_angels.append(inner_angel)
            inter_angles.append(inter_angle)

    writer.writerow(['average', np.mean(inner_angel), np.mean(inter_angle)])



