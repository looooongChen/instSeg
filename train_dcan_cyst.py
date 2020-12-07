
import instSeg
from skimage.io import imread
import numpy as np
import os
import csv

config = instSeg.ConfigContour()
run_name = 'model_DCAN_cyst'
base_dir = './'
splits = [0]

for s in splits:
    # load dataset
    with open('D:/instSeg/ds_cyst/cyst.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        X_train, y_train = [], []
        X_val, y_val = [], []
        for row in csv_reader:
            if int(row[0]) == s:
                X_val.append(row[1])
                y_val.append(row[2])
            else:
                X_train.append(row[1])
                y_train.append(row[2])

    train_ds = {'image': np.array(list(map(imread,X_train))),
                'instance': np.expand_dims(np.array(list(map(imread,y_train))), axis=-1)}
    val_ds = {'image': np.array(list(map(imread,X_val))),
              'instance': np.expand_dims(np.array(list(map(imread,y_val))), axis=-1)}
    # create model and train

    model = instSeg.InstSegDCAN(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
    model.train(train_ds, val_ds, batch_size=4, epochs=300, augmentation=False)

# model.load_weights(load_best=False)
# img = imread('./test/Sf_RP1_2020_3a.tif')
# inst, semantic = model.predict(img)
# import matplotlib.pyplot as plt
# from skimage.color import label2rgb
# plt.subplot(1,3,1)
# plt.imshow(img)
# plt.subplot(1,3,2)
# plt.imshow(label2rgb(inst, bg_label=0))
# plt.subplot(1,3,3)
# plt.imshow(semantic)
# plt.show()

