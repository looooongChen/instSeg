
import instSeg
import glob
from skimage.io import imread
import numpy as np
import os

config = instSeg.ConfigContour()
run_name = 'cystJKI_contour'
ds_dir = './ds_cystJKI'
base_dir = './'

X = sorted(glob.glob(os.path.join(ds_dir, 'image/*.tif')))
Y = sorted(glob.glob(os.path.join(ds_dir, 'ground_truth/*.png')))
train_data = {'image': np.array(list(map(imread,X))),
              'object': np.expand_dims(np.array(list(map(imread,Y))), axis=-1)}

model = instSeg.InstSegContour(config=config, base_dir=base_dir, run_name=run_name)

model.train(train_data, batch_size=4, epochs=300, augmentation=False)

model.load_weights(load_best=False)
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

