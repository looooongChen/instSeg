
import instSeg
import glob
from skimage.io import imread
import numpy as np
import os

config = instSeg.Config(semantic=True, dist=True, embedding=True)
config.module_order = ['dist', 'semantic', 'embedding']
config.feature_forward_dimension = 32
run_name = 'cystJKI'
ds_dir = './ds_cystJKI'
base_dir = './'

X = sorted(glob.glob(os.path.join(ds_dir, 'image/*.tif')))
Y = sorted(glob.glob(os.path.join(ds_dir, 'ground_truth/*.png')))
train_data = {'image': np.array(list(map(imread,X))),
              'object': np.expand_dims(np.array(list(map(imread,Y))), axis=-1)}

model = instSeg.InstSeg_Mul(config=config, base_dir=base_dir, run_name=run_name)
model.train(train_data, batch_size=4, epochs=150, augmentation=False)
