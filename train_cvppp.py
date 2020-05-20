
import instSegV2
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from instSegV2.post_process import *

config = instSegV2.Config(semantic_module=True, dist_module=True, embedding_module=True)
config.random_rotation = True

dirs = ['A1', 'A2', 'A3', 'A4']

X, Y = [], []
for d in dirs:
    X = X + sorted(glob.glob('./ds_cvppp/training_images/'+d+'/*.png'))
    Y = Y + sorted(glob.glob('./ds_cvppp/training_truth/'+d+'/*.png'))
train_data = {'image': list(map(cv2.imread,X)),
              'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y))}

# print(len(['image']), len(train_data['object']))
# plt.subplot(1,2,1)
# plt.imshow(train_data['image'][0])
# plt.subplot(1,2,2)
# plt.imshow(train_data['object'][0]*20)
# plt.show()

model = instSegV2.InstSeg(config=config, base_dir='./', run_name='cvppp_aug')
model.train(train_data, batch_size=4, epochs=800, augmentation=False)

