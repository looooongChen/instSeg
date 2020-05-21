
import instSegV2
import glob
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from instSegV2.post_process import *

config = instSegV2.Config(semantic_module=True, dist_module=True, embedding_module=True)
config.batch_normalization = False

X = sorted(glob.glob('./ds_cysts/image/*.tif'))
Y = sorted(glob.glob('./ds_cysts/ground_truth/*.png'))
train_data = {'image': np.array(list(map(imread,X))),
              'object': np.expand_dims(np.array(list(map(imread,Y))), axis=-1)}

# print(train_data['image'].shape, train_data['object'].shape)
# print(X.shape, Y.shape)
# plt.subplot(1,2,1)
# plt.imshow(X[0])
# plt.subplot(1,2,2)
# plt.imshow(Y[0])
# plt.show()

model = instSegV2.InstSeg(config=config, base_dir='./', run_name='cyst_aug')
model.train(train_data, batch_size=4, epochs=500, augmentation=False)

# import time
# start = time.time()
# in_list, out_list = model.ds_from_np(train_data)
# print(time.time()-start)
# plt.subplot(1,4,1)
# plt.imshow((in_list[0][0]*25+128).astype(np.uint8))
# print(in_list[1][0,0:20,0:20].astype(np.uint8))
# plt.subplot(1,4,2)
# plt.imshow(out_list[0][0,:,:,0])
# plt.subplot(1,4,3)
# plt.imshow(out_list[1][0,:,:,0])
# plt.subplot(1,4,4)
# plt.imshow(out_list[2][0,:,:,0])
# plt.show()
# print(out_list[0].dtype, out_list[1].dtype, out_list[2].dtype)


# X = sorted(glob.glob('./ds_cysts/image/*.tif'))[0:3]
# imgs = np.array(list(map(imread,X)))

# print(imgs.shape)
# pred = model.predict(imgs)
# # l = maskViaSeed(pred)

# # plt.subplot(1,3,1)
# # plt.imshow(imgs[1])
# # plt.subplot(1,3,2)
# # plt.imshow(l)
# # plt.subplot(1,3,3)
# # plt.imshow(np.squeeze(np.argmax(pred['semantic'][1], axis=-1)))

# m = np.squeeze(np.argmax(pred['semantic'][0], axis=-1))
# plt.subplot(1,2,1)
# plt.imshow(m)
# plt.subplot(1,2,2)
# m = np.expand_dims(m, axis=-1)
# plt.imshow(pred['embedding'][0,:,:,3:6]*m)
# plt.show()
# np.save('./emb.npy', pred['embedding']*m)

# from visualization import * 
# import cv2
# # l = cv2.resize(l, (imgs[1].shape[1], imgs[1].shape[0]), interpolation=cv2.INTER_NEAREST)
# # print(imgs[1].shape, l.shape)
# ll = mask2masks(l)

# img = l = cv2.resize(imgs[1], (512, 512), interpolation=cv2.INTER_NEAREST)

# vis = visulize_masks(img, ll, random_color=True, fill=True)
# plt.imshow(vis)
# plt.show()
