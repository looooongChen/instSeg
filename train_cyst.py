
import instSegV2
import glob
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

config = instSegV2.Config(semantic_module=True, dist_module=True, embedding_module=True)
config.verbose = False

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

model = instSegV2.InstSeg(config=config, base_dir='./', run_name='complete')
model.train(train_data, batch_size=4, epochs=300, augmentation=True)

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


imgs = train_data['image'][0:5]
pred = model.predict(imgs)

plt.subplot(1,3,1)
plt.imshow(imgs[0])
plt.subplot(1,3,2)
plt.imshow(np.argmax(pred['semantic'][0], axis=-1))
plt.subplot(1,3,3)
plt.imshow(np.squeeze(pred['dist'][0]))
# m = np.expand_dims(np.argmax(pred['semantic'][0], axis=-1), axis=-1)
# plt.subplot(1,2,1)
# plt.imshow(m)
# plt.subplot(1,2,2)
# plt.imshow(pred['embedding'][0,:,:,3:6]*m)
plt.show()