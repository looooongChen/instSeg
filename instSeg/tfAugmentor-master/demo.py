from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import tfAugmentor as tfaug

# img = imread('./demo/image/cell.png')
# mask = np.expand_dims(imread('./demo/image/mask_cell.png'), axis=-1)
img = imread('./demo/image/plant_grid.png')
mask = np.expand_dims(imread('./demo/image/mask_plant.png'), axis=-1)
imgs = np.repeat(np.expand_dims(img, axis=0), 10, axis=0) 
masks1 = np.repeat(np.expand_dims(mask, axis=0), 10, axis=0)
masks2 = masks1.copy() 

#### nested tuple input ####
aug = tfaug.Augmentor(('image', ('mask1', 'mask2')), image=['image'], label=['mask1'])
ds = tf.data.Dataset.from_tensor_slices((imgs, (masks1, masks2)))

# aug.flip_left_right(probability=0.5)
# aug.flip_up_down(probability=0.5)
# aug.rotate90(probability=0.5)
# aug.rotate180(probability=0.5)
# aug.rotate270(probability=0.5)
# aug.rotate(60, probability=1)
# aug.random_rotate(probability=1)
# aug.translate((100,200), probability=1)
# aug.random_translate((-200,200), probability=1)
# aug.gaussian_blur(sigma=3)
# aug.random_crop([0.5, 0.8], probability=1, preserve_aspect_ratio=False)
# aug.elastic_deform(scale=10, strength=200, probability=1)
aug.random_contrast(contrast_range=[0.3, 1.5], probability=1)
# aug.random_gamma(gamma_range=[0.5, 2], probability=1)

ds = ds.batch(5, drop_remainder=True)

ds_aug = aug(ds).unbatch()

# ds = aug(ds).batch(2, drop_remainder=True)
# ds_aug = ds_aug.concatenate(ds)

# i = 0
# for img, mask in ds_aug:
#     print(i)
#     imsave('./demo/img_'+str(i)+'.png', np.squeeze(img))
#     imsave('./demo/mask_'+str(i)+'.png', 20*np.squeeze(mask[0]))
#     imsave('./demo/mask2_'+str(i)+'.png', 20*np.squeeze(mask[1]))
#     i += 1

import imageio
import cv2
writer = imageio.get_writer("./demo.gif", mode='I', fps=1)
for img, _ in ds_aug:
    img = cv2.resize(np.array(np.squeeze(img)).astype(np.uint8), (256, 256))
    writer.append_data(img)
writer.close()


#### dictionary input ####

# aug = tfaug.Augmentor(('image', 'mask1', 'mask2'), image=['image'], label=['mask1'])
# ds = tf.data.Dataset.from_tensor_slices({'image': imgs, 'mask1': masks1, 'mask2': masks2})

# ds_aug = aug(ds)

# for i, item in enumerate(ds_aug):
#     print(i)
#     imsave('./demo/img_'+str(i)+'.png', np.squeeze(item['image'][0]))
#     imsave('./demo/mask_'+str(i)+'.png',20* np.squeeze(item['mask1'][0]))
#     imsave('./demo/mask2_'+str(i)+'.png', 20*np.squeeze(item['mask2'][0]))