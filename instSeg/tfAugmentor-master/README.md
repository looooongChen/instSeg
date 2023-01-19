
# tfAugmentor
An image augmentation library for tensorflow. The libray is designed to be easily used with tf.data.Dataset. The augmentor accepts tf.data.Dataset object or a nested tuple of numpy array. 

## Augmentations
| **Original** | **Flip** | **Rotation** | **Translation** |
|:---------|:---------|:---------| :-------- |
| ![original](/demo/image/plant_grid.png) | ![demo_flip](/demo/demo_flip.png) | ![demo_rotation](/demo/demo_rotation.png) | ![demo_translation](/demo/demo_translation.png) |
| **Crop** | **Elactic Deform** |  |  |
| ![demo_crop](/demo/demo_crop.png) | ![demo_elastic](/demo/demo_elastic.png) |  |  |
| **Gaussian Blur**  | **Contrast** | **Gamma** | 
| ![demo_blur](/demo/demo_blur.png) | ![demo_contrast](/demo/demo_contrast.png) | ![demo_gamma](/demo/demo_gamma.png) |  |

## Demo

| **Random Rotation** | **Random Translation** | **Random Crop** |
|:---------|:---------|:---------| 
| ![demo_ratation](/demo/gif/demo_random_rotation.gif) | ![demo_translation](/demo/gif/demo_random_translation.gif) | ![demo_crop](/demo/gif/demo_random_crop.gif) |
| **Random Contrast** | **Random Gamma** | **Elastic Deform** |
| ![demo_contrast](/demo/gif/demo_random_contrast.gif) | ![demo_gamma](/demo/gif/demo_random_gamma.gif) | ![demo_elastic](/demo/gif/demo_random_elastic.gif) |




## Installation
tfAugmentor is written in Python and can be easily installed via:
```python
pip install tfAugmentor
```
Required packages:
- tensorflow (developed under tf 2.4), should work with any 2.x version
- numpy (numpy=1.20 may cause error of tf.meshgrid, use another version)

## Quick Start
tfAugmentor is implemented to work seamlessly with tf.data. The tf.data.Dataset object can be directly processed by tfAugmentor.

To instantiate an `Augmentor` object, three arguments are required:

```python
class Augmentor(object):
    def __init__(self, signature, image=[], label_map=[]):
		...
```

- signature: to present the structure of the dataset 
    - a nested tuple of string
    - a list/tuple of dictionary keys, if your dataset is in a dictionary form
- image: list of string items in signature, which will be treated as normal images
- label: list of string items in signature, which will be treated as segmentation masks

**Note:** only the items in 'image' and 'label' will be processed, others will remain untouched

### A simple example
```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(signature=('image', ('mask1', 'mask2')), 
                      image=['image'], 
                      label=['mask1', 'mask2'])

# add augumentation operations
aug.flip_left_right(probability=0.5)
aug.rotate90(probability=0.5)
aug.elastic_deform(strength=2, scale=20, probability=1)

# assume we have three numpy arrays
X_image = ... # shape [batch, height, width, channel]
Y_mask1 = ... # shape [batch, height, width, 1]
Y_mask2 = ... # shape [batch, height, width, 1]

# create tf.data.Dataset object
tf_dataset = tf.data.Dataset.from_tensor_slices((X_image, (Y_mask1, Y_mask2))))
# do the actual augmentation
ds1 = aug(tf_dataset)

# or you can directly pass the numpy arrays, a tf.data.Dataset object will be returned 
ds2 = aug((X_image, (Y_mask1, Y_mask2))), keep_size=True)
```

If the data is passed as a python dictionary, the signature should be the list/tuple of keys. For example:

```python
import tfAugmentor as tfaug

# new tfAugmentor object
aug = tfaug.Augmentor(signature=('image', 'mask1', 'mask2'), 
                      image=['image'], 
                      label=['mask1', 'mask2'])

# add augumentation operations
aug.flip_left_right(probability=0.5)
aug.rotate90(probability=0.5)
aug.elastic_deform(strength=2, scale=20, probability=1)

# assume we have three numpy arrays
X_image = ... # shape [batch, height, width, channel]
Y_mask1 = ... # shape [batch, height, width, 1]
Y_mask2 = ... # shape [batch, height, width, 1]

ds_dict = {'image': X_image,
           'mask1': Y_mask1,
           'mask2': Y_mask2}
# create tf.data.Dataset object
tf_dataset = tf.data.Dataset.from_tensor_slices(ds_dict)
# do the actual augmentation
ds1 = aug(tf_dataset)

# or directly pass the data
ds2 = aug(ds_dict)
```

**Note:** All added operations will be executed one by one, but you can create multiply tfAugmentor to realize parallel pipelines

### A more complicated example

```python
import tfAugmentor as tfaug

# since 'class' is neither in 'image' nor in 'label', it will not be processed 
aug1 = tfaug.Augmentor((('image_rgb', 'image_depth'), ('semantic_mask', 'class')), 
                       image=['image_rgb', 'image_depth'], 
                       label=['semantic_mask'])
aug2 = tfaug.Augmentor((('image_rgb', 'image_depth'), ('semantic_mask', 'class')), 
                       image=['image_rgb', 'image_depth'], 
                       label=['semantic_mask'])

# add different augumentation operations to aug1 and aug2 
aug1.flip_left_right(probability=0.5)
aug1.random_crop_resize(sacle_range=(0.7, 0.9), probability=0.5)
aug2.elastic_deform(strength=2, scale=20, probability=1)

# assume we have the 1000 data samples
X_rgb = ...  # shape [1000 x 512 x 512 x 3]
X_depth = ... # shape [1000 x 512 x 512 x 1]
Y_semantic_mask = ... # shape [1000 x 512 x 512 x 1]
Y_class = ... # shape [1000 x 1]

# create tf.data.Dataset object
ds_origin = tf.data.Dataset.from_tensor_slices(((X_rgb, X_depth), (Y_semantic_mask, Y_class))))
# do the actual augmentation
ds1 = aug1(ds_origin)
ds2 = aug2(ds_origin)
# combine them
ds = ds_origin.concatenate(ds1)
ds = ds.concatenate(ds1)

```

## Main Features

The argument **'probability'** controls the possibility of a certain augmentation taking place. 

### Mirroring
```python
# flip the image left right  
aug.flip_left_right(probability=1)
# flip the image up down 
aug.flip_up_down(probability=1) 
```
### Rotation
```python
# rotate by 90 degree clockwise
a.rotate90(probability=1) 
# rotate by 180 degree clockwise
a.rotate180(probability=1)
# rotate by 270 degree clockwise 
a.rotate270(probability=1) 
# rotate by a certrain degree, Args: angle - scala, in degree
a.rotate(angle, probability=1) 
# randomly rotate the image
a.random_rotate(probability=1) 
```

### Translation
```python
# tranlate image, Args: offset - [x, y]
a.translate(offset, probability=1):
# randoms translate image 
a.random_translate(translation_range=[-100, 100], probability=1):
```

### Crop and Resize
```python
# randomly crop a sub-image and resize to the original image size
a.random_crop(scale_range=([0.5, 0.8], preserve_aspect_ratio=False, probability=1) 
```

### Elastic Deformation
```python
# performa elastic deformation
a.elastic_deform(scale=10, strength=200, probability=1)
```

### Photometric Adjustment
```python
# adjust image contrast randomly
a.random_contrast(contrast_range=[0.6, 1.4], probability=1)
# perform gamma correction with random gamma values
a.random_gamma(gamma_range=[0.5, 1.5], probability=1)
```

### Noise
```python
# blur the image with gaussian kernel
a.gaussian_blur(sigma=2, probability=1)
```


## Caution
- If .batch() of tf.data.Dataset is used before augmentation, please set **drop_remainder=True**. Oherwise, the batch_size will be set to None. The augmention of tfAgmentor requires the batch_size dimension    