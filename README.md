# instSeg - Instance segmentation via pixel embedding learning

This repository contains the implementation of pixel embedding learning model for instance segmentation, as described in the papers:

# instSeg
a collection python implementatsions of deep learning approaches for instance segmentation of biomedical images

- Long Chen, Martin Strauch, and Dorit Merhof.  
[*Instance Segmentation of Biomedical Images with an Object-aware Embedding Learned with Local Constraints*](https://arxiv.org/abs/2004.09821).  
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Shenzhen, China, 2019.

- Long Chen, Yuli Wu, and Dorit Merhof.  
[*Instance Segmentation of Dense and Overlapping Objects via Layering*](https://arxiv.org/abs/2210.03551).  
The British Machine Vision Conference (BMVC), London, UK, 2022


## Overview:


# Prerequisites 

## Python dependencies:

- tensorflow 2.x
- scikit-image
- opencv
- [tfAugmentor](https://github.com/looooongChen/tfAugmentor) (required only if image augmentation function is activated)
- tqdm
- numpy

# Usage


## a very brief example

```python

import instSeg
from skimage.io import imread, imsave
import numpy as np

config = instSeg.Config(image_channel=1)

# a list containing images of size H x W x C
imgs_train, imgs_val = [...], [...] 
# a list contraining instance mask of size H x W x 
masks_train, masks_val = [...], [...] 

ds_train = {'image': imgs_train, 'instance': masks_train}
ds_val = {'image': imgs_val, 'instance': masks_val}


X_train, y_train = [], [] # list of img/gt path for training
X_val, y_val = [], [] # list of img/gt path for test

val_ds = {'image': np.array(list(map(imread,X_val))),
            'instance': np.expand_dims(np.array(list(map(imread,y_val))), axis=-1)}
train_ds = {'image': np.array(list(map(imread,X_train))),
            'instance': np.expand_dims(np.array(list(map(imread,y_train))), axis=-1)}

# create model and train
model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
model.train(train_ds, val_ds, batch_size=4, epochs=300)
```

## train the model

## prediction


## evaluation


# Results 

## dataset 1

## dataset 1

## dataset 1




## How to cite 
```

@inproceedings{LongMACCAIInstance,  
  author = {Long Chen, Martin Strauch, Dorit Merhof},  
  title = {Instance Segmentation of Biomedical Images with an Object-Aware Embedding Learned with Local Constraints},  
  booktitle = {MICCAI 2019},  
  year = {2019},  
}

@inproceedings{LongMACCAIInstance,  
  author = {Long Chen, Martin Strauch, Dorit Merhof},  
  title = {Instance Segmentation of Biomedical Images with an Object-Aware Embedding Learned with Local Constraints},  
  booktitle = {MICCAI 2019},  
  year = {2019},  
}


```
