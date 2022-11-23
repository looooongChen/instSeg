# Under construction, kinda messy now...

# instSeg
a collection implementatsions of deep learning approaches for instance segmentation of biomedical images

instSegDCAN: DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation


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


## train the model

```python

import instSeg
from skimage.io import imread, imsave
import numpy as np

config = instSeg.ConfigParallel()

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
