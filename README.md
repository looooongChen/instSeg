# InstNetV2
InstNetV2: an CNN framework for instance segmtnation 


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
import instSegV2

# three modules can be activated independtly
config = instSegV2.Config(semantic_module=True, dist_module=True, embedding_module=True)
# activated modules will be cacaded in the following order, refer to model.Config for all configurations
config.module_order = ['semantic', 'dist', 'embedding']
# create the model
model = instSegV2.InstSeg(config=config, base_dir='./', run_name='complete')
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
