import instSeg
import os
import numpy as np
from skimage.morphology import dilation, disk
from skimage.measure import regionprops
from skimage.io import imread, imsave
from skimage.color import label2rgb

model_dir = './model_DCAN_cyst512_resnet101'
image_path = '...'
save_path = '...'
min_size = 200


model = instSeg.load_model(model_dir=model_dir + '_' + str(s), load_best=True)

img = imread(image_path)[:,:,:3]
instance = instSeg.seg_in_tessellation(model, img, patch_sz=[512,512], margin=[128,128], overlap=[0,0], mode='wsi')
for p in regionprops(instance):
    if p.area < min_size:
        instance[p.coords[:,0], p.coords[:,1]] = 0
        
instance = dilation(instance, disk(2))
instance = label2rgb(instance, bg_label=0)*255
vis = 0.6*img + 0.4*instance
imsave(os.path.join(save_dir, item[2]), vis.astype(np.uint8))
