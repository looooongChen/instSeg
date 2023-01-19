import numpy as np
from skimage.measure import regionprops, label
from instSeg.stitcher import *
from skimage.morphology import square
import copy
import os
import cv2


def seg_in_tessellation(model, img, patch_sz=[512,512], margin=[64,64], overlap=[64,64], mode='lst', min_obj_size=None):

    '''
    Args:
        model: model for prediction, easiest way to load model: instSeg.load_model(model_dir)
        img: input image
        patch_sz: size of patches to split
        margin: parts to ignore, since predictions at the image surrounding area tend to not be as good as the centeral part
            2*margin should less than image size
        overlap: size of overlapping area
            2*margin + overlap should also less than image size
        mode: 'lst' or 'wsi'
            'lst' (label stitch) model: instances are obtained in patch level, then stitched. An overlap is necessray to match intances from neighbouring patches
            'bi' (binary instance) model: we can stitch binary semantic segmentation and boundary map and then get instances
    '''

    meta = split(img, patch_sz, (overlap[0]+2*margin[0], overlap[1]+2*margin[1]), patch_in_ram=True)

    # from skimage.segmentation import mark_boundaries

    if mode == 'foreground':
        for idx, patch in enumerate(meta['patches']):
            p = model.predict_raw(patch['data'])
            # foreground map
            fg = np.squeeze(p['foreground']).copy()
            patch['data'] = fg[margin[0]:fg.shape[0]-margin[0], margin[1]:fg.shape[1]-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]
            
        fg = stitch(meta, channel=1, mode='average') > 0.5

        return fg
    
    if mode == 'lst':
        if overlap[0] == 0 or overlap[1]==0:
            print("WARNING: overlap is necessary for label map stitching!!!")
        for idx, patch in enumerate(meta['patches']):
            p = model.predict(patch['data'], keep_size=True)[0]
            
            # print(idx)
            # vis = mark_boundaries(patch['data'], p, color=(0, 1, 1))
            # cv2.imwrite(os.path.join('./test',str(idx)+'.tif'), (vis*255).astype(np.uint8))
            
            patch['data'] = p[margin[0]:p.shape[0]-margin[0], margin[1]:p.shape[1]-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]


        instances = stitch(meta, channel=1, mode='label')
    
    if mode == 'bi':

        assert 'foreground' in model.config.modules and 'contour' in model.config.modules
        # only for binary foreground segmentation
        meta_contour = copy.deepcopy(meta)
        for idx, patch in enumerate(meta['patches']):
            p = model.predict_raw(patch['data'])
            # foreground map
            fg = np.squeeze(p['foreground']).copy()
            patch['data'] = fg[margin[0]:fg.shape[0]-margin[0], margin[1]:fg.shape[1]-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]
            # contour map
            contour = np.squeeze(p['contour']).copy()
            meta_contour['patches'][idx]['data'] = contour[margin[0]:fg.shape[0]-margin[0], margin[1]:fg.shape[1]-margin[1]]
            meta_contour['patches'][idx]['position'] = patch['position']
            meta_contour['patches'][idx]['size'] = patch['size']
            
        fg = stitch(meta, channel=1, mode='average') > 0.5
        contour = stitch(meta_contour, channel=1, mode='max') > 0.5

        contour = cv2.dilate(contour.astype(np.uint8), square(3), iterations = 1)

        instances = fg * (contour == 0)
        instances = label(instances).astype(np.uint16)
        while True:
            pixel_add = cv2.dilate(instances, square(3), iterations = 1) * (instances == 0) * fg
            if np.sum(pixel_add) != 0:
                instances += pixel_add
            else:
                break

    if min_obj_size is not None:
        for obj in regionprops(instances):
            if obj.area < min_obj_size :
                instances[obj.coords[:,0], obj.coords[:,1]] = 0
    
    return instances










