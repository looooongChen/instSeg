import numpy as np
from skimage.measure import regionprops, label
from instSeg.stitcher import *
import copy
import os
import cv2

# def seg_view(model, img, patch_sz=[512,512], min_sz=900):
#     overlap = [patch_sz[0]//2, patch_sz[1]//2]
#     margin = [overlap[0]//2, overlap[1]//2]

#     pad = ((margin[0],margin[0]), (margin[1],margin[1])) if len(img.shape) == 2 else ((margin[0],margin[0]), (margin[1],margin[1]), (0,0))
#     img = np.pad(img, pad, mode='reflect')

#     meta, patches = split2D(img, patch_sz, overlap, patch_in_ram=True)
#     print(len(meta['patches']), len(patches))
#     if 'semantic' in model.config.modules and 'contour' in model.config.modules:
#         patches_semantic, patches_contour = {}, {}
#         for k in patches.keys():
#             print('processing patch: ', k)
#             p = model.predict_raw(patches[k], keep_size=True)
#             patches_semantic[k] = p['semantic'][:,:,1].copy()
#             patches_contour[k] = p['contour']
#             patches_contour[k] = cv2.resize(patches_contour[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k] = cv2.resize(patches_semantic[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k][:margin[0],:] = 0
#             patches_semantic[k][-margin[0]:,:] = 0
#             patches_semantic[k][:,:margin[1]] = 0
#             patches_semantic[k][:,-margin[1]:] = 0
#         semantic = stitch2D(meta, channel=1, patches=patches_semantic, mode='max')
#         contour = stitch2D(meta, channel=1, patches=patches_contour, mode='max')
#         # print(np.sum(semantic), np.max(semantic))
#         instance = (semantic > 0.5) *  (contour < 0.5)
#         instance = label(instance)
#         for p in regionprops(instance):
#             if p.area < min_sz:
#                 instance[p.coords[:,0], p.coords[:,1]] = 0
#         instance = dilation(instance, disk(2))
#         instance = instance[margin[0]:-margin[0], margin[1]:-margin[1]]
#         return instance


# def seg_view_tessellation(model, img, patch_sz=[512,512]):
#     overlap = [patch_sz[0]//2, patch_sz[1]//2]
#     margin = [overlap[0]//2, overlap[1]//2]

#     pad = ((margin[0],margin[0]), (margin[1],margin[1])) if len(img.shape) == 2 else ((margin[0],margin[0]), (margin[1],margin[1]), (0,0))
#     img = np.pad(img, pad, mode='reflect')

#     meta = split2D(img, patch_sz, overlap, patch_in_ram=True)
#     if 'semantic' in model.config.modules and 'contour' in model.config.modules:
#         patches_semantic, patches_contour = {}, {}
#         for k in patches.keys():
#             print('processing patch: ', k)
#             p = model.predict_raw(patches[k], keep_size=True)
#             patches_semantic[k] = p['semantic'][:,:,1].copy()
#             patches_contour[k] = p['contour'].copy()
#             patches_contour[k] = cv2.resize(patches_contour[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k] = cv2.resize(patches_semantic[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k][:margin[0],:] = 0
#             patches_semantic[k][-margin[0]:,:] = 0
#             patches_semantic[k][:,:margin[1]] = 0
#             patches_semantic[k][:,-margin[1]:] = 0
#         semantic = stitch2D(meta, channel=1, patches=patches_semantic, mode='max')
#         contour = stitch2D(meta, channel=1, patches=patches_contour, mode='max')
#         instance = (semantic > 0.5) *  (contour < 0.5)
#         instance = instance[margin[0]:-margin[0], margin[1]:-margin[1]]
#         instance = label(instance)
#         return instance


# def seg_in_tessellation(model, img, patch_sz=[512,512]):
#     overlap = [patch_sz[0]//2, patch_sz[1]//2]
#     margin = [overlap[0]//2, overlap[1]//2]

#     pad = ((margin[0],margin[0]), (margin[1],margin[1])) if len(img.shape) == 2 else ((margin[0],margin[0]), (margin[1],margin[1]), (0,0))
#     img = np.pad(img, pad, mode='reflect')

#     meta = split2D(img, patch_sz, overlap, patch_in_ram=True)
#     if 'semantic' in model.config.modules and 'contour' in model.config.modules:
#         patches_semantic, patches_contour = {}, {}
#         for k in patches.keys():
#             print('processing patch: ', k)
#             p = model.predict_raw(patches[k], keep_size=True)
#             patches_semantic[k] = p['semantic'][:,:,1].copy()
#             patches_contour[k] = p['contour'].copy()
#             patches_contour[k] = cv2.resize(patches_contour[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k] = cv2.resize(patches_semantic[k], (patch_sz[1], patch_sz[0]))
#             patches_semantic[k][:margin[0],:] = 0
#             patches_semantic[k][-margin[0]:,:] = 0
#             patches_semantic[k][:,:margin[1]] = 0
#             patches_semantic[k][:,-margin[1]:] = 0
#         semantic = stitch2D(meta, channel=1, patches=patches_semantic, mode='max')
#         contour = stitch2D(meta, channel=1, patches=patches_contour, mode='max')
#         instance = (semantic > 0.5) *  (contour < 0.5)
#         instance = instance[margin[0]:-margin[0], margin[1]:-margin[1]]
#         instance = label(instance)
#         return instance

def seg_in_tessellation(model, img, patch_sz=[512,512], margin=[64,64], overlap=[128,128], mode='lst'):

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
            'lst' (label stitch) model: instances are obtained in patch level, then stiched. An overlap is necessray to match intances from neighbouring patches
            'wsi' (while slide image) model: for some approaches, we can stiched necessary raw outputs first and then get instances, for example, the forground and contour map in DCAN (deep contour-aware network)
    '''

    pad = ((margin[0],margin[0]), (margin[1],margin[1])) if len(img.shape) == 2 else ((margin[0],margin[0]), (margin[1],margin[1]), (0,0))
    img = np.pad(img, pad, mode='reflect')

    meta = split2D(img, patch_sz, (overlap[0]+2*margin[0], overlap[1]+2*margin[1]), patch_in_ram=True, save_dir=None)

    if mode == 'lst':
        if overlap[0] == 0 or overlap[1]==0:
            print("WARNING: overlap is necessary for label map stitching!!!")
        for k, patch in meta['patches'].items():
            print('processing: ', k)
            p = model.predict(patch['data'], keep_size=True)[0]
            patch['data'] = p[margin[0]:-margin[0], margin[1]:-margin[1]]
            patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
            patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]
        instance = stitch2D(meta, channel=1, mode='label')
        instance = instance[margin[0]:-margin[0], margin[1]:-margin[1]]
        return instance
    
    if mode == 'wsi':

        if 'semantic' in model.config.modules and 'contour' in model.config.modules:
            meta_contour = copy.deepcopy(meta)
            for k, patch in meta['patches'].items():
                print('processing patch: ', k)
                p = model.predict_raw(patch['data'], keep_size=True)
                semantic = np.argmax(p['semantic'], axis=-1).copy()
                patch['data'] = semantic[margin[0]:-margin[0], margin[1]:-margin[1]]
                patch['position'] = [patch['position'][0]+margin[0], patch['position'][1]+margin[1]]
                patch['size'] = [patch['size'][0]-2*margin[0], patch['size'][1]-2*margin[1]]

                contour = p['contour'].copy()
                meta_contour['patches'][k]['data'] = contour
                
            semantic = stitch2D(meta, channel=1, mode='average')
            contour = stitch2D(meta_contour, channel=1, mode='max')
            instance = (semantic > 0.5) *  (contour < 0.5)
            instance = instance[margin[0]:-margin[0], margin[1]:-margin[1]]
            instance = label(instance)
            return instance








