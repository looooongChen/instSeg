import numpy as np
import cv2
from skimage.morphology import disk, square, dilation, closing, opening
from skimage.measure import regionprops, label
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from instSeg.flow import *


def instance_from_emb_and_dist(raw, config):
    '''
    raw: a dict containing predictions of at least 'embedding', 'edt'
    Parameters should be set in config:
        emb_thres: threshold distinguishing object in the embedding space
        dist_thres: threshold to get seeds from distance regression map
        dist_intensity: if only instance is not evident, ignore
        emb_max_step: max step for expanding the instance region
    '''
    embedding, dist = np.squeeze(raw['embedding']), np.squeeze(raw['edt'])
    # embedding = smooth_emb(embedding, 3)
    emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    dist = gaussian(dist, sigma=1)
    regions = label(dist > config.dist_thres)
    props = regionprops(regions)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]
        emb_mean = np.mean(embedding[row, col], axis=0)
        emb_mean = emb_mean/np.linalg.norm(emb_mean)
        mean[p.label] = emb_mean

    step = 0
    while True:
        dilated = dilation(regions, square(3))
        front_r, front_c = np.nonzero((regions != dilated) * (regions == 0))

        similarity = [np.dot(embedding[r, c, :], mean[dilated[r, c]])
                      for r, c in zip(front_r, front_c)]
        add_ind = np.array([s > config.emb_thres for s in similarity])

        if np.all(add_ind == False):
            break
        regions[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]
        step += 1
        if step > config.emb_max_step:
            break

    for p in regionprops(regions, intensity_image=dist):
        if p.mean_intensity < config.dist_intensity:
            regions[p.coords[:,0], p.coords[:,1]] = 0

    return regions

def instance_from_semantic_and_contour(raw, config):

    semantic = np.squeeze(np.argmax(raw['semantic'], axis=-1)).astype(np.uint16)
    contour = cv2.dilate((np.squeeze(raw['contour']) > config.dcan_thres_contour).astype(np.uint8), square(3), iterations = 1)

    instances = label(semantic * (contour == 0)).astype(np.uint16)
    fg = (semantic > 0).astype(np.uint16)
    while True:
        pixel_add = cv2.dilate(instances, square(3), iterations = 1) * (instances == 0) * fg
        if np.sum(pixel_add) != 0:
            instances += pixel_add
        else:
            break
    
    return instances

def instance_from_edt(raw, config):
    '''
    Parameters should be set in config:
        dist_mode: 'thresholding', 'tracking'
        edt_instance_thres: thres to get instance seeds, if dist_mode == 'thresholding'
        edt_fg_thres: thres to get forground, if dist_mode == 'thresholding'
    '''
    dist = np.squeeze(raw['edt'])
    dist = gaussian(dist, sigma=1)
    fg = dist > config.edt_fg_thres
    
    if config.edt_mode == 'thresholding':
        instances = label(dist > config.edt_instance_thres)
        instances = watershed(-dist, instances) * fg
    elif config.edt_mode == 'tracking':
        flow = get_flow(dist, sigma=3, normalize=True)
        instances = seg_from_flow(flow, config.tracking_iters, mask=fg)
    
    return instances

def instance_from_edt_and_semantic(raw, config):

    '''
    Parameters should be set in config:
        dist_mode: 'thresholding', 'tracking'
        edt_instance_thres: thres to get instance seeds, if dist_mode == 'thresholding'
        semantic_bg: lalel of background label
    '''
    semantic = np.squeeze(np.argmax(raw['semantic'], axis=-1)).astype(np.uint16)
    dist = np.squeeze(raw['edt'])
    dist = gaussian(dist, sigma=1)
    fg =  closing(semantic != config.semantic_bg, square(1))
    
    if config.edt_mode == 'thresholding':
        instances = label(dist > config.edt_instance_thres)
        instances = watershed(-dist, instances) * fg
    elif config.edt_mode == 'tracking':
        flow = get_flow(dist, sigma=3, normalize=True)
        instances = seg_from_flow(flow, config.tracking_iters, mask=fg)
    
    return instances

def instance_from_flow(flow, mask, config):

    '''
    Args:
        flow: H x W
        mask: H x W
    Parameters should be set in config:
    '''
    instances = seg_from_flow(flow, config.tracking_iters, mask=mask)
    
    return instances






    
# from mws import MutexPixelEmbedding
# def mutex(pred):

#     embedding = np.squeeze(pred['embedding'])
#     emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    
#     semantic = np.squeeze(np.argmax(pred['semantic'], axis=-1)) if 'semantic' in pred.keys() else np.ones(embedding.shape[:-1])

#     m = MutexPixelEmbedding(similarity='cos', lange_range=8, min_size=10)
#     seg = m.run(embedding, semantic>0)

#     return label(seg)
