import numpy as np
import cv2
from skimage.morphology import disk, square, dilation, closing, opening
from skimage.measure import regionprops
from skimage.measure import label as label_connected_component
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from instSeg.flow import *
from scipy.ndimage.filters import gaussian_filter

def instance_from_foreground(raw, config):

    # foreground = np.squeeze(np.argmax(raw['foreground'], axis=-1)).astype(np.uint16)
    foreground = np.squeeze(raw['foreground']>config.thres_foreground).astype(np.uint16)
    instances = label_connected_component(foreground).astype(np.uint16)

    return instances

def instance_from_foreground_and_contour(raw, config):

    # foreground = np.squeeze(np.argmax(raw['foreground'], axis=-1)).astype(np.uint16)
    foreground = np.squeeze(raw['foreground']>config.thres_foreground).astype(np.uint16)
    contour = cv2.dilate((np.squeeze(raw['contour']) > config.thres_contour).astype(np.uint8), square(3), iterations = 1)

    instances = label_connected_component(foreground * (contour == 0)).astype(np.uint16)
    fg = (foreground > 0).astype(np.uint16)
    while True:
        pixel_add = cv2.dilate(instances, square(3), iterations = 1) * (instances == 0) * fg
        if np.sum(pixel_add) != 0:
            instances += pixel_add
        else:
            break
    
    return instances

def instance_from_emb_and_edt(raw, config):
    '''
    raw: a dict containing predictions of at least 'embedding', 'edt'
    Parameters should be set in config:
        emb_cluster_thres: threshold distinguishing object in the embedding space
        emb_cluster_max_step: max step for expanding the instance region
        edt_instance_thres: threshold to get seeds from distance regression map
        dist_intensity: if only instance is not evident, ignore
    '''
    embedding, edt = np.squeeze(raw['embedding']), np.squeeze(raw['edt'])
    # embedding = smooth_emb(embedding, 3)
    emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    edt = gaussian(edt, sigma=1)
    objs = label_connected_component(edt > config.edt_instance_thres)
    if 'foreground' in raw.keys():
        fg = np.squeeze(raw['foreground'] > config.thres_foreground).astype(np.uint16)
    else:
        fg = np.ones(objs.shape).astype(np.uint16)
    props = regionprops(objs)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]
        emb_mean = np.mean(embedding[row, col], axis=0)
        emb_mean = emb_mean/np.linalg.norm(emb_mean)
        mean[p.label] = emb_mean

    while True:
        dilated = dilation(objs, square(3))
        front_r, front_c = np.nonzero((objs != dilated) * (objs == 0) * fg)

        similarity = [np.dot(embedding[r, c, :], mean[dilated[r, c]])
                      for r, c in zip(front_r, front_c)]
        add_ind = np.array([s > 0.7 for s in similarity])

        if np.all(add_ind == False):
            break
        objs[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    for p in regionprops(objs, intensity_image=edt):
        if p.mean_intensity < config.obj_min_edt:
            objs[p.coords[:,0], p.coords[:,1]] = 0

    return objs

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
        instances = label_connected_component(dist > config.edt_instance_thres)
        instances = watershed(-dist, instances) * fg
    elif config.edt_mode == 'tracking':
        flow = get_flow(dist, sigma=3, normalize=True)
        instances = seg_from_flow(flow, config.tracking_iters, mask=fg)
    
    return instances


def instance_from_embedding(raw, config):
    layered = np.squeeze(raw['embedding']) > 0.5
    instances = []
    for idx in range(layered.shape[-1]):
        L = label_connected_component(layered[:,:,idx])
        for r in regionprops(L):
            instances.append(L == r.label)
    if len(instances) == 0:
        instances.append(np.zeros(layered.shape[:2]))
    instances = np.moveaxis(np.array(instances), 0, -1)
    return instances


def instance_from_layered_embedding(raw, config):
    
    embedding = np.copy(np.squeeze(raw['layered_embedding']))
    fg = np.copy(np.squeeze(raw['foreground'] > config.thres_foreground))
    for r in regionprops(label_connected_component(fg)):
        if r.area < config.obj_min_size:
            fg[r.coords[:,0], r.coords[:,1]] = 0

    # layering foreground
    layered_objects = np.zeros(embedding.shape, bool)
    rr, cc = np.nonzero(fg)
    # layered_objects[rr, cc, :] = np.logical_or(embedding[rr, cc, :] > 0.5, embedding[rr, cc, :] == np.amax(embedding[rr, cc, :], axis=-1, keepdims=True))
    layered_objects[rr, cc, :] = embedding[rr, cc, :] > 0.5
    idx = np.argmax(embedding[rr, cc, :], axis=-1)
    layered_objects[rr, cc, idx] = 1
    # get instances
    instances = []
    for idx in range(layered_objects.shape[-1]):
        L = label_connected_component(layered_objects[:,:,idx])
        for r in regionprops(L):
            if r.area < config.obj_min_size:
                layered_objects[r.coords[:,0], r.coords[:,1], idx] = 0
            else:
                instances.append(L == r.label)
    
    return instances, layered_objects


