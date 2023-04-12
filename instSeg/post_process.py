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
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.measure import label as relabel
from scipy.spatial import distance_matrix


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


def embedding_meanshift(raw, config):
    embedding = np.squeeze(raw['embedding'])
    if config.loss['embedding'] == 'cosine':
        embedding = embedding / (np.sqrt(np.sum(embedding ** 2, axis=-1, keepdims=True)) + 1e-12)
    sz = (embedding.shape[0], embedding.shape[1])
    clusters = np.zeros(sz, np.uint16)
    
    if 'foreground' in raw.keys():
        fg = np.copy(np.squeeze(raw['foreground'] > config.thres_foreground))
    else:
        fg = np.ones(sz)
    
    rr, cc = np.nonzero(fg)
    if len(rr) > 0:
        embedding = embedding[rr, cc, :]
        # sz, dim = embedding.shape[:-1], embedding.shape[-1]
        
        # embedding = np.reshape(embedding, (-1, dim))
        # bandwidth = estimate_bandwidth(embedding, quantile=0.2, n_samples=500)
        # print(bandwidth)
        if config.loss['embedding'] == 'euclidean':
            bandwidth = 2*config.margin_attr if config.margin_attr != 0 else 0.1
        if config.loss['embedding'] == 'cosine':
            bandwidth = 0.5
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(embedding)
        clusters[rr, cc] = ms.labels_ + 1
        labels = relabel(clusters)
    else:
        labels = clusters

    return labels, clusters


def embedding_layering(raw, config):
    
    embedding = np.copy(np.squeeze(raw['embedding']))
    fg = np.copy(np.squeeze(raw['foreground'] > config.thres_foreground))
    for r in regionprops(label_connected_component(fg)):
        if r.area < config.obj_min_size:
            fg[r.coords[:,0], r.coords[:,1]] = 0

    # layering foreground
    layered_objects = np.zeros(embedding.shape, bool)
    rr, cc = np.nonzero(fg)
    # layered_objects[rr, cc, :] = np.logical_or(embedding[rr, cc, :] > 0.5, embedding[rr, cc, :] == np.amax(embedding[rr, cc, :], axis=-1, keepdims=True))    
    layered_objects[rr, cc, :] = embedding[rr, cc, :] > 0.5
    if config.non_overlap:
        layered_objects[rr, cc, :] = layered_objects[rr, cc, :] * 0
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


def size_screening(instances,config):
    obj_min_size, obj_max_size = config.obj_min_size, config.obj_max_size
    if obj_min_size > 0 or obj_max_size < float('inf'):
        if isinstance(instances, list):
            sz = [np.sum(obj) for obj in instances]
            instances = [obj for s, obj in zip(sz, instances) if s > obj_min_size and s < obj_max_size]
        elif len(instances.shape) == 2:
            for r in regionprops(instances):
                if r.area <obj_min_size or r.area > obj_max_size:
                    instances[r.coords[:,0], r.coords[:,1]] = 0
        elif len(instances.shape) == 3:
            sz = np.apply_over_axes(np.sum, instances>0, [1,2])
            sz = np.squeeze(np.squeeze(sz, axis=1), axis=1)
            idx = np.logical_and(sz > obj_min_size, sz < obj_max_size)
            instances = instances[idx]
    return instances


def fill_gap(instances, raw, config):
    if 'foreground' not in raw.keys():
        return instances
    if len(instances.shape) != 2:
        return instances
    

    fg = np.copy(np.squeeze(raw['foreground'] > config.thres_foreground))
    coord_gap = np.array(np.nonzero(fg * (instances == 0))).T
    coord_instance = np.array(np.nonzero(instances > 0)).T

    D = distance_matrix(coord_gap, coord_instance)
    D = np.argmin(D, axis=1)
    instances[coord_gap[:,0], coord_gap[:,1]] = instances[coord_instance[D,0], coord_instance[D,1]]
    
    # while True:
    #     instances_d = dilation(instances, square(3))

    #     front = instances_d * fg * (instances == 0)
    #     print(np.count_nonzero(front))
    #     if np.count_nonzero(front) == 0:
    #         break
    #     rr, cc = np.nonzero(front)
    #     instances[rr, cc] = instances_d[rr, cc]
    
    return instances

