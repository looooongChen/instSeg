import tensorflow as tf
import tensorflow.keras.backend as K 
import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.measure import label as relabel
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from instSeg.constant import * 
# MAX_OBJ_ADJ_MATRIX = 10


def image_resize_np(images, sz, method='bilinear'):
    '''
    Args:
        images: B x H x W x C or list of H x W x C
        sz: (heigh, width)
    '''
    resized = []
    for img in images:
        if img.shape[0] == sz[0] and img.shape[1] == sz[1]:
            resized.append(img)
        elif method == 'nearest':
            resized.append(cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST))
        else:
            resized.append(cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR))
    resized = np.array(resized)
    # to keep shape unchanged
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, axis=-1)
    return resized


# def relabel_instance(instance, background=0):
#     '''
#     Args:
#         instance: label map of size B x H x W x 1 or B x H x W
#     '''
#     pbar = tqdm(instance, desc='Relabel instance map:', ncols=100)
#     for idx, _ in enumerate(pbar):
#         instance[idx,:,:,0] = relabel(instance[idx,:,:,0], background=background)
#     return instance

def trim_images(images, sz, interpolation='bilinear', process_disp=True):
    '''
    Args:
        images:
            - list of instance masks: [img1, img2, ...]
            - numpy array: nothing changed
    '''
    if isinstance(images, list):
        images_trimmed = []
        interpolation = cv2.INTER_LINEAR if interpolation == 'bilinear' else cv2.INTER_NEAREST
        if process_disp:
            images = tqdm(images, desc='Processing input images:', ncols=100)
        for m in images:
            if m.shape[0] == sz[0] and m.shape[1] == sz[1]:
                images_trimmed.append(m)
            else:
                images_trimmed.append(cv2.resize(m, (sz[1], sz[0]), interpolation=interpolation)) 
        images_trimmed = np.array(images_trimmed)
    else:
        images = image_resize_np(images, sz, method=interpolation)
        images_trimmed = images
    if len(images_trimmed.shape) == 3:
        images_trimmed = np.expand_dims(images_trimmed, axis=-1)
    return images_trimmed

def trim_instance_label(masks, sz, process_disp=True):
    '''
    Args:
        masks:
            - list of instance masks: [[img1_obj1, img1_obj2, ...], [img2_obj1, img2_obj2, ...], ...]
            - list of instance masks: [mask1, mask2, ...]
            - numpy array: nothing changed
    '''
    if isinstance(masks, list) and isinstance(masks[0], list):
        obj_max = np.max([len(M) for M in masks])
        assert obj_max < MAX_OBJ_MASK
        assert obj_max > 1
        masks_trimmed = []
        if process_disp:
            masks = tqdm(masks, desc='Processing label map:', ncols=100)
        for M in masks:
            M_trimmed = np.zeros((sz[0], sz[1], obj_max), bool)
            for idx in range(len(M)):
                if M[idx].shape[0] == sz[0] and M[idx].shape[1] == sz[1]:
                    M_trimmed[...,idx] = cv2.resize(M[idx].astype(np.uint8), (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST) > 0
                else:
                    M_trimmed[...,idx] = M[idx] > 0
            masks_trimmed.append(M_trimmed)
        masks_trimmed = np.array(masks_trimmed, bool)
    elif isinstance(masks, list):
        masks_trimmed = [cv2.resize(m.astype(np.int32), (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST) for m in masks]
        masks_trimmed = np.array(masks_trimmed)
    else:
        masks_trimmed = masks.astype(np.int32)
    if len(masks_trimmed.shape) == 3:
        masks_trimmed = np.expand_dims(masks_trimmed, axis=-1)
    return masks_trimmed


def labeled_instance(masks, overlap_label=0):
    '''
    Args:
        masks: N x H x W x C
    Return:
        labeled: N x H x W x 1, bg = 0, overlap=overlap_label
    '''
    if masks.shape[-1] > 1:
        masks = masks > 0
        S = np.sum(masks, axis=-1, keepdims=True)
        non_overlap, overlap =  S == 1, S > 1

        L = np.sum(masks * np.expand_dims(np.arange(masks.shape[-1])+1, axis=(0,1,2)), axis=-1, keepdims=True)
        L = L * non_overlap + overlap_label * overlap
    else:
        L = masks
    return L


def adj_matrix(labels, radius):
    
    '''
    backgrounb: 0
    Args:
        labels: label map of size B x H x W x 1 or B x H x W or B x H x W x # objects
        radius: radius to determine neighbout relationship
    '''
    if radius < 1:
        radius = int(max(labels.shape[1:]) * radius)
    D = disk_np(radius//2)
    labels = labels.astype(np.int16)
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)

    adjs = []
    pbar = tqdm(labels, desc='Adjacent matrix:', ncols=100)
    for l in pbar:
        if l.shape[-1] == 1:
            label_stack = np.zeros((max(np.unique(l))+1, *l.shape[:2]), dtype=np.uint8)
            X, Y = np.meshgrid(np.arange(0, l.shape[0]), np.arange(0, l.shape[1]), indexing='ij')
            label_stack[l.flatten(), X.flatten(), Y.flatten()] = 1
        else:
            label_stack = np.moveaxis(l, -1, 0) > 0
            label_stack = np.pad(label_stack, ((1,0),(0,0),(0,0)), mode='constant', constant_values=0)
            label_stack = label_stack.astype(np.uint8)

        for i in range(label_stack.shape[0]):
            label_stack[i] = cv2.dilate(label_stack[i], D, iterations = 1)
        
        adj = np.zeros((MAX_OBJ_ADJ_MATRIX, MAX_OBJ_ADJ_MATRIX), bool)
        for i in range(label_stack.shape[0]):
            idx_r, idx_c = np.nonzero(label_stack[i])
            if len(idx_r) > 0:
                neighbor = np.any(label_stack[:, idx_r, idx_c], axis=1)
                adj[i, :len(neighbor)] = neighbor
                adj[:len(neighbor), i] = neighbor
        adj[0,:] = True
        adj[:,0] = True
        adjs.append(adj)
    
    return np.array(adjs, dtype=bool)

def mean_embedding(labels, embedding):
    '''
    Args:
        labels: H x W 
        embdding: H x W x C
    '''
    objs = regionprops(labels)
    m = {}
    for obj in objs:
        m[obj.label] = np.mean(embedding[obj.coords[:,0], obj.coords[:,1], :], axis=0)
    return m

def disk_tf(radius, channel=1, dtype=tf.int32): 
    '''
    disk structure element for morhporlogical dilation/erosion
    Return:
        H x W x channel
    '''  
    L = tf.range(-radius, radius + 1)
    X, Y = tf.meshgrid(L, L)
    d = tf.cast((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    return tf.tile(tf.expand_dims(d, axis=-1), [1,1,channel])

def disk_np(radius, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


def edt(label, normalize=True, process_disp=False):
    '''
    Args:
        label: B x H x W x 1 or B x H x W x # objects
    Return:
        dts: B x H x W x 1, distance transform
    overlapping could happen, we set it to zero.
    '''
    # label = np.squeeze(label, axis=-1).astype(np.uint16)
    label = label.astype(np.uint16)
    dts = []
    if process_disp:
        label = tqdm(label, desc='Distance transform:', ncols=100)
    for l in label:
        DT = []
        for idx in range(l.shape[-1]):
            l_dil = cv2.dilate(l[:,:,idx], np.ones((3,3), np.uint16), iterations = 1)
            l_ero = cv2.erode(l[:,:,idx], np.ones((3,3), np.uint16), iterations = 1)
            dt_area = np.logical_and(l_dil == l_ero, l[:,:,idx]>0)
            dt = cv2.distanceTransform(dt_area.astype(np.uint8), cv2.DIST_L2, 3)
            if normalize:
                props = regionprops(l[:,:,idx], intensity_image=dt, cache=True)
                for p in props:
                    rr, cc = p['coords'][:,0], p['coords'][:,1]
                    dt[rr, cc] = dt[rr, cc] / (p['max_intensity'] + 1e-8)
            DT.append(dt)
        DT = np.sum(DT, axis=0) * (np.sum(l>0, axis=-1)==1)
        dts.append(DT)
    return np.expand_dims(np.array(dts), axis=-1)


def contour(label, radius=2, mode='overlap_ignore', process_disp=False):
    '''
    Args:
        label: B x H x W x 1 or B x H x W x # objects
    Return:
        conturs: B x H x W x 1
    overlapping could happen, we use the mid-line of overlapping area (skeletonlize) as the pesudo boundary.
    '''
    # label = np.squeeze(label, axis=-1).astype(np.uint16)
    label = label.astype(np.uint16)
    contours = []
    if process_disp:
        label = tqdm(label, desc='Computing contours:', ncols=100)
    for l in label:
        C = []
        for idx in range(l.shape[-1]):
            l_dil = cv2.dilate(l[:,:,idx], disk_np(1), iterations = 1)
            l_ero = cv2.erode(l[:,:,idx], disk_np(1), iterations = 1)
            C.append(l_dil != l_ero)
        C = np.sum(C, axis=0) > 0
        # overlapping could happen, we use the mid-line of overlapping area (skeletonlize) as the pesudo boundary.
        if l.shape[-1] > 1:
            if mode == 'overlap_midline':
                C = np.logical_or(C, np.sum(l>0, axis=-1)>1)
            elif mode == 'overlap_ignore':
                C_o = cv2.dilate((np.sum(l>0, axis=-1)>1).astype(np.uint8), disk_np(1), iterations = 1)
                C = np.logical_and(C, C_o==0)
        C = skeletonize(C)
        C = cv2.dilate(C.astype(np.uint8), disk_np(radius), iterations = 1)
        contours.append(C)
    return np.expand_dims(np.array(contours, bool), axis=-1)

def layered2stack(layered):
    instances = []
    for idx in range(layered.shape[-1]):
        L = relabel(layered[:,:,idx])
        for r in regionprops(L):
            instances.append(L == r.label)
    instances = np.moveaxis(np.array(instances), 0, -1)
    return instances

def size_screening(instances, obj_min_size, obj_max_size):
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


if __name__ == '__main__':
    import numpy as np 
    # from skimage.io import imread
    # from skimage.draw import circle
    # test = np.zeros((1,8,10,1))
    # test[0, :5, :5, 0] = 1
    # test[0, 5:, 5:, 0] = 2
    # # gt = imread('./cysts/ground_truth/DA PCY 119 09082018JKI.png')
    # # test = np.repeat(np.expand_dims(gt, axis=0), 10, axis=0)
    # # test = np.expand_dims(test, axis=-1)

    # # gt = imread('./cysts/image/DA PCY 119 09082018JKI.tif')
    # # test = np.repeat(np.expand_dims(gt, axis=0), 10, axis=0)

    # import time
    # import matplotlib.pyplot as plt

    # img = imread('I:/instSeg/ds_cyst/export-mask-2020-Dec-07-18-47-41/mask_00000001.png')
    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    # flow = edt_flow_np(img)
    # angle = (np.arctan2(flow[0,:,:,0], flow[0,:,:,1]) + 3.141593)/(2*3.14159)*179
    # vis = np.stack([angle, 200*np.ones_like(angle), 200*np.ones_like(angle)], axis=-1)
    # vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # bg_y, bg_x = np.nonzero(img[0,:,:,0] == 0)
    # vis[bg_y, bg_x, :] = 255

    # color_wheel = np.zeros((512,512,3))
    # rr, cc = circle(256, 256, 200)
    # angle = (np.arctan2(255-rr, 255-cc) + 3.141593)/(2*3.14159)*179
    # color_wheel[rr, cc, 0] = angle
    # color_wheel[rr, cc, 1] = 200
    # color_wheel[rr, cc, 2] = 200
    # color_wheel = cv2.cvtColor(color_wheel.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # # plt.imshow(vis)
    # # plt.show()
    # plt.subplot(1,2,1)
    # plt.imshow(vis)
    # plt.subplot(1,2,2)
    # plt.imshow(color_wheel)
    # plt.show()

    # start = time.time()
    # t = adj_matrix_np(test, radius=2)
    # print(t.shape)
    # tf.image.resize(test, [512,512], method='bilinear', antialias=True, name='img_pre')
    # img_resized = image_resize_np(test, (512,512), method='bilnear')
    # img_normalized = image_normalization_np(img_resized)
    # dts = edt_np(test)
    # print(time.time()-start)

    # import matplotlib.pyplot as plt 
    # # plt.imshow((img_normalized[0]*25+125).astype(np.uint8))
    # # plt.show()
    # plt.imshow(dts[0,:,:,0])
    # plt.show()

    ### adj_matrix test
    # labeled = np.zeros((1,10,10))
    # labeled[0,:4,:4] = 1
    # labeled[0,:4,-4:] = 3

    labeled = np.zeros((1,10,10,5))
    labeled[0,:4,:4,0] = 1
    labeled[0,:4,-4:,3] = 1

    adj = adj_matrix(labeled,3)
    print(adj)

    L = labeled_instance(labeled, overlap_label=0)
    print(L[0,:,:,0])

