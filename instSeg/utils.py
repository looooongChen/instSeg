import tensorflow as tf
import tensorflow.keras.backend as K 
import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.measure import label as relabel
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def image_resize_np(images, sz, method='bilinear'):
    '''
    Args:
        images: B x H x W x C or list of H x W x C
        sz: (heigh, width)
    '''
    resized = []
    for img in images:
        if method == 'nearest':
            resized.append(cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST))
        else:
            resized.append(cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR))
    resized = np.array(resized)
    # to keep shape unchanged
    if len(resized.shape) == 3:
        resized = np.expand_dims(resized, axis=-1)
    return resized

# def image_normalization_np(images):
#     '''
#     Args:
#         images: label map of size B x H x W x C
#     '''
#     normalized = []
#     for img in images:
#         normalized.append((img-img.mean())/(img.std()+1e-8))
#     return np.array(normalized)

def relabel_instance(instance, background=0):
    '''
    Args:
        instance: label map of size B x H x W x 1 or B x H x W
    '''
    pbar = tqdm(instance, desc='Relabel instance map:', ncols=100)
    for idx, _ in enumerate(pbar):
        instance[idx,:,:,0] = relabel(instance[idx,:,:,0], background=background)
    return instance

def adj_matrix(labels, radius, max_obj=300):
    
    '''
    Args:
        labels: label map of size B x H x W x 1 or B x H x W
        radius: radius to determine neighbout relationship
    '''
    if radius < 1:
        radius = int(max(labels.shape[1:]) * radius)
    labels = labels.astype(np.int16)
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=-1)

    adjs = []
    pbar = tqdm(labels, desc='Adjacent matrix:', ncols=100)
    for l in pbar:
        label_depth = len(np.unique(l))
        label_stack = np.zeros((label_depth, *l.shape[:2]), dtype=np.uint8)

        X, Y = np.meshgrid(np.arange(0, l.shape[0]), np.arange(0, l.shape[1]), indexing='ij')
        label_stack[l.flatten(), X.flatten(), Y.flatten()] = 1
        for i in range(label_depth):
            label_stack[i] = cv2.dilate(label_stack[i], disk_np(radius), iterations = 1)
        label_stack = label_stack * np.moveaxis(l, -1, 0)
        adj = np.zeros((max_obj, max_obj), np.bool)
        for i in range(label_depth):
            neighbor = np.unique(label_stack[i])
            adj[i, neighbor] = True
            adj[neighbor, i] = True
        adjs.append(adj)
    
    return np.array(adjs, dtype=np.bool)

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


def edt(label, normalize=True, bg=False, process_disp=False):
    '''
    Args:
        label: B x H x W x 1
    Return:
        dts: B x H x W x 1, distance transform
    '''
    label = np.squeeze(label, axis=-1).astype(np.uint16)
    dts = []
    if process_disp:
        label = tqdm(label, desc='Distance transform:', ncols=100)
    for l in label:
        l_dil = cv2.dilate(l, np.ones((3,3), np.uint16), iterations = 1)
        l_ero = cv2.erode(l, np.ones((3,3), np.uint16), iterations = 1)
        dt_area = np.logical_and(l_dil == l_ero, l>0)
        dt = cv2.distanceTransform(dt_area.astype(np.uint8), cv2.DIST_L2, 3)
        if bg:
            dt_bg = cv2.distanceTransform(np.logical_not(dt_area).astype(np.uint8), cv2.DIST_L2, 3)
            dt = dt - dt_bg
        if normalize:
            props = regionprops(l, intensity_image=dt, cache=True)
            for p in props:
                rr, cc = p['coords'][:,0], p['coords'][:,1]
                dt[rr, cc] = dt[rr, cc] / (p['max_intensity'] + 1e-8)
        dts.append(dt)
    return np.expand_dims(np.array(dts), axis=-1)

def edt_flow(label, normalize=True, bg=False, process_disp=False):
    flow = edt(label, normalize=False, bg=bg)
    flow = np.concatenate(np.gradient(flow, axis=[1,2]), axis=-1)
    flow = gaussian_filter(flow, [0,3,3,0])
    if normalize:
        flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    return flow


def contour(label, radius=2, process_disp=False):
    '''
    Args:
        label: B x H x W x 1
    Return:
        conturs: B x H x W x 1
    '''
    label = np.squeeze(label, axis=-1).astype(np.uint16)
    contours = []
    if process_disp:
        label = tqdm(label, desc='Computing contours:', ncols=100)
    for l in label:
        l_dil = cv2.dilate(l, disk_np(radius), iterations = 1)
        l_ero = cv2.erode(l, disk_np(radius), iterations = 1)
        contours.append(l_dil != l_ero)
    return np.expand_dims(np.array(contours, np.bool), axis=-1)



if __name__ == '__main__':
    import numpy as np 
    from skimage.io import imread
    from skimage.draw import circle
    test = np.zeros((1,8,10,1))
    test[0, :5, :5, 0] = 1
    test[0, 5:, 5:, 0] = 2
    # gt = imread('./cysts/ground_truth/DA PCY 119 09082018JKI.png')
    # test = np.repeat(np.expand_dims(gt, axis=0), 10, axis=0)
    # test = np.expand_dims(test, axis=-1)

    # gt = imread('./cysts/image/DA PCY 119 09082018JKI.tif')
    # test = np.repeat(np.expand_dims(gt, axis=0), 10, axis=0)

    import time
    import matplotlib.pyplot as plt

    img = imread('I:/instSeg/ds_cyst/export-mask-2020-Dec-07-18-47-41/mask_00000001.png')
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    flow = edt_flow_np(img)
    angle = (np.arctan2(flow[0,:,:,0], flow[0,:,:,1]) + 3.141593)/(2*3.14159)*179
    vis = np.stack([angle, 200*np.ones_like(angle), 200*np.ones_like(angle)], axis=-1)
    vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_HSV2RGB)
    bg_y, bg_x = np.nonzero(img[0,:,:,0] == 0)
    vis[bg_y, bg_x, :] = 255

    color_wheel = np.zeros((512,512,3))
    rr, cc = circle(256, 256, 200)
    angle = (np.arctan2(255-rr, 255-cc) + 3.141593)/(2*3.14159)*179
    color_wheel[rr, cc, 0] = angle
    color_wheel[rr, cc, 1] = 200
    color_wheel[rr, cc, 2] = 200
    color_wheel = cv2.cvtColor(color_wheel.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # plt.imshow(vis)
    # plt.show()
    plt.subplot(1,2,1)
    plt.imshow(vis)
    plt.subplot(1,2,2)
    plt.imshow(color_wheel)
    plt.show()

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
