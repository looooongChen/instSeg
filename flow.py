import numpy as np
from skimage.measure import label
from scipy.ndimage import gaussian_filter



def diffuse(instance):
    '''
    Args:
        flow: H x W x 2 or 1 x H x W x 2 
        mask: H x W 
        iter: maximal tracking steps
    '''

    if flow.ndim == 4:
        flow = np.squeeze(flow, axis=0)

    shape = flow.shape

    pts = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), axis=-1)
    pts = np.reshape(pts, (-1, 2))
    if mask is not None:
        idx = np.reshape(mask, (-1)) > 0
        pts = pts[idx,:]
    pts_track = pts.copy()

    for _ in range(iters):
        pts_track_floor = np.floor(pts_track).astype(np.uint32)
        pts_track_ceil = np.ceil(pts_track).astype(np.uint32)
        D_tl = flow[pts_track_floor[:,0], pts_track_floor[:,1]]
        D_tr = flow[pts_track_floor[:,0], pts_track_ceil[:,1]]
        D_bl = flow[pts_track_ceil[:,0], pts_track_floor[:,1]]
        D_br = flow[pts_track_ceil[:,0], pts_track_ceil[:,1]]

        W_floor, W_ceil = pts_track_ceil - pts_track + 1e-7, pts_track - pts_track_floor + 1e-7
        W_tl = np.expand_dims(W_floor[:,0] * W_floor[:,1], axis=-1)
        W_tr = np.expand_dims(W_floor[:,0] * W_ceil[:,1], axis=-1)
        W_bl = np.expand_dims(W_ceil[:,0] * W_floor[:,1], axis=-1)
        W_br = np.expand_dims(W_ceil[:,0] * W_ceil[:,1], axis=-1)

        D = (D_tl * W_tl + D_tr * W_tr + D_bl * W_bl + D_br * W_br) / (W_tl + W_tr + W_bl + W_br)

        pts_track = pts_track + D
        pts_track = np.maximum(pts_track, 0)
        pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
        pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)
    
    pts_track = np.round(pts_track).astype(np.int32)

    return pts_track, pts

def get_flow(geo, sigma=3, normalize=False):
    '''
    Args:
        flow: (1) x H x W x (1) 
        normalize: normalize the flow amplitude or not
    '''
    geo = np.squeeze(geo)
    flow = np.stack(np.gradient(geo, axis=[0,1]), axis=-1)
    flow = gaussian_filter(flow, [sigma,sigma,0])
    if normalize:
        flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    return flow

def track_flow(flow, iters, mask=None):
    '''
    Args:
        flow: H x W x 2 or 1 x H x W x 2 
        mask: H x W 
        iter: maximal tracking steps
    '''

    if flow.ndim == 4:
        flow = np.squeeze(flow, axis=0)

    shape = flow.shape

    pts = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), axis=-1)
    pts = np.reshape(pts, (-1, 2))
    if mask is not None:
        idx = np.reshape(mask, (-1)) > 0
        pts = pts[idx,:]
    pts_track = pts.copy()

    for _ in range(iters):
        pts_track_floor = np.floor(pts_track).astype(np.uint32)
        pts_track_ceil = np.ceil(pts_track).astype(np.uint32)
        D_tl = flow[pts_track_floor[:,0], pts_track_floor[:,1]]
        D_tr = flow[pts_track_floor[:,0], pts_track_ceil[:,1]]
        D_bl = flow[pts_track_ceil[:,0], pts_track_floor[:,1]]
        D_br = flow[pts_track_ceil[:,0], pts_track_ceil[:,1]]

        W_floor, W_ceil = pts_track_ceil - pts_track + 1e-7, pts_track - pts_track_floor + 1e-7
        W_tl = np.expand_dims(W_floor[:,0] * W_floor[:,1], axis=-1)
        W_tr = np.expand_dims(W_floor[:,0] * W_ceil[:,1], axis=-1)
        W_bl = np.expand_dims(W_ceil[:,0] * W_floor[:,1], axis=-1)
        W_br = np.expand_dims(W_ceil[:,0] * W_ceil[:,1], axis=-1)

        D = (D_tl * W_tl + D_tr * W_tr + D_bl * W_bl + D_br * W_br) / (W_tl + W_tr + W_bl + W_br)

        pts_track = pts_track + D
        pts_track = np.maximum(pts_track, 0)
        pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
        pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)
    
    pts_track = np.round(pts_track).astype(np.int32)

    return pts_track, pts

def seg_from_flow(flow, iters, mask=None):
    '''
    Args:
        flow: H x W x 2 or 1 x H x W x 2 
        mask: H x W 
        iter: maximal tracking steps
    '''

    if flow.ndim == 4:
        flow = np.squeeze(flow, axis=0)

    seg = np.zeros((flow.shape[0], flow.shape[1]))
    clusters = np.zeros((flow.shape[0], flow.shape[1])).astype(np.uint8)
    pts_track, pts = track_flow(flow, iters, mask)

    clusters[pts_track[:,0], pts_track[:,1]] = 1
    clusters = label(clusters)
    seg[pts[:,0], pts[:,1]] = clusters[pts_track[:,0], pts_track[:,1]]

    return seg.astype(np.int32)


    



if __name__ == '__main__':
    from utils import *
    import numpy as np 
    from skimage.io import imread
    from skimage.draw import circle 
    import time
    import matplotlib.pyplot as plt

    img = imread('I:/instSeg/ds_cyst/export-mask-2020-Dec-07-18-47-41/mask_00000001.png')
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    flow = edt_flow(img)
    

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

    img = np.squeeze(img)
    plt.subplot(1,2,1)
    plt.imshow(img>0)

    # mask = np.zeros_like(img)
    # pts_track, _ = track_flow(flow, iters=40, mask=img>0)
    # mask[pts_track[:,0], pts_track[:,1]] = 1
    mask = seg_from_flow(flow, iters=10, mask=img>0)

    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.show()
