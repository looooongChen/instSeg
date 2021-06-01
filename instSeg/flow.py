import numpy as np
from numpy.core.fromnumeric import argmin
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import scipy
from skimage.morphology import white_tophat
from skimage.measure import regionprops
from skimage.measure import label as label_connected_component
from scipy.spatial import distance_matrix

DISK = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
SQUARE = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
D_DIG = 2 ** 0.5

def boundary(label):
    '''
    Args:
        label: (1) x H x W x (1)
    Return:
        boundary: H x W
    '''
    label = np.squeeze(label).astype(np.uint16)
    l_dil = cv2.dilate(label, DISK, iterations = 1)
    l_ero = cv2.erode(label, DISK, iterations = 1)
    B = (l_dil != l_ero) * label

    return B


def get_center(instance_mask, ctype='mass_center'):
    instance_mask = instance_mask.astype(np.uint16)
    if ctype == 'mass_center':
        labels = np.unique(instance_mask[instance_mask!=0])
        centers = np.round(np.array(scipy.ndimage.center_of_mass(instance_mask>0, labels=instance_mask, index=[int(l) for l in labels]))).astype(int)
        # check if center in the object or not
        not_in = instance_mask[centers[:,0], centers[:,1]] != labels
        for idx in np.nonzero(not_in)[0]:
            yy, xx = np.nonzero(instance_mask == labels[idx])
            idmin = np.argmin((yy-centers[idx,0])**2 + (xx-centers[idx,1])**2)
            centers[idx,0], centers[idx,1] = yy[idmin], xx[idmin]
        return centers, labels
    if ctype == 'edt_max':
        instance_mask[:,[0,-1]] = 0
        instance_mask[[0,-1],:] = 0
        centers, labels = [], []
        mask = (1-(boundary(instance_mask)>0))*(instance_mask>0)
        edt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        for p in regionprops(instance_mask):
            labels.append(p.label)
            dst = edt[p.coords[:,0], p.coords[:,1]]
            idmin = np.argmax(dst)
            centers.append(p.coords[idmin])
        return np.array(centers), np.array(labels)

def spath(instance_mask, mode='boundary'):
    if mode == 'boundary':
        B = boundary(instance_mask)
        D = np.zeros(instance_mask.shape)
        coordB = np.array(np.nonzero(B)).T
        coordF = np.array(np.nonzero(instance_mask)).T
        dm = distance_matrix(coordF, coordB, p=2, threshold=10000)
        D[coordF[:,0], coordF[:,1]] = np.amin(dm, axis=1)
        return D

# def shortest_distance_transform(instance_mask, mode='outwards', ctype='center_of_mass', niter=100):
#     '''
#     shortest distance to center, path through background is not allowed
#     Args:
#         label: (1) x H x W x (1)
#         ctype: center type, 'center_of_mass', 'edt_max', ''
#     '''
#     sz_pad = (instance_mask.shape[0]+2, instance_mask.shape[1]+2)
#     pim = np.zeros(sz_pad)
#     pim[1:-1,1:-1] = instance_mask
#     dt = np.zeros(sz_pad) + np.inf
#     visited = np.zeros(sz_pad, np.bool)
#     if mode == 'outwards':
#         centers, labels = get_center(instance_mask, ctype)
#         centers += 1
#         # initialize distance map 
#         dt[centers[:,0], centers[:,1]] = 0
#         # initialize visit map
#         visited[pim==0] = True
#         visited[centers[:,0], centers[:,1]] = True
#     elif mode == "inwards":
#         b = boundary(pim)
#         # initialize distance map 
#         # dt += np.inf
#         dt[b!=0] = 1
#         # initialize visit map
#         visited[pim==0] = True
#         visited[b!=0] = True
#     else:
#         print("Invalid Mode: ", mode)
#     # compute the shortest path to centers
#     iter = 0
#     idr, idc = np.nonzero(np.logical_not(visited))
#     n_idr = np.array([idr, idr, idr+1, idr+1, idr+1, idr, idr-1, idr-1, idr-1])
#     n_idc = np.array([idc, idc+1, idc+1, idc, idc-1, idc-1, idc-1, idc, idc+1])
#     offset = np.expand_dims(np.array([0, 1,D_DIG,1,D_DIG,1,D_DIG,1,D_DIG]), axis=-1)
#     changed = True
#     while changed and iter<niter:
#         # get neighbor distance:
#         D_neighbor = dt[n_idr, n_idc] + offset
#         D_neighbor[pim[n_idr, n_idc] != np.expand_dims(pim[idr, idc], 0)] = np.inf
#         idx = np.argmin(D_neighbor, axis=0)
#         D_neighbor = D_neighbor[idx, range(len(idx))]
#         update = D_neighbor != np.inf
#         # update distance and coordinated
#         D_neighbor = D_neighbor[update]
#         idru, idcu = idr[update], idc[update]
#         changed = np.any(dt[idru, idcu] != D_neighbor)
#         if changed:
#             dt[idru, idcu] = D_neighbor
#             visited[idru, idcu] = True
#         iter += 1

#     return dt[1:-1,1:-1]

def diffuse(instance_mask, mode='outwards', ctype='center_of_mass', niter=150):
    '''
    shortest distance to center, path through background is not allowed
    Args:
        label: (1) x H x W x (1)
        ctype: center type, 'mass_center', 'edt_max', 's2s'
    '''
    # if mode == 'mix':
    #     D1, flow1 =  diffuse(instance_mask, mode='outwards', ctype=ctype, niter=niter)
    #     D2, flow2 =  diffuse(instance_mask, mode='inwards', ctype=ctype, niter=niter)
    #     D = D1 + D2
    #     flow = flow1 + flow2
    #     flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    #     return D, flow

    instance_mask = np.copy(instance_mask).astype(np.uint16)
    instance_mask = cv2.dilate(instance_mask, SQUARE, iterations = 1)
    sz_pad = (instance_mask.shape[0]+2, instance_mask.shape[1]+2)
    L = np.zeros(sz_pad)
    L[1:-1,1:-1] = instance_mask
    D = np.zeros(sz_pad)
    assert mode in ['inwards', 'outwards', 's2s']
    if mode == 'outwards':
        heating = np.zeros(sz_pad)
        centers, labels = get_center(L, ctype)
        heating[centers[:,0], centers[:,1]] = 1
    if mode == "inwards":
        heating = boundary(L) > 0
    if mode == "s2s":
        heating = boundary(L) > 0
        cooling = np.zeros(sz_pad)
        centers, labels = get_center(L, ctype)
        cooling[centers[:,0], centers[:,1]] = np.sum(heating)
    # compute the shortest path to centers
    iter = 0
    idr, idc = np.nonzero(L)
    n_idr = np.array([idr, idr, idr+1, idr+1, idr+1, idr, idr-1, idr-1, idr-1])
    n_idc = np.array([idc, idc+1, idc+1, idc, idc-1, idc-1, idc-1, idc, idc+1])
    D = D + heating
    if mode == 's2s':
        D = D - cooling
    while iter<niter:
        D_neighbor = D[n_idr, n_idc]
        is_neighbor = L[n_idr, n_idc] == np.expand_dims(L[idr, idc], 0)
        S = np.sum(D_neighbor * is_neighbor, axis=0)/(np.sum(is_neighbor, axis=0)+1e-7)
        D[idr, idc] = S
        D += heating
        if mode == 's2s':
            D = D - cooling
        iter += 1

    flow_y, flow_x = D[1:,1:-1]-D[:-1,1:-1], D[1:-1,1:]-D[1:-1,:-1]
    flow = np.stack((flow_y[:-1,:], flow_x[:,:-1]), axis=-1)
    flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)

    if mode == 'inwards' or mode == 's2s':
        flow = -flow

    return D[1:-1,1:-1], flow


def dt2flow(sdt, normalize=False):
    '''
    Args:
        sdt: (1) x H x W x (1) 
        sigma: smooth vector fild
        normalize: normalize the flow amplitude or not
    Return:
        flow: flow: H x W x 2 
    '''
    sdt = np.squeeze(sdt)
    # sdt = np.log(1+sdt)
    # sdt = gaussian_filter(sdt, [2,2])
    flow = np.stack((cv2.Sobel(sdt,cv2.CV_64F,0,1,ksize=3), cv2.Sobel(sdt,cv2.CV_64F,1,0,ksize=3)), axis=-1)

    # flow = np.stack(np.gradient(sdt, axis=[0,1]), axis=-1)
    if normalize:
        flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    return flow

# def flow(instance_mask, mode='inwards'):
#     instance_mask = np.squeeze(instance_mask)
#     if mode == 'inwards':
#         instance_mask = instance_mask.astype(np.uint16)
#         l_dil = cv2.dilate(instance_mask, np.ones((3,3), np.uint16), iterations = 1)
#         l_ero = cv2.erode(instance_mask, np.ones((3,3), np.uint16), iterations = 1)
#         boundary = np.logical_and(l_dil == l_ero, instance_mask>0)
#         sd = cv2.distanceTransform(boundary.astype(np.uint8), cv2.DIST_L2, 3)
#         return sd

def visualize_flow(flow, mask=None):
    '''
    Args:
        flow: H x W x 2
    '''
    if mask is None:
        mask = np.logical_and(flow[:,:,0] < 1e-5, flow[:,:,1] == 1e-5)
    else:
        mask = mask == 0
    angle = (np.arctan2(flow[:,:,0], flow[:,:,1]) + 3.141593)/(2*3.14159)*179
    visD = np.stack([angle, 200*np.ones_like(angle), 200*np.ones_like(angle)], axis=-1)
    visD = cv2.cvtColor(visD.astype(np.uint8), cv2.COLOR_HSV2RGB)
    bg_y, bg_x = np.nonzero(mask)
    visD[bg_y, bg_x, :] = 255
    visA = np.sum(flow**2, axis=-1) ** 0.5

    # color_wheel = np.zeros((512,512,3))
    # rr, cc = circle(256, 256, 200)
    # angle = (np.arctan2(255-rr, 255-cc) + 3.141593)/(2*3.14159)*179
    # color_wheel[rr, cc, 0] = angle
    # color_wheel[rr, cc, 1] = 200
    # color_wheel[rr, cc, 2] = 200
    # color_wheel = cv2.cvtColor(color_wheel.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return visD.astype(np.uint8), visA

def track_flow(flow, niter, step=0.5, mask=None):
    '''
    Args:
        flow: H x W x 2 or 1 x H x W x 2 
        mask: H x W, only pixels in mask will be tracked 
        iter: maximal tracking steps
    Return:
        pts_track: position of tracked points after "niter" tracking steps
        pts: the tracked points
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

    for _ in range(niter):
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

        pts_track = pts_track + D * step
        pts_track = np.maximum(pts_track, 0)
        pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
        pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)
    
    pts_track = np.round(pts_track).astype(np.int32)

    return pts_track, pts

# def track_flow(flow, niter, step=0.5, mask=None):
#     '''
#     Args:
#         flow: H x W x 2 or 1 x H x W x 2 
#         mask: H x W, only pixels in mask will be tracked 
#         iter: maximal tracking steps
#     Return:
#         pts_track: position of tracked points after "niter" tracking steps
#         pts: the tracked points
#     '''

#     if flow.ndim == 4:
#         flow = np.squeeze(flow, axis=0)

#     shape = flow.shape
#     acc = np.zeros((shape[0], shape[1]))

#     pts = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), axis=-1)
#     pts = np.reshape(pts, (-1, 2))
#     if mask is not None:
#         idx = np.reshape(mask, (-1)) > 0
#         pts = pts[idx,:]
#     pts_track = pts.copy()

#     for _ in range(niter):
#         pts_track_floor = np.floor(pts_track).astype(np.uint32)
#         pts_track_ceil = np.ceil(pts_track).astype(np.uint32)
#         D_tl = flow[pts_track_floor[:,0], pts_track_floor[:,1]]
#         D_tr = flow[pts_track_floor[:,0], pts_track_ceil[:,1]]
#         D_bl = flow[pts_track_ceil[:,0], pts_track_floor[:,1]]
#         D_br = flow[pts_track_ceil[:,0], pts_track_ceil[:,1]]

#         W_floor, W_ceil = pts_track_ceil - pts_track + 1e-7, pts_track - pts_track_floor + 1e-7
#         W_tl = np.expand_dims(W_floor[:,0] * W_floor[:,1], axis=-1)
#         W_tr = np.expand_dims(W_floor[:,0] * W_ceil[:,1], axis=-1)
#         W_bl = np.expand_dims(W_ceil[:,0] * W_floor[:,1], axis=-1)
#         W_br = np.expand_dims(W_ceil[:,0] * W_ceil[:,1], axis=-1)

#         D = (D_tl * W_tl + D_tr * W_tr + D_bl * W_bl + D_br * W_br) / (W_tl + W_tr + W_bl + W_br)

#         pts_track = pts_track + D * step
#         pts_track = np.maximum(pts_track, 0)
#         pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
#         pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)

#         pts_track_int = np.round(pts_track).astype(np.int32)
#         acc[pts_track_int[:,0], pts_track_int[:,1]] += 1
    
#     pts_track = np.round(pts_track).astype(np.int32)

#     return pts_track, pts, acc

def seg_from_flow(flow, niter, step=0.5, mask=None):
    '''
    Args:
        flow: H x W x 2 or 1 x H x W x 2 
        niter: maximal tracking steps
        mask: H x W 
    Return:
        seg: instance segmenations
        clusters: connected components after "niter" tracking steps 
    '''

    if flow.ndim == 4:
        flow = np.squeeze(flow, axis=0)

    seg = np.zeros((flow.shape[0], flow.shape[1]))
    clusters = np.zeros((flow.shape[0], flow.shape[1])).astype(np.uint8)
    pts_track, pts = track_flow(flow, niter, step=step, mask=mask)

    clusters[pts_track[:,0], pts_track[:,1]] = 1
    clusters = label_connected_component(clusters)
    seg[pts[:,0], pts[:,1]] = clusters[pts_track[:,0], pts_track[:,1]]

    return seg.astype(np.int32), clusters

# def seg_from_flow(flow, niter, mode='inwards', step=0.5, mask=None):
#     '''
#     Args:
#         flow: H x W x 2 or 1 x H x W x 2 
#         niter: maximal tracking steps
#         mask: H x W 
#     Return:
#         seg: instance segmenations
#         clusters: connected components after "niter" tracking steps 
#     '''

#     if flow.ndim == 4:
#         flow = np.squeeze(flow, axis=0)
#     seg = np.zeros((flow.shape[0], flow.shape[1]))
#     clusters = np.zeros((flow.shape[0], flow.shape[1])).astype(np.uint8)

#     if mode == 'outwards':
#         pts_track, pts, _ = track_flow(flow, niter, step=step, mask=mask)

#         clusters[pts_track[:,0], pts_track[:,1]] = 1
#         clusters = label_connected_component(clusters)
#         seg[pts[:,0], pts[:,1]] = clusters[pts_track[:,0], pts_track[:,1]]
#     if mode == "inwards":

#         pts_track, pts, acc = track_flow(flow, niter, step=step, mask=mask)

#         clusters = white_tophat(acc, square(3)) > 5
#         # clusters[pts_track[:,0], pts_track[:,1]] = 1
#         clusters = label_connected_component(clusters)
#         seg[pts[:,0], pts[:,1]] = clusters[pts_track[:,0], pts_track[:,1]]
#         # pts_track, pts = track_flow(-flow, niter, step=step, mask=mask)
#         # clusters[pts_track[:,0], pts_track[:,1]] = np.linalg.norm(pts_track - pts, ord=2, axis=1)

#     return seg.astype(np.int32), clusters


    

if __name__ == '__main__':
    from utils import *
    import numpy as np 
    from skimage.io import imread, imsave
    from skimage.color import label2rgb
    from skimage.draw import circle 
    from skimage.morphology import area_closing
    from skimage.morphology import dilation, disk, erosion, square, skeletonize
    import time
    import matplotlib.pyplot as plt
    import os

    def vis(seg, gt):
        vis_seg = seg + 1
        vis_seg = label2rgb(vis_seg, bg_label=0)
        rr, cc = np.nonzero(seg == 0)
        vis_seg[rr,cc,:] = 1
        brr, bcc = np.nonzero(boundary(gt))
        vis_seg[brr, bcc, 0] = 1
        vis_seg[brr, bcc, 1] = 0
        vis_seg[brr, bcc, 2] = 0
        vis_seg = (vis_seg * 255).astype(np.uint8)
        return vis_seg

    # img_path = "./MP6843/seg/F01_120_GT_01.tif"
    # img_path = "./MP6843/individual/convexity_0.00-0.20/142.png"
    # img_path = "./MP6843/individual/convexity_0.20-0.40/2962.png"
    img_path = "D:/Datasets/MP6843/individual/convexity_0.20-0.40/1337.png"
    # img_path = "./MP6843/individual/convexity_0.20-0.40/2584.png"
    # img_path = "./MP6843/individual/convexity_0.40-0.60/50.png"
    # img_path = "./MP6843/individual/convexity_0.60-0.80/205.png"
    # img_path = "./MP6843/individual/convexity_0.80-1.00/75.png"

    img = cv2.imread(img_path)
    gt = label_connected_component(img[:,:,0], background=0, connectivity=1)
    # gt = area_closing(gt)
    for p in regionprops(gt):
        if p.area < 100:
            gt[p.coords[:,0], p.coords[:,1]] = 0

    ###############################
    #### center computing test ####
    ###############################

    # centers, labels = get_center(gt, ctype='center_of_mass')
    # m = np.zeros(gt.shape)
    # for c in centers:
    #     m[c[0], c[1]] = 1
    # m = dilation(m, disk(2))

    # vis = np.copy(gt>0).astype(np.float32)
    # vis[m > 0] = 0
    # vis = np.stack([gt>0, vis, vis], axis=-1)
    # plt.imshow(vis)
    # plt.axis('off')
    # plt.show()

    #####################
    #### color wheel ####
    #####################
    
    # color_wheel = np.zeros((512,512,3))
    # color_wheel[:,:,1] = 0
    # color_wheel[:,:,2] = 255
    # rr, cc = circle(256, 256, 200)
    # angle = (np.arctan2(255-rr, 255-cc) + 3.141593)/(2*3.14159)*179
    # color_wheel[rr, cc, 0] = angle
    # color_wheel[rr, cc, 1] = 200
    # color_wheel[rr, cc, 2] = 200
    # color_wheel = cv2.cvtColor(color_wheel.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # plt.imshow(color_wheel)
    # plt.axis('off')
    # plt.show()

    ##########################
    #### vector flow test ####
    ##########################

    D = spath(gt)
    flow = dt2flow(D)

    # gt = np.pad(gt, ((1,1),(1,1)))
    # D, flow = diffuse(gt, mode='outwards', ctype='mass_center', niter=10000)

    # visA, visD = visualize_flow(flow, mask=gt)
    # plt.subplot(1,2,1)
    # plt.imshow(visA)
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(visD)
    # plt.axis('off')
    # plt.show()

    flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    plt.figure(figsize=(30,10),dpi=80)
    steps = [10, 50, 100, 200, 300]
    for idx, s in enumerate(steps):
        seg, cluster = seg_from_flow(flow, niter=s, step=0.5, mask=gt>0)
        plt.subplot(2,len(steps),idx+1)
        plt.imshow(vis(cluster, gt))
        plt.axis('off')
        plt.subplot(2,len(steps),len(steps)+idx+1)
        plt.imshow(vis(seg, gt))
        plt.axis('off')
    plt.show()

    ########################
    #### batch analysis ####
    ########################

    # niter = 50

    # ddir = 'D:/Datasets/MP6843/individual'
    # subs = ['convexity_0.00-0.20','convexity_0.20-0.40', 'convexity_0.40-0.60', 'convexity_0.60-0.80', 'convexity_0.80-1.00']
    # test = 'inwards' 
    # success = []
    # for subdir in subs:
    #     save_dir = os.path.join(ddir, subdir+'_'+test)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     success_sub = []
    #     for f in os.listdir(os.path.join(ddir, subdir)):
    #         gt = cv2.imread(os.path.join(ddir, subdir, f))[:,:,0]
    #         gt = np.pad(gt, ((1,1),(1,1)))
    #         D, flow = diffuse(gt, mode=test, ctype='edt_max', niter=500)
    #         seg, cluster = seg_from_flow(flow, niter=niter, step=0.5, mask=gt>0)
    #         count = 0
    #         for p in regionprops(seg):
    #             if p.area > 100:
    #                 count += 1
    #         if count == 1:
    #             print('Successful')
    #             success_sub.append(True)
    #         else:
    #             print('Fail')
    #             success_sub.append(False)
    #         v = vis(seg, gt)
    #         imsave(os.path.join(save_dir, f), v)
    #     success.append(success_sub)
    
    # for s in success:
    #     print(sum(s), len(s)-sum(s), sum(s)/len(s))
            





