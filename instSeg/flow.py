from numpy.core.fromnumeric import argmin
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, opening, dilation, square, disk
from skimage.morphology import skeletonize
# from skimage.morphology import medial_axis
from skimage.measure import regionprops
from skimage.measure import label as label_connected_component
from scipy.spatial import distance_matrix
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import numpy as np
import cv2
import scipy

DISK = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
SQUARE = np.array([[1,1,1],[1,1,1],[1,1,1]], np.uint8)
D_DIG = 2 ** 0.5

def boundary(instance_mask):
    '''
    Args:
        instance_mask: (1) x H x W x (1)
    Return:
        boundary: H x W
    '''
    instance_mask = np.squeeze(instance_mask).astype(np.uint16)
    l_dil = cv2.dilate(instance_mask, DISK, iterations = 1)
    l_ero = cv2.erode(instance_mask, DISK, iterations = 1)
    boundary = (l_dil != l_ero) * instance_mask
    return boundary


def get_center(instance_mask, ctype='median'):
    '''
    Args:
        instance_mask: (1) x H x W x (1)
    Return:
        centers: N x 2
        labels: N
    '''
    instance_mask = np.squeeze(instance_mask.astype(np.uint16))
    if ctype == 'mass':
        labels = np.unique(instance_mask[instance_mask!=0])
        centers = np.round(np.array(scipy.ndimage.center_of_mass(instance_mask>0, labels=instance_mask, index=[int(l) for l in labels]))).astype(int)
    if ctype == 'median':
        centers, labels = [], []
        for r in regionprops(instance_mask):
            labels.append(r.label)
            centers.append(np.median(r.coords, axis=0))
        labels = np.array(labels)
        centers = np.round(np.array(centers)).astype(int)
    if ctype == 'edt':
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
    
    # for 'median' and 'mass', check if center in the object or not
    not_in = instance_mask[centers[:,0], centers[:,1]] != labels
    for idx in np.nonzero(not_in)[0]:
        yy, xx = np.nonzero(instance_mask == labels[idx])
        idmin = np.argmin((yy-centers[idx,0])**2 + (xx-centers[idx,1])**2)
        centers[idx,0], centers[idx,1] = yy[idmin], xx[idmin]

    return centers, labels

def spath(instance_mask, mode='boundary'):
    if mode == 'boundary':
        B = boundary(instance_mask)
        D = np.zeros(instance_mask.shape)
        coordB = np.array(np.nonzero(B)).T
        coordF = np.array(np.nonzero(instance_mask)).T
        dm = distance_matrix(coordF, coordB, p=2, threshold=10000)
        D[coordF[:,0], coordF[:,1]] = np.amin(dm, axis=1)
        return D


def flow_from_diffuse(instance_mask, mode='outwards'):
    '''
    shortest distance to center, path through background is not allowed
    Args:
        label: (1) x H x W x (1)
        mode: 'median', 'edt', 'skeleton'
    '''
    assert mode in ['median', 'edt', 'skeleton']

    instance_mask = np.copy(instance_mask).astype(np.uint16)
    flow = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 2))

    for r in regionprops(instance_mask):
        patch = (instance_mask[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] == r.label).astype(np.uint8)
        patch_pad = np.zeros((patch.shape[0]+4, patch.shape[1]+4), np.uint8)
        patch_pad[2:-2,2:-2] = patch
        patch_pad = cv2.dilate(patch_pad, SQUARE, iterations = 1)
        D = np.zeros(patch_pad.shape)

        heating = np.zeros(patch_pad.shape)
        if mode == 'skeleton':
            # skel, distance = medial_axis(patch_pad, return_distance=True)
            skel = skeletonize(patch_pad, method='lee') > 0
            niter_max = max(r.bbox[2]-r.bbox[0]+2, r.bbox[3]-r.bbox[1]+2)
            # niter_max = 2 * (r.bbox[2]+r.bbox[3]-r.bbox[0]-r.bbox[1]+4)
        else:
            centers, _ = get_center(patch_pad, mode)
            h_rr, h_cc = centers[:,0], centers[:,1] 
            niter_max = 2 * (r.bbox[2]+r.bbox[3]-r.bbox[0]-r.bbox[1]+4)

        # perform diffusion
        idr, idc = np.nonzero(patch_pad)
        idr_, idc_ = np.nonzero(patch_pad==0)
        n_idr = np.array([idr, idr, idr+1, idr+1, idr+1, idr, idr-1, idr-1, idr-1])
        n_idc = np.array([idc, idc+1, idc+1, idc, idc-1, idc-1, idc-1, idc, idc+1])

        for niter in range(niter_max):
            # heating / keep heating
            if mode == 'skeleton':
                D[skel] = niter + 1
            else:
                D[h_rr, h_cc] += 1
            # diffuse
            D_neighbor = D[n_idr, n_idc]
            S = np.mean(D_neighbor, axis=0)
            D[idr, idc] = S
                
            D[idr_, idc_] = 0
            
        flow_y, flow_x = (D[3:-1,2:-2]-D[2:-2,2:-2]) * patch, (D[2:-2,3:-1]-D[2:-2,2:-2]) * patch
        flow[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3], :] = flow[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3], :] + np.stack((flow_y, flow_x), axis=-1)        

    flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)

    return flow


def offset(instance_mask, ctype='median'):

    instance_mask = np.squeeze(instance_mask.astype(np.uint16))

    offset = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 2))

    for r in regionprops(instance_mask):
        center = np.median(r.coords, axis=0)
        if instance_mask[int(center[0]), int(center[1])] != r.label:
            idmin = np.argmin((r.coords[:,0]-center[0])**2 + (r.coords[:,1]-center[1])**2)
            center = r.coords[idmin]
        offset[r.coords[:,0], r.coords[:,1], 0] = center[0] - r.coords[:,0] 
        offset[r.coords[:,0], r.coords[:,1], 1] = center[1] - r.coords[:,1]

    return offset 
        
# def flow_from_edt(instance_mask):
#     '''
#     shortest distance to center, path through background is not allowed
#     Args:
#         label: (1) x H x W x (1)
#         ctype: center type, 'mass', 'edt', 'median'
#     '''

#     '''
#     Args:
#         sdt: (1) x H x W x (1) 
#         sigma: smooth vector fild
#         normalize: normalize the flow amplitude or not
#     Return:
#         flow: flow: H x W x 2 
#     '''
#     l = np.squeeze(instance_mask).astype(np.uint16)
#     l_dil = cv2.dilate(l, np.ones((3,3), np.uint16), iterations = 1)
#     l_ero = cv2.erode(l, np.ones((3,3), np.uint16), iterations = 1)
#     dt_area = np.logical_and(l_dil == l_ero, l>0)
#     dt = cv2.distanceTransform(dt_area.astype(np.uint8), cv2.DIST_L2, 3)

#     # sdt = np.log(1+sdt)
#     dt = gaussian_filter(dt, [1,1])
#     flow = np.stack((cv2.Sobel(dt,cv2.CV_64F,0,1,ksize=3), cv2.Sobel(dt,cv2.CV_64F,1,0,ksize=3)), axis=-1)

#     flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
#     return flow


def flow(instance_mask, mode='outwards', epoch=2, process_disp=False):
    '''
    shortest distance to center, path through background is not allowed
    Args:
        label: B x H x W x 1
        mode: 'inwards' or 'outwards'
    Return:
        flows: B x H x W x 2
    '''
    instance_mask = np.squeeze(instance_mask, axis=-1).astype(np.uint16)
    flows = []
    if process_disp:
        instance_mask = tqdm(instance_mask, desc='Vector flow:', ncols=100)
    for l in instance_mask:
        if mode == 'offset':
            f = offset(l)
        else:
            f = diffuse(l, mode=mode, epoch=epoch)
        flows.append(f)
    return np.array(flows)


def visualize_flow(flow, mask=None):
    '''
    Args:
        flow: H x W x 2
    '''
    if mask is None:
        mask = (flow[:,:,0]**2 + flow[:,:,1]**2) > 1e-5
    angle = (np.arctan2(flow[:,:,0], flow[:,:,1]) + 3.141593)/(2*3.14159)*179
    visD = np.stack([angle, 200*np.ones_like(angle), 200*np.ones_like(angle)], axis=-1)
    visD = cv2.cvtColor(visD.astype(np.uint8), cv2.COLOR_HSV2RGB)
    bg_y, bg_x = np.nonzero(mask == 0)
    visD[bg_y, bg_x, :] = 0
    visA = np.sum(flow**2, axis=-1) ** 0.5

    # color_wheel = np.zeros((512,512,3))
    # rr, cc = circle(256, 256, 200)
    # angle = (np.arctan2(255-rr, 255-cc) + 3.141593)/(2*3.14159)*179
    # color_wheel[rr, cc, 0] = angle
    # color_wheel[rr, cc, 1] = 200
    # color_wheel[rr, cc, 2] = 200
    # color_wheel = cv2.cvtColor(color_wheel.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return visD.astype(np.uint8), visA


def track_flow(flow, niter=100, step=0.5, stop=float('inf'), mask=None):
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

    # flow[:,:,0] = gaussian_filter(flow[:,:,0], sigma=1)
    # flow[:,:,1] = gaussian_filter(flow[:,:,1], sigma=1)


    shape = flow.shape

    pts = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), axis=-1)
    pts = np.reshape(pts, (-1, 2))
    if mask is not None:
        idx = np.reshape(mask, (-1)) > 0
        pts = pts[idx,:]
    pts_track = pts.copy()

    D_init = flow[pts[:,0], pts[:,1]]
    update = np.ones(D_init.shape[0], np.bool)

    # tracking
    pts_track = pts_track + D_init * step 
    pts_track = np.maximum(pts_track, 0)
    pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
    pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)

    for idx in range(niter):
        # pts_track_floor = np.floor(pts_track).astype(np.uint32)
        # pts_track_ceil = np.ceil(pts_track).astype(np.uint32)
        # D_tl = flow[pts_track_floor[:,0], pts_track_floor[:,1]]
        # D_tr = flow[pts_track_floor[:,0], pts_track_ceil[:,1]]
        # D_bl = flow[pts_track_ceil[:,0], pts_track_floor[:,1]]
        # D_br = flow[pts_track_ceil[:,0], pts_track_ceil[:,1]]

        # W_floor, W_ceil = pts_track_ceil - pts_track + 1e-7, pts_track - pts_track_floor + 1e-7
        # W_tl = np.expand_dims(W_floor[:,0] * W_floor[:,1], axis=-1)
        # W_tr = np.expand_dims(W_floor[:,0] * W_ceil[:,1], axis=-1)
        # W_bl = np.expand_dims(W_ceil[:,0] * W_floor[:,1], axis=-1)
        # W_br = np.expand_dims(W_ceil[:,0] * W_ceil[:,1], axis=-1)

        # D = (D_tl * W_tl + D_tr * W_tr + D_bl * W_bl + D_br * W_br) / (W_tl + W_tr + W_bl + W_br)

        pts_track_int = np.floor(pts_track).astype(np.uint32)
        D = flow[pts_track_int[:,0], pts_track_int[:,1]]
        
        if stop < 1:
            update = np.logical_and(np.sum(D_init * D, axis=-1) > stop, update)
            pts_track = pts_track + D * step * np.expand_dims(update, axis=-1)
        else:
            pts_track = pts_track + D * step 
   
        pts_track = np.maximum(pts_track, 0)
        pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
        pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)

    pts_track = np.round(pts_track).astype(np.int32)

    return pts_track, pts

def track_offset(offset, mask=None):
    if offset.ndim == 4:
        offset = np.squeeze(offset, axis=0)

    offset = np.copy(offset)

    offset[:,:,0] = gaussian_filter(offset[:,:,0], sigma=1)
    offset[:,:,1] = gaussian_filter(offset[:,:,1], sigma=1)

    shape = offset.shape

    pts = np.stack(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij'), axis=-1)
    pts = np.reshape(pts, (-1, 2))
    if mask is not None:
        idx = np.reshape(mask, (-1)) > 0
        pts = pts[idx,:]
    pts_track = pts + offset[pts[:,0], pts[:,1]]
    pts_track[:,0] = np.minimum(pts_track[:,0], shape[0]-1)
    pts_track[:,1] = np.minimum(pts_track[:,1], shape[1]-1)

    pts_track = np.round(pts_track).astype(np.int32)

    return pts_track, pts


def seg_from_flow(flow, mode=None, niter=30, step=0.5, mask=None, stop=0.5):
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

    mask = mask>0 if mask is not None else np.ones(seg.shape, np.bool)
    seg = np.zeros((flow.shape[0], flow.shape[1]))
    clusters = np.zeros((flow.shape[0], flow.shape[1])).astype(np.uint8)
    if mode != 'offset':
        pts_track, pts = track_flow(flow, niter, stop=stop, step=step, mask=mask)
    else:
        pts_track, pts = track_offset(flow, mask=mask)

    # idx = np.sqrt((pts_track[:,0] - pts[:, 0])**2 + (pts_track[:,1] - pts[:, 1])**2) > 2
    # pts_track, pts = pts_track[idx], pts[idx]
    clusters[pts_track[:,0], pts_track[:,1]] = 1
    clusters = clusters * mask

    clusters = label_connected_component(clusters)
    seg[pts[:,0], pts[:,1]] = clusters[pts_track[:,0], pts_track[:,1]]
    # clusters_ = dilation(clusters, square(3))
    # clusters_ = label_connected_component(clusters_)
    # seg[pts[:,0], pts[:,1]] = clusters_[pts_track[:,0], pts_track[:,1]]

    seg = seg.astype(np.int32)

    # for r in regionprops(seg):
    #     if r.area < 50:
    #         seg[r.coords[:,0], r.coords[:,1]] = 0

    while True:
        seg_D = dilation(seg, square(3))
        seg_add = seg_D * mask * (seg == 0)
        if np.sum(seg_add) == 0:
            break
        seg = seg + seg_add

    return seg, clusters



    

if __name__ == '__main__':
    from utils import *
    import numpy as np 
    from skimage.io import imread, imsave
    from skimage.color import label2rgb
    from skimage.draw import circle 
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

    # img_path = "D:/Datasets/MP6843/seg/F01_120_GT_01.tif"
    # img_path = "./MP6843/individual/convexity_0.00-0.20/142.png"
    # img_path = "./MP6843/individual/convexity_0.20-0.40/2962.png"
    # img_path = "D:/Datasets/MP6843/individual/convexity_0.20-0.40/1337.png"
    # img_path = "./MP6843/individual/convexity_0.20-0.40/2584.png"
    img_path = "D:/Datasets/MP6843/individual/convexity_0.40-0.60/50.png"
    # img_path = "./MP6843/individual/convexity_0.60-0.80/205.png"
    # img_path = "./MP6843/individual/convexity_0.80-1.00/75.png"

    # gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,0]
    gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(gt.shape)
    gt = erosion(gt, square(3))
    gt = label_connected_component(gt, background=0, connectivity=1)
    for p in regionprops(gt):
        if p.area < 100:
            gt[p.coords[:,0], p.coords[:,1]] = 0

    ###############################
    #### center computing test ####
    ###############################

    # for idx, ctype in enumerate(['mass', 'median', 'edt']):
    #     centers, _ = get_center(gt, ctype=ctype)
    #     m = np.zeros(gt.shape)
    #     m[centers[:,0], centers[:,1]] = 1
    #     m = dilation(m, disk(2))

    #     vis = np.copy(gt>0).astype(np.float32)
    #     vis[m > 0] = 0
    #     vis = np.stack([gt>0, vis, vis], axis=-1)
    #     plt.subplot(1,3,idx+1)
    #     plt.imshow(vis)
    #     plt.title(ctype)
    #     plt.axis('off')
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

    # D = spath(gt)
    # flow = dt2flow(D)

    # gt = np.pad(gt, ((1,1),(1,1)))
    D, flow = diffuse(gt, mode='inwards', ctype='median')

    visA, visD = visualize_flow(flow, mask=gt)
    plt.subplot(1,2,1)
    plt.imshow(visA)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(visD)
    plt.axis('off')
    plt.show()

    # flow = flow / (np.linalg.norm(flow, ord=2, axis=-1, keepdims=True)+1e-7)
    # plt.figure(figsize=(30,10),dpi=80)
    # steps = [10, 50, 100, 200, 300]
    # for idx, s in enumerate(steps):
    #     seg, cluster = seg_from_flow(flow, niter=s, step=0.5, mask=gt>0)
    #     plt.subplot(2,len(steps),idx+1)
    #     plt.imshow(vis(cluster, gt))
    #     plt.axis('off')
    #     plt.subplot(2,len(steps),len(steps)+idx+1)
    #     plt.imshow(vis(seg, gt))
    #     plt.axis('off')
    # plt.show()

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
            





