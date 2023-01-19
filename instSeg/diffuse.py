
import numpy as np
import cv2
import scipy
from scipy.ndimage.measurements import label
from skimage.measure import label as label_connected_component

DISK = np.array([[0,1,0],[1,1,1],[0,1,0]])
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


def get_center(instance_mask, ctype='center_of_mass'):
    if ctype == 'center_of_mass':
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
        edt = cv2.distanceTransform((instance_mask>0).astype(np.uint8), cv2.DIST_L2, 3)
        for p in regionprops(instance_mask):
            labels.append(p.label)
            dst = edt[p.coords[:,0], p.coords[:,1]]
            idmin = np.argmax(dst)
            centers.append(p.coords[idmin])
        return np.array(centers), np.array(labels)


def spath_to_center(instance_mask, ctype='center_of_mass', niter=100):
    '''
    Args:
        label: (1) x H x W x (1)
        ctype: center type, 'center_of_mass', 'edt_max', ''
    '''
    sz_pad = (instance_mask.shape[0]+2, instance_mask.shape[1]+2)
    centers, labels = get_center(instance_mask, ctype)
    centers += 1
    # pad instance mask
    pim = np.zeros(sz_pad)
    pim[1:-1,1:-1] = instance_mask
    # distance map
    dt = np.zeros(sz_pad) + np.inf
    dt[centers[:,0], centers[:,1]] = 0
    # visit map
    visited = np.zeros(sz_pad, np.bool)
    visited[pim==0] = True
    visited[centers[:,0], centers[:,1]] = True
    # compute the shortest path to centers
    iter = 0
    while not np.all(visited) and iter<niter:
        idr, idc = np.nonzero(np.logical_not(visited))
        n_idr = np.array([idr, idr+1, idr+1, idr+1, idr, idr-1, idr-1, idr-1])
        n_idc = np.array([idc+1, idc+1, idc, idc-1, idc-1, idc-1, idc, idc+1])
        offset = np.expand_dims(np.array([1,D_DIG,1,D_DIG,1,D_DIG,1,D_DIG]), axis=-1)
        # get neighbor distance:
        D_neighbor = dt[n_idr, n_idc] + offset
        D_neighbor[pim[n_idr, n_idc] != np.expand_dims(pim[idr, idc], 0)] = np.inf
        # D_neighbor[visited[n_idr, n_idc] == True] = np.inf
        idx = np.argmin(D_neighbor, axis=0)
        D_neighbor = D_neighbor[idx, range(len(idx))]
        update = D_neighbor != np.inf
        # update distance and coordinated
        D_neighbor = D_neighbor[update]
        idr, idc = idr[update], idc[update]
        dt[idr, idc] = D_neighbor
        visited[idr, idc] = True
        iter += 1

    return dt




if __name__ == '__main__':
    from utils import *
    import numpy as np 
    from skimage.io import imread
    from skimage.draw import circle 
    from skimage.morphology import dilation, disk, erosion, square, skeletonize
    from skimage.measure import regionprops
    import time
    import matplotlib.pyplot as plt
    import time

    img_path = "./MP6843/seg/F01_120_GT_01.tif"
    # img_path = "D:/Datasets/MP6843/individual/convexity_0.60-0.80/70.png"

    img = cv2.imread(img_path)
    seg = label_connected_component(img[:,:,0], background=0, connectivity=1)
    for p in regionprops(seg):
        if p.area < 100:
            seg[p.coords[:,0], p.coords[:,1]] = 0

    ###############
    #### ridge ####
    ###############

    # from skimage.features import hessian_matrix, hessian_matrix_eigvals
    # def detect_ridges(gray, sigma=3.0):
    #     hxx, hyy, hxy = hessian_matrix(gray, sigma)
    #     i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    #     return i1, i2

    
    ###############################
    #### center computing test ####
    ###############################

    # centers, labels = get_center(seg, ctype='edt_max')
    # m = np.zeros(seg.shape)
    # for c in centers:
    #     m[c[0], c[1]] = 1
    # m = dilation(m, disk(2))

    # vis = np.copy(seg>0).astype(np.float32)
    # vis[m > 0] = 0
    # vis = np.stack([seg>0, vis, vis], axis=-1)
    # plt.imshow(vis)
    # plt.show()

    ########################
    #### diffusion test ####
    ########################

    t = time.time()
    dt = diffuse_from_center(seg, ctype='center_of_mass')
    print(time.time()-t)
    plt.imshow(dt)
    plt.show()

    # img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)

    # flow = edt_flow(img)
    

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

    # img = np.squeeze(img)
    # plt.subplot(1,2,1)
    # plt.imshow(img>0)

    # mask = np.zeros_like(img)
    # pts_track, _ = track_flow(flow, iters=40, mask=img>0)
    # mask[pts_track[:,0], pts_track[:,1]] = 1
    # mask = seg_from_flow(flow, iters=10, mask=img>0)

    # plt.subplot(1,2,2)
    # plt.imshow(mask)
    # plt.show()
