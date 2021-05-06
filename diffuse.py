
import numpy as np
import cv2

def boundary(label):
    '''
    Args:
        label: (1) x H x W x (1)
    Return:
        boundary: H x W
    '''
    label = np.squeeze(label).astype(np.uint16)
    l_dil = cv2.dilate(label, disk_np(1), iterations = 1)
    l_ero = cv2.erode(label, disk_np(1), iterations = 1)
    B = (l_dil != l_ero) * label

    return B



if __name__ == '__main__':
    from utils import *
    import numpy as np 
    from skimage.io import imread
    from skimage.draw import circle 
    import time
    import matplotlib.pyplot as plt

    img = imread('I:/instSeg/ds_cyst/export-mask-2020-Dec-07-18-47-41/mask_00000001.png')
    b = boundary(img)
    plt.imshow(b)
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
