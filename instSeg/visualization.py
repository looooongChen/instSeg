from skimage.color import label2rgb
from skimage.morphology import dilation, erosion
from skimage.color import rgb_colors as colors
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np
import random

default_colors = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 'violet', 'gold', 'olive', 'plum', 'firebrick')
# default_colors = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen', 'aqua', 'firebrick', 'gold', 'hotpink', 'greenyellow', 'mediumorchid', 'mediumpurple', 'olive', 'orange', 'peru', 'plum', 'tomato', 'violet', 'tan', 'teal')
# default_colors = ('red', 'blue', 'magenta', 'green', 'cyan', 'indigo')
# default_colors = list(colors.__dict__.keys())
# default_colors = [c for c in default_colors if not c.startswith('_')]

GT_COLOR = 'red'
PRED_COLOR = 'yellow'

def shape(img):
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.stack((img, img, img), axis=-1)
    img = img/img.max()*255
    return img

def get_color_by_name(color_name):
    color = getattr(colors, color_name)
    return np.array(color) * 255

def vis_semantic_area(img, semantic, colors=None, bg_label=0):
    img = shape(img)
    vis = label2rgb(semantic, colors=colors, bg_label=bg_label) * 255
    vis = img * 0.7 + vis * 0.3
    return vis.astype(np.uint8)

def vis_instance_area(img, instance, colors=None, bg_label=0):
    img = shape(img)
    instance = label(instance)
    vis = label2rgb(instance, colors=colors, bg_label=bg_label) * 255
    vis = img * 0.7 + vis * 0.3
    return vis.astype(np.uint8)

def vis_instance_contour(img, instance, colors=None):
    '''
    img: H x W
    instance: label map of shape H x W or stack of mask with shape N x H x W
    '''

    colors = default_colors if colors is None else colors

    img = shape(img)
    if len(instance.shape) == 2:
        # boundary = dilation(instance) != erosion(instance)
        # boundary = (instance * boundary).astype(np.uint16)
        # for r in regionprops(boundary):
        #     color_idx = r.label % len(colors)
        #     color = get_color_by_name(colors[color_idx])
        #     img[r.coords[:,0], r.coords[:,1],:] = color
        for idx, l in enumerate(np.unique(instance)):
            m = instance == l
            boundary = dilation(m) != erosion(m)
            color_idx = idx % len(colors)
            # color_idx = random.randint(0, len(colors)-1)
            color = get_color_by_name(colors[color_idx])
            idx_r, idx_c = np.nonzero(boundary)
            img[idx_r, idx_c,:] = color
    if len(instance.shape) == 3:
        for idx, m in enumerate(instance):
            boundary = dilation(m) != erosion(m)
            color_idx = idx % len(colors)
            # color_idx = random.randint(0, len(colors)-1)
            color = get_color_by_name(colors[color_idx])
            idx_r, idx_c = np.nonzero(boundary)
            img[idx_r, idx_c,:] = color
            

    return img.astype(np.uint8)

if __name__ == '__main__':
    from skimage.io import imread, imsave

    img = imread('./test/cell/gt/mcf-z-stacks-03212011_i01_s1_w14fc74585-6706-47ea-b84b-ed638d101ae8.png')
    # vis = vis_instance_area(img, img)
    # vis = vis_instance_area(img, img, color=['red'])
    # vis = vis_instance_contour(img, img)
    vis = vis_instance_contour(img, img, 'red')
    imsave('./vis.png', vis)


        
