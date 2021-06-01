from skimage.color import label2rgb
from skimage.morphology import dilation, erosion
from skimage.color import rgb_colors as colors
from skimage.measure import label
from skimage.measure import regionprops
import numpy as np

default_colors = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')

GT_COLOR = 'red'
PRED_COLOR = 'yellow'

def shape(img):
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.stack((img, img, img), axis=-1)
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

def vis_instance_contour(img, instance, color=None):
    img = shape(img)
    boundary = dilation(instance) != erosion(instance)
    boundary = (instance * boundary).astype(np.uint16)
    if color is None:
        for r in regionprops(boundary):
            color_idx = r.label % len(default_colors)
            color = get_color_by_name(default_colors[color_idx])
            img[r.coords[:,0], r.coords[:,1],:] = color
    else:
        color = get_color_by_name(color)
        rr, cc = np.nonzero(boundary)
        img[rr, cc, :] = color

    return img

if __name__ == '__main__':
    from skimage.io import imread, imsave

    img = imread('./test/cell/gt/mcf-z-stacks-03212011_i01_s1_w14fc74585-6706-47ea-b84b-ed638d101ae8.png')
    # vis = vis_instance_area(img, img)
    # vis = vis_instance_area(img, img, color=['red'])
    # vis = vis_instance_contour(img, img)
    vis = vis_instance_contour(img, img, 'red')
    imsave('./vis.png', vis)


        
