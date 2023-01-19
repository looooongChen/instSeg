import json
import cv2
import os
import numpy as np
from skimage.measure import regionprops, label
import copy
import sys
from tqdm import tqdm

# META = {'image_sz': [], 'patch_sz': [], 'overlap': [], 'patches': []}
META = {'image_sz': [], 'patches': []}
PATCH = {'path': None, 'data': None, 'position': [], 'size': []}


def split(image, patch_sz, overlap, remainder='drop', patch_in_ram=True, save_dir=None):

    '''
    image: np array of the image
    patch_sz: tuple (dim1, dim2, ...), patch size to crop
    overlap: tuple (dim1, dim2, ...), overlap long height and width direction
    remainder: 'drop', 'same', 'valid'
    patch_in_ram: keep patches in ram or not, if your ram is limited, consider to use this mode
        remenber to provide save_dir, otherwise, patches are not store in any places (ram/disk)
    save_dir: path to save patches on the disk
    '''
    img_sz = image.shape

    assert len(img_sz) >= len(patch_sz)
    assert np.all([img_sz[i] >= patch_sz[i] for i in range(len(patch_sz))])
    assert patch_in_ram == True or save_dir is not None
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    D = len(patch_sz)

    step = [patch_sz[i]-overlap[i] for i in range(D)]

    meta = copy.deepcopy(META)
    meta['image_sz'] = img_sz[0:D]
    patches = {}
    
    position, size = [], []
    for i in range(D):
        pos_ = list(range(0, img_sz[i]-patch_sz[i]+1, step[i]))
        sz_ = [patch_sz[i]] * len(pos_)
        if remainder != 'drop' and pos_[-1] + patch_sz[i] != img_sz[i]:
            if remainder == 'same':
                sz_.append(patch_sz[i])
                pos_.append(img_sz[i] - patch_sz[i])
            else:
                sz_.append(img_sz[i]-(pos_[-1] + step[i]))
                pos_.append(pos_[-1] + step[i])

        position.append(pos_)
        size.append(sz_)

    position = np.stack(np.meshgrid(*position), axis=-1)
    position = position.reshape((-1,D))
    size = np.stack(np.meshgrid(*size), axis=-1)
    size = size.reshape((-1,D))
    for i in tqdm(range(len(position)), ncols=100, desc='splitting: '):
        s = tuple([slice(position[i,j], position[i,j]+size[i,j]) for j in range(D)])
        patch_img = image[s]
        # print(patch.shape, position[i], patch[0,0,0], patch[0,0,0]//(500*64))
        patch = copy.deepcopy(PATCH)
        patch['position'] = [position[i,j] for j in range(D)]
        patch['size'] = [size[i,j] for j in range(D)]
        patch['data'] = patch_img if patch_in_ram else None
        if save_dir is not None:
            fname = [str(position[i,j]) for j in range(D)]
            fname = 'patch-' + '-'.join(fname)
            if D == 2:
                patch['path'] = os.path.join(save_dir, fname+'.png')
                cv2.imwrite(patch['path'], patch_img)
            else:
                patch['path'] = os.path.join(save_dir, fname+'.npy')
                np.save(patch['path'], patch_img)

        meta['patches'].append(patch)
        
    return meta


MATCH_THRES = 0.5
MAX_LABEL_PER_PATCH = 1000
MIN_OVERLAP = 10


def stitch(meta, channel, mode='average'):
    '''
    Args:
        meta: meta data
        channel: number of channels of the stitched image
        mode: 'raw', 'average', 'max', 'label'
    the 'label' mode only works properly when two patches overlap
    '''


    img_sz = (*meta['image_sz'], channel)
    image = np.squeeze(np.zeros(img_sz))

    if mode == 'average':
        norm = np.zeros(tuple(meta['image_sz']))
    if mode == 'label':
        image = image.astype(np.uint16)

    for idx in tqdm(range(len(meta['patches'])), ncols=100, desc='stitching: '):

        item = meta['patches'][idx]
        if 'data' in item.keys() and item['data'] is not None:
            patch = item['data']
        elif item['path'] is not None:
            if item['path'].endswith('npy'):
                patch = np.load(item['path'])
            else:
                patch = cv2.imread(item['path'], cv2.IMREAD_UNCHANGED)
        else:
            continue

        patch = np.squeeze(patch)
        D = len(item['position'])

        # print(item['position'], item['size'])
        s = tuple([slice(item['position'][i], item['position'][i]+item['size'][i]) for i in range(D)])

        if mode == 'raw':
            image[s] = image[s] + patch
        elif mode == 'max':
            image[s] = np.maximum(image[s], patch)
        elif mode == 'average':
            image[s] = image[s] + patch
            norm[s] = norm[s] + 1
        elif mode == 'label':
            patch = patch + (patch>0) * MAX_LABEL_PER_PATCH * idx
            roi= image[s]
            for p in regionprops(roi):
                labels, counts = np.unique(patch[p.coords[:,0], p.coords[:,1]], return_counts=True)
                i_max = np.argsort(counts)[-1]
                if labels[i_max] != 0 and counts[i_max]/p.area > MATCH_THRES:
                    patch[patch==labels[i_max]] = p.label
            image[s] = patch
            
    if mode == 'average':
        if norm.ndim != image.ndim:
            norm = np.expand_dims(norm, axis=-1)
        norm[norm==0] = 1
        image = image / norm
    if mode == 'label':
        image = label(image)

    return image




if __name__ == '__main__':
    # print(len(np.unique(img)))
    # meta = split2D(img, (100,100), (10, 10), patch_in_ram=False, save_dir='./test')
    # img = stitch2D(meta, channel=1, mode='label')
    # print(len(np.unique(img)))
    # # img  = 255*(img-img.min())/(img.max()-img.min())
    # cv2.imwrite('./st.png', img.astype(np.uint8))
    # print(meta)
    # image = np.array(range(0,512*500*64))
    # image = image.reshape((512,500,64))

    # image = cv2.imread('./test/land.jpg')
    image = cv2.imread('./test/cell/gt/mcf-z-stacks-03212011_i02_s1_w112a67c56-029e-4a7d-bb09-a3f0a95a94c7.png')[:,:,0]

    meta = split(image, (128,128), (20,20), remainder='valid', patch_in_ram=True, save_dir=None)
    image = stitch(meta, channel=1, mode='label')
    cv2.imwrite('./st.png', image.astype(np.uint8))
    print(image.shape)