import json
import cv2
import os
import numpy as np
from skimage.measure import regionprops, label
import copy
import sys

# META = {'image_sz': [], 'patch_sz': [], 'overlap': [], 'patches': []}
META = {'image_sz': [], 'patches': {}}
PATCH_META = {'path': None, 'data': None, 'position': [], 'size': []}


def split2D(image, patch_sz, overlap, patch_in_ram=True, save_dir=None):

    '''
    image: np array of the image
    patch_sz: tuple (height, width), patch size to crop
    overlap: tuple (y, x), overlap long height and width direction
    patch_in_ram: keep patches in ram or not, if your ram is limited, consider to use this mode
        remenber to provide save_dir, otherwise, patches are not store in any places (ram/disk)
    save_dir: path to save patches on the disk
    '''
    assert patch_in_ram == True or save_dir is not None

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_sz = image.shape
    step = [patch_sz[0]-overlap[0], patch_sz[1]-overlap[1]]

    assert len(patch_sz) == 2
    assert len(overlap) == 2
    assert img_sz[0] >= patch_sz[0] and img_sz[1] >= patch_sz[1]

    meta = copy.deepcopy(META)
    meta['image_sz'] = img_sz[0:2]
    # meta['patch_sz'] = list(patch_sz)
    # meta['overlap'] = list(overlap)
    patches = {}
    
    idx0, idx1 = 0, 0 

    print('Split progress: ',  end="")
    sys.stdout.flush()
    while idx0 * step[0] < img_sz[0]:
        print('=', end="")
        sys.stdout.flush()
        if idx0 * step[0] + patch_sz[0] > img_sz[0]:
            Hmin, Hmax = img_sz[0] - patch_sz[0], img_sz[0]
        else:
            Hmin, Hmax = idx0 * step[0], idx0 * step[0] + patch_sz[0]

        idx1 = 0
        while idx1 * step[1] < img_sz[1]:
            if idx1 * step[1] + patch_sz[1] > img_sz[1]:
                Wmin, Wmax = img_sz[1] - patch_sz[1], img_sz[1]
            else:
                Wmin, Wmax = idx1 * step[1], idx1 * step[1] + patch_sz[1]
            fname = 'patch-{:03d}-{:03d}'.format(idx0, idx1)
            patch_meta = copy.deepcopy(PATCH_META)
            patch_meta['position'] = [Hmin, Wmin]
            patch_meta['size'] = [Hmax-Hmin, Wmax-Wmin]
            patch_meta['data'] = image[Hmin:Hmax, Wmin:Wmax] if patch_in_ram else None
            if save_dir is not None:
                patch_meta['path'] = os.path.join(save_dir, fname+'.png')
                cv2.imwrite(patch_meta['path'], image[Hmin:Hmax, Wmin:Wmax])
            meta['patches'][fname] = patch_meta
            

            if Wmax == img_sz[1]:
                break
            else:
                idx1 += 1

        if Hmax == img_sz[0]:
            break
        else:
            idx0 += 1
    print(' <Done!>')

    if save_dir is not None:
        meta_save = copy.deepcopy(META)
        meta_save['image_sz'] = meta['image_sz']
        for k, patch in meta['patches'].items():
            patch_save = copy.deepcopy(patch)
            del patch_save['data']
            patch_save['path'] = os.path.basename(patch['path'])
            meta_save['patches'][k] = patch_save
        with open(os.path.join(save_dir, 'meta.json'), 'w') as outfile:
            json.dump(meta_save, outfile, indent=2)

    return meta

MATCH_THRES = 0.5
MAX_LABEL_PER_PATCH = 1000
MIN_OVERLAP = 10


def stitch2D(meta, channel, mode='average'):
    '''
    Args:
        meta: meta data
        patches: dict of patches
        mode: 'raw', 'average', 'max', 'label'
    the 'label' mode only works properly when two patches overlap
    '''

    print('Stitch progress: ',  end="")
    sys.stdout.flush()

    img_sz = (meta['image_sz'][0], meta['image_sz'][1], channel)
    image = np.squeeze(np.zeros(img_sz))

    step = int(np.sqrt(len(meta['patches'])))

    if mode == 'average':
        norm = np.zeros(meta['image_sz'][0:2])

    for idx, item in enumerate(list(meta['patches'].values())):
        if idx//step != (idx+1)//step:
            print('=', end="")
        sys.stdout.flush()

        if 'data' in item.keys() and item['data'] is not None:
            patch = item['data']
        else:
            patch = cv2.imread(item['path'], cv2.IMREAD_UNCHANGED)
            if patch is None:
                continue

        patch = np.squeeze(patch)
        patch = cv2.resize(patch, (item['size'][1], item['size'][0]))

        Hmin, Wmin = item['position'][0], item['position'][1]
        Hmax, Wmax = Hmin + item['size'][0], Wmin + item['size'][1]
        if mode == 'raw':
            image[Hmin:Hmax, Wmin:Wmax] = image[Hmin:Hmax, Wmin:Wmax] + patch
        elif mode == 'max':
            image[Hmin:Hmax, Wmin:Wmax] = np.maximum(image[Hmin:Hmax, Wmin:Wmax], patch)
        elif mode == 'average':
            image[Hmin:Hmax, Wmin:Wmax] = image[Hmin:Hmax, Wmin:Wmax] + patch
            norm[Hmin:Hmax, Wmin:Wmax] = norm[Hmin:Hmax, Wmin:Wmax] + np.ones(item['size'])
    

    if mode == 'label':
        image = image.astype(np.uint16)
        for idx, item in enumerate(list(meta['patches'].values())):
            if idx//step != (idx+1)//step:
                print('=', end="")
            sys.stdout.flush()
            if 'data' in item.keys() and item['data'] is not None:
                patch = item['data']
            else:
                patch = cv2.imread(item['path'], cv2.IMREAD_UNCHANGED)
                if patch is None:
                    continue

            patch = np.squeeze(patch)
            patch = cv2.resize(patch, (item['size'][1], item['size'][0]), cv2.INTER_NEAREST)

            Hmin, Wmin = item['position'][0], item['position'][1]
            Hmax, Wmax = Hmin + item['size'][0], Wmin + item['size'][1]
            patch = patch + (patch>0) * MAX_LABEL_PER_PATCH * idx

            roi= image[Hmin:Hmax, Wmin:Wmax]
            for p in regionprops(roi):
                labels, counts = np.unique(patch[p.coords[:,0], p.coords[:,1]], return_counts=True)
                idx = np.argsort(counts)[-1]
                if labels[idx] != 0 and counts[idx]/p.area > MATCH_THRES:
                    patch[patch==labels[idx]] = p.label
            image[Hmin:Hmax, Wmin:Wmax] = patch
            # image[Hmin:Hmax, Wmin:Wmax] = np.maximum(image[Hmin:Hmax, Wmin:Wmax], patch)
            
            # patch_pad = patch.copy()
            # HHmin, HHmax = Hmin, Hmax
            # WWmin, WWmax = Wmin, Wmax
            # if meta['overlap'][0] < MIN_OVERLAP:
            #     deltaH = MIN_OVERLAP - meta['overlap'][0]
            #     HHmin, HHmax = max(0, Hmin-deltaH), min(Hmax+deltaH, meta['image_sz'][0])
            #     deltaHmin, deltaHmax = Hmin - HHmin, HHmax - Hmax
            #     patch_pad = np.pad(patch_pad, ((deltaHmin, deltaHmax),(0,0)), 'edge')
            # if meta['overlap'][1] < MIN_OVERLAP:
            #     deltaW = MIN_OVERLAP - meta['overlap'][1]
            #     WWmin, WWmax = max(0, Wmin-deltaW), min(Wmax+deltaW, meta['image_sz'][1])
            #     deltaWmin, deltaWmax = Wmin - WWmin, WWmax - Wmax
            #     patch_pad = np.pad(patch_pad, ((0,0),(deltaWmin, deltaWmax)), 'edge')

            # roi= image[HHmin:HHmax, WWmin:WWmax]
            # for p in regionprops(roi):
            #     labels, counts = np.unique(patch_pad[p.coords[:,0], p.coords[:,1]], return_counts=True)
            #     idx = np.argsort(counts)[-1]
            #     if labels[idx] != 0 and counts[idx]/p.area > MATCH_THRES:
            #         patch[patch==labels[idx]] = p.label
            # image[Hmin:Hmax, Wmin:Wmax] = patch
    
    if mode == 'average':
        if norm.ndim != image.ndim:
            norm = np.expand_dims(norm, axis=-1)
        norm[norm==0] = 1
        image = image / norm
    if mode == 'label':
        image = label(image)
        
    print(' <Done!>')
    return image




if __name__ == '__main__':
    img = cv2.imread('./test/cell/gt/mcf-z-stacks-03212011_i02_s1_w112a67c56-029e-4a7d-bb09-a3f0a95a94c7.png')[:,:,0]
    print(len(np.unique(img)))
    meta = split2D(img, (100,100), (0, 0), patch_in_ram=True, save_dir=None)
    img = stitch2D(meta, channel=1, mode='label')
    print(len(np.unique(img)))
    # img  = 255*(img-img.min())/(img.max()-img.min())
    cv2.imwrite('./st.png', img.astype(np.uint8))
    # print(meta)