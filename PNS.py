import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import copy
import shutil
from PIL import Image
from skimage.measure import label as relabel

palette = [255, 255, 255, 231, 10, 156, 231, 60, 26, 231, 129, 69, 231, 217, 30, 231, 30, 74, 231, 79, 200, 231, 148, 243, 231, 237, 204, 231, 49, 247, 68, 164, 255, 67, 252, 216, 67, 65, 3, 68, 115, 129, 68, 183, 172, 68, 116, 24, 68, 185, 68, 68, 234, 193, 68, 47, 237, 68, 135, 198, 73, 28, 217, 73, 97, 4, 73, 185, 221, 73, 254, 9, 73, 47, 134, 73, 116, 178, 73, 48, 30, 73, 117, 73, 73, 167, 199, 73, 236, 243, 72, 134, 200, 72, 184, 70, 72, 252, 113, 72, 85, 74, 72, 154, 118, 72, 203, 243, 72, 16, 31, 73, 204, 139, 73, 17, 182, 73, 67, 52, 71, 221, 9, 71, 34, 53, 71, 83, 179, 71, 152, 222, 71, 241, 183, 71, 54, 226, 71, 103, 96, 71, 172, 140, 72, 104, 248, 72, 173, 35, 71, 33, 157, 71, 121, 118, 71, 190, 162, 71, 239, 32, 71, 52, 75, 70, 141, 36, 70, 209, 79, 70, 3, 205, 70, 72, 249, 71, 4, 101, 76, 153, 120, 76, 221, 163, 76, 54, 124, 76, 123, 167, 76, 172, 37, 76, 241, 81, 76, 73, 42, 76, 142, 85, 76, 191, 211, 76, 4, 254, 75, 3, 103, 75, 52, 228, 75, 121, 16, 75, 210, 233, 75, 23, 20, 75, 72, 146, 75, 141, 189, 75, 229, 150, 75, 42, 194, 75, 91, 64, 74, 90, 168, 74, 159, 212, 74, 208, 81, 74, 21, 125, 74, 110, 86, 74, 178, 129, 74, 228, 255, 74, 41, 42, 74, 129, 3, 74, 198, 47, 74, 158, 60, 74, 246, 21, 74, 59, 64, 74, 108, 190, 74, 177, 234, 73, 9, 195, 73, 78, 238, 74, 128, 108, 73, 197, 151, 73, 29, 112, 126, 126, 200, 126, 175, 70, 126, 244, 113, 126, 77, 74, 126, 145, 118, 126, 195, 243, 126, 8, 31, 126, 96, 248, 126, 165, 35, 126, 214, 161, 125, 213, 10, 125, 26, 53, 126, 75, 179, 126, 144, 222, 125, 233, 183, 125, 45, 227, 125, 95, 96, 125, 164, 140, 125, 252, 101, 125, 65, 144, 125, 25, 157, 125, 113, 118, 125, 182, 162, 125, 231, 32, 125, 44, 75, 125, 132, 36, 125, 201, 79, 125, 251, 205, 125, 64, 249, 125, 152, 210, 130, 144, 120, 130, 213, 163, 130, 46, 124, 130, 114, 167, 130, 164, 37, 130, 233, 81, 130, 65, 42, 130, 134, 85, 130, 183, 211, 130, 252, 254, 129, 251, 103, 129, 44, 229, 129, 113, 16, 129, 201, 233, 129, 14, 20, 129, 64, 146, 129, 133, 190, 129, 221, 151, 129, 34, 194, 129, 83, 64, 129, 82, 168, 128, 151, 212, 129, 200, 81, 129, 13, 125, 128, 101, 86, 128, 170, 129, 128, 220, 255, 128, 33, 42, 128, 121, 3, 128, 190, 47, 129, 250, 207, 128, 238, 21, 128, 51, 65, 128, 100, 190, 128, 169, 234, 128, 1, 195, 128, 70, 238, 128, 120, 108, 128, 188, 151, 128, 21, 112, 128, 81, 17, 128, 149, 60, 133, 170, 27, 133, 239, 70, 133, 33, 196, 133, 102, 239, 133, 190, 200, 133, 3, 244, 133, 52, 114, 133, 121, 157, 133, 220, 153, 133, 13, 22, 133, 82, 66, 132, 70, 136, 132, 139, 179, 132, 189, 49, 132, 1, 92, 132, 90, 53, 132, 159, 97, 132, 208, 223, 132, 51, 218, 132, 120, 6, 132, 169, 131, 132, 238, 175, 131, 226, 245, 131, 39, 32, 132, 89, 158, 131, 157, 201, 131, 246, 162, 131, 59, 206, 192, 223, 252, 192, 36, 40, 192, 125, 1, 192, 193, 44, 192, 243, 170, 192, 56, 213, 192, 144, 174, 192, 213, 218, 192, 6, 87, 192, 75, 131, 197, 107, 132, 197, 156, 2, 197, 225, 45, 197, 57, 6, 197, 126, 50, 197, 175, 175, 197, 244, 219, 197, 77, 180, 197, 146, 223, 197, 195, 93, 197, 194, 197, 197, 6, 241, 197, 56, 111, 197, 125, 154, 197, 213, 115, 197, 26, 158, 197, 75, 28, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

def read_pns_cyst(ds_dir, split='all', sz=None, num_read=None, keys=None):
    imgs, masks = {}, {}

    img_dir = os.path.join(ds_dir, 'PNS-Cyst-Part1-Images') 
    img_dir = img_dir if os.path.exists(img_dir) else os.path.join(ds_dir, 'PNS-Cyst-Part2-Images')
    gt_dir = os.path.join(ds_dir, 'PNS-Cyst-Part1-Segmentations')
    gt_dir = gt_dir if os.path.exists(gt_dir) else os.path.join(ds_dir, 'PNS-Cyst-Part2-Segmentations')

    count = 0
    for f in os.listdir(img_dir):
        k = f.split('_')[1][:3]
        if keys is not None and k not in keys:
            continue
            
        if split.lower() == 'train' and int(k) % 2 == 1:
            continue
        if split.lower() == 'test' and int(k) % 2 == 0:
            continue    
        print(count, f)

        img = cv2.imread(os.path.join(img_dir, f), cv2.IMREAD_UNCHANGED)[:,:,:3]
        gt =  np.array(Image.open(os.path.join(gt_dir, 'gt_'+k+'.png')))
        if sz is not None:
            img = cv2.resize(img, sz, interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, sz, interpolation=cv2.INTER_NEAREST)
        
        imgs[k] = img
        masks[k] = gt
        count += 1

        if num_read is not None and count == num_read:
            break
    
    return imgs, masks

def img2patches(image, patch_sz, overlap):

    '''
    image: np array of the image
    patch_sz: tuple (dim1, dim2, ...), patch size to crop
    overlap: tuple (dim1, dim2, ...), overlap long height and width direction
    '''
    img_sz = image.shape
    step = [patch_sz[i]-overlap[i] for i in range(len(patch_sz))]

    patches = []
    position, size = [], []
    for i in range(2):
        pos_ = list(range(0, img_sz[i]-patch_sz[i]+1, step[i]))
        sz_ = [patch_sz[i]] * len(pos_)
        if pos_[-1] + patch_sz[i] != img_sz[i]:
            sz_.append(patch_sz[i])
            pos_.append(img_sz[i] - patch_sz[i])

        position.append(pos_)
        size.append(sz_)

    position = np.stack(np.meshgrid(*position), axis=-1).reshape((-1,2))
    size = np.stack(np.meshgrid(*size), axis=-1).reshape((-1,2))
    for i in tqdm(range(len(position)), ncols=100, desc='splitting: '):
        s = tuple([slice(position[i,j], position[i,j]+size[i,j]) for j in range(2)])
        patch_img = image[s]
        # print(patch.shape, position[i], patch[0,0,0], patch[0,0,0]//(500*64))
        patches.append(patch_img)
        
    return patches

def read_patches(ds_dir, split='pos', num_read=None):

    def phase_key(fname):
        key = fname.split('.')[0].split('_')
        return key[1] + '_' + key[2]

    imgs, masks = {}, {}
    if split == 'all':
        splits = ['pos', 'neg']
    else:
        splits = [split]
    for s in splits:
        if s == 'pos':
            img_dir = os.path.join(ds_dir, 'patches_pos') 
            gt_dir = os.path.join(ds_dir, 'mask_pos')
        else:
            img_dir = os.path.join(ds_dir, 'patches_neg') 
            gt_dir = os.path.join(ds_dir, 'mask_neg')

        for f in os.listdir(img_dir):
            key = phase_key(f)
            imgs[key] = os.path.join(img_dir, f)
        
        for f in os.listdir(gt_dir):
            key = phase_key(f)
            masks[key] = os.path.join(gt_dir, f)

    count = 0
    for k, img in imgs.items():
        print(k)

        imgs[k] = cv2.imread(img, cv2.IMREAD_UNCHANGED)[:,:,:3]

        if k in masks.keys():
            masks[k] = np.array(Image.open(masks[k]))
        else:
            masks[k] = np.zeros(imgs[k].shape[:2])

        count += 1

        if num_read is not None and count == num_read:
            break
    
    for k in list(imgs.keys()):
        if isinstance(imgs[k], str):
            del imgs[k]
            if k in masks.keys():
                del masks[k] 
    
    # print(imgs.keys(), masks.keys())
    
    return imgs, masks

def extract_patches(ds_dir, save_dir, cyst_bank=None, split='train', sz=(512, 512), overlap=(0, 0), 
                    obj_aug=True, cyst_min=5, cyst_max=10):
    
    from blender import CystBank, Blender

    if obj_aug:
        assert cyst_bank is not None
        bank = CystBank(cyst_bank)
        blender = Blender(bank)
        

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(os.path.join(save_dir, 'patches_pos'))
    os.makedirs(os.path.join(save_dir, 'patches_neg'))
    os.makedirs(os.path.join(save_dir, 'mask_pos'))
    os.makedirs(os.path.join(save_dir, 'mask_neg'))

    imgs, masks = read_pns_cyst(ds_dir, split=split, num_read=None)
    for k, img in imgs.items():
        patches_img = img2patches(img, sz, overlap)
        patches_mask = img2patches(masks[k], sz, overlap)

        for idx in range(len(patches_img)):
            img_p, mask_p = patches_img[idx], patches_mask[idx]
            
            if np.any(mask_p > 0):
                patch_dir = os.path.join(save_dir, 'patches_pos')
                mask_dir = os.path.join(save_dir, 'mask_pos')
            else:
                patch_dir = os.path.join(save_dir, 'patches_neg')
                mask_dir = os.path.join(save_dir, 'mask_neg')
            
            if obj_aug:
                img_p, mask_p = blender.generate(img_p, cyst_min=cyst_min, cyst_max=cyst_max, matting=True)
            
            cv2.imwrite(os.path.join(patch_dir, "patch_{}_{}.png".format(k, idx)), img_p)
            if np.any(mask_p > 0):
                mask_p = relabel(mask_p).astype(np.uint8)
                mask_p = Image.fromarray(mask_p, mode='P')
                mask_p.putpalette(palette)
                mask_p.save(os.path.join(mask_dir, "mask_{}_{}.png".format(k, idx)))



if __name__ == "__main__":

    ds_dir = './PNS-Cyst-part1'

    # imgs, masks = read_pns_cyst(ds_dir, part='part1', split='test', num_read=10)

    # extract_patches(ds_dir, './patches_train_add10', cyst_bank='./CystBank/train', split='train', obj_aug=True, cyst_min=5, cyst_max=10)

    imgs, masks = read_patches('./patches_train_add10', split='all', num_read=10)

    for k, img in imgs.items():
        cv2.imwrite('./test-{}.png'.format(k), img)
        cv2.imwrite('./mask-{}.png'.format(k), masks[k])


    # patches = split(imgs['001'], (512, 512), (128,128))
    # masks = split(masks['001'], (512, 512), (128,128))



    # for i in range(len(patches)):
    #     extract_patches
    #     cv2.imwrite('img_{}.png'.format(i), patches[i])
    #     cv2.imwrite('mask_{}.png'.format(i), masks[i]*255)

    # print(imgs.keys(), masks.keys())
    # print(imgs['066'].shape, masks['066'].shape)


    




