from skimage.measure import label as label_connected_component
from skimage.measure import regionprops
from skimage.measure import label as relabel
from pycocotools.coco import COCO
import os
import numpy as np
import h5py
import cv2
import scipy
from PIL import Image

palette = [255, 255, 255, 231, 10, 156, 231, 60, 26, 231, 129, 69, 231, 217, 30, 231, 30, 74, 231, 79, 200, 231, 148, 243, 231, 237, 204, 231, 49, 247, 68, 164, 255, 67, 252, 216, 67, 65, 3, 68, 115, 129, 68, 183, 172, 68, 116, 24, 68, 185, 68, 68, 234, 193, 68, 47, 237, 68, 135, 198, 73, 28, 217, 73, 97, 4, 73, 185, 221, 73, 254, 9, 73, 47, 134, 73, 116, 178, 73, 48, 30, 73, 117, 73, 73, 167, 199, 73, 236, 243, 72, 134, 200, 72, 184, 70, 72, 252, 113, 72, 85, 74, 72, 154, 118, 72, 203, 243, 72, 16, 31, 73, 204, 139, 73, 17, 182, 73, 67, 52, 71, 221, 9, 71, 34, 53, 71, 83, 179, 71, 152, 222, 71, 241, 183, 71, 54, 226, 71, 103, 96, 71, 172, 140, 72, 104, 248, 72, 173, 35, 71, 33, 157, 71, 121, 118, 71, 190, 162, 71, 239, 32, 71, 52, 75, 70, 141, 36, 70, 209, 79, 70, 3, 205, 70, 72, 249, 71, 4, 101, 76, 153, 120, 76, 221, 163, 76, 54, 124, 76, 123, 167, 76, 172, 37, 76, 241, 81, 76, 73, 42, 76, 142, 85, 76, 191, 211, 76, 4, 254, 75, 3, 103, 75, 52, 228, 75, 121, 16, 75, 210, 233, 75, 23, 20, 75, 72, 146, 75, 141, 189, 75, 229, 150, 75, 42, 194, 75, 91, 64, 74, 90, 168, 74, 159, 212, 74, 208, 81, 74, 21, 125, 74, 110, 86, 74, 178, 129, 74, 228, 255, 74, 41, 42, 74, 129, 3, 74, 198, 47, 74, 158, 60, 74, 246, 21, 74, 59, 64, 74, 108, 190, 74, 177, 234, 73, 9, 195, 73, 78, 238, 74, 128, 108, 73, 197, 151, 73, 29, 112, 126, 126, 200, 126, 175, 70, 126, 244, 113, 126, 77, 74, 126, 145, 118, 126, 195, 243, 126, 8, 31, 126, 96, 248, 126, 165, 35, 126, 214, 161, 125, 213, 10, 125, 26, 53, 126, 75, 179, 126, 144, 222, 125, 233, 183, 125, 45, 227, 125, 95, 96, 125, 164, 140, 125, 252, 101, 125, 65, 144, 125, 25, 157, 125, 113, 118, 125, 182, 162, 125, 231, 32, 125, 44, 75, 125, 132, 36, 125, 201, 79, 125, 251, 205, 125, 64, 249, 125, 152, 210, 130, 144, 120, 130, 213, 163, 130, 46, 124, 130, 114, 167, 130, 164, 37, 130, 233, 81, 130, 65, 42, 130, 134, 85, 130, 183, 211, 130, 252, 254, 129, 251, 103, 129, 44, 229, 129, 113, 16, 129, 201, 233, 129, 14, 20, 129, 64, 146, 129, 133, 190, 129, 221, 151, 129, 34, 194, 129, 83, 64, 129, 82, 168, 128, 151, 212, 129, 200, 81, 129, 13, 125, 128, 101, 86, 128, 170, 129, 128, 220, 255, 128, 33, 42, 128, 121, 3, 128, 190, 47, 129, 250, 207, 128, 238, 21, 128, 51, 65, 128, 100, 190, 128, 169, 234, 128, 1, 195, 128, 70, 238, 128, 120, 108, 128, 188, 151, 128, 21, 112, 128, 81, 17, 128, 149, 60, 133, 170, 27, 133, 239, 70, 133, 33, 196, 133, 102, 239, 133, 190, 200, 133, 3, 244, 133, 52, 114, 133, 121, 157, 133, 220, 153, 133, 13, 22, 133, 82, 66, 132, 70, 136, 132, 139, 179, 132, 189, 49, 132, 1, 92, 132, 90, 53, 132, 159, 97, 132, 208, 223, 132, 51, 218, 132, 120, 6, 132, 169, 131, 132, 238, 175, 131, 226, 245, 131, 39, 32, 132, 89, 158, 131, 157, 201, 131, 246, 162, 131, 59, 206, 192, 223, 252, 192, 36, 40, 192, 125, 1, 192, 193, 44, 192, 243, 170, 192, 56, 213, 192, 144, 174, 192, 213, 218, 192, 6, 87, 192, 75, 131, 197, 107, 132, 197, 156, 2, 197, 225, 45, 197, 57, 6, 197, 126, 50, 197, 175, 175, 197, 244, 219, 197, 77, 180, 197, 146, 223, 197, 195, 93, 197, 194, 197, 197, 6, 241, 197, 56, 111, 197, 125, 154, 197, 213, 115, 197, 26, 158, 197, 75, 28, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]

#### Datasets ####

## U2OS:https://bbbc.broadinstitute.org/BBBC039
# required directory 

## Celegans: https://bbbc.broadinstitute.org/BBBC010
# required directory: BBBC010_v2_images, BBBC010_v1_foreground_eachworm

## CervicalCytology2014: https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html
# required directory: Dataset 

## Neuroblastoma: http://www.cellimagelibrary.org/images/CCDB_6843
# required directory: MP6843_img_full, MP6843_seg

## BreastCancerCell: https://zenodo.org/record/4034976#.Y_d399LMJiN
# required directory: Training dataset

## CVPPP-LSC: https://competitions.codalab.org/competitions/18405#phases
# required directory: CVPPP2017_training_images.h5, CVPPP2017_training_truth.h5, CVPPP2017_testing_images.h5

## PanNuke: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
# required directory: Fold 1, Fold 2, Fold 3

## PNS-Cyst: 
# required directory: cyst20180529, cyst20180618, cyst20180809, cyst20180904, cyst20190207, cyst20200917, cyst20200924

PATH = {'U2OS': '/work/scratch/chen/Datasets/BBBC039/',
        'Celegans': '/work/scratch/chen/Datasets/Celegans/',
        'Cervical2014': '/work/scratch/chen/Datasets/Cervical2014/',
        'Neuroblastoma': '/work/scratch/chen/Datasets/Neuroblastoma/',
        'BreastCancerCell': '/work/scratch/chen/Datasets/BreastCancerCell/',
        'CVPPP-LSC': '/work/scratch/chen/Datasets/CVPPP-LSC/',
        'PanNuke': '/work/scratch/chen/Datasets/PanNuke/',
        'PNS-Cyst': '/work/scratch/chen/Datasets/PNS-Cyst/',
        'PNS-Cyst-p1': '/work/scratch/chen/Datasets/PNS-Cyst/',
        'PNS-Cyst-p2': '/work/scratch/chen/Datasets/PNS-Cyst/',
        'LIVECell': '/work/scratch/chen/Datasets/LIVECell/'}

# PATH = {'U2OS': '/work/scratch/chen/Datasets/BBBC039/',
#         'Celegans': '/work/scratch/chen/Datasets/Celegans/',
#         'Cervical2014': '/work/scratch/chen/Datasets/Cervical2014/',
#         'Neuroblastoma': '/work/scratch/chen/Datasets/Neuroblastoma/',
#         'BreastCancerCell': '/work/scratch/chen/Datasets/BreastCancerCell/',
#         'CVPPP-LSC': '/work/scratch/chen/Datasets/CVPPP-LSC/',
#         'PanNuke': 'e:/PanNuke/'}
BBX = {'Celegans': (50, 50+448, 120, 120+448)}


#### utils ####

def save_indexed_png(gt, path):
    gt = gt.astype(np.uint8)
    if len(np.unique(gt)) < 256:
        # gt = relabel(gt).astype(np.uint8)
        gt_save = Image.fromarray(gt, mode='P')
        gt_save.putpalette(palette)
        gt_save.save(path)
    else:
        # gt = relabel(gt).astype(np.uint8)
        cv2.imwrite(path, gt.astype(np.uint16))

def read_indexed_png(path):
    mask = Image.open(os.path.join(path))
    return np.array(mask)

def label_instance(gt, obj_min=100):
    L = label_connected_component(gt>0)
    for r in regionprops(L):
        if r.area < obj_min:
            L[r.coords[:,0], r.coords[:,1]] = 0
    return L

def map2stack(label_map):
    M = []
    for l in np.unique(label_map):
        if l == 0 :
            continue
        M.append(label_map == l)
    return M

# def stack2map(masks):
#     if not isinstance(masks, list):
#         return masks
#     else:
#         masks = [m>0 for m in masks]
#         S = np.sum(masks, axis=0)
#         labeled = np.sum(masks * np.expand_dims(np.arange(len(masks))+1, axis=(1,2)), axis=0)
#         labeled = labeled * (S == 1)
#         return labeled

def labeled_non_overlap(masks):
    '''
    overlap region marked as zero
    '''
    if isinstance(masks, list):
        masks = [m>0 for m in masks]
        S = np.sum(masks, axis=0)
        labeled = np.sum(masks * np.expand_dims(np.arange(len(masks))+1, axis=(1,2)), axis=0)
        labeled = labeled * (S == 1)
        return labeled
    else:
        return masks

#### Datasets ####

def alias(ds):
    if ds.lower() in {'u2os', 'bbbc039'}:
        return 'U2OS'
    if ds.lower() in {'bbbc010', 'celegans'}:
        return 'Celegans'
    if ds.lower() in {'occ2014', 'cervicalcytology2014', 'cervical2014'}:
        return 'Cervical2014'
    if ds.lower() in {'neuroblastoma', 'ccdb6843'}:
        return 'Neuroblastoma'
    if ds.lower() in {'breast', 'breastcancercell', 'breastcancer'}:
        return 'BreastCancerCell'
    if ds.lower() in {'cvppp', 'lsc', 'cvppp-lsc'}:
        return 'CVPPP-LSC'
    if ds.lower() in {'pannuke', 'pathology'}:
        return 'PanNuke'
    if ds.lower() in {'pns-p1', 'pns-cyst-p1', 'cyst-p1'}:
        return 'PNS-Cyst-p1'
    if ds.lower() in {'pns-p2', 'pns-cyst-p2', 'cyst-p2'}:
        return 'PNS-Cyst-p2'
    if ds.lower() in {'pns', 'pns-cyst', 'cyst'}:
        return 'PNS-Cyst'
    return ds

class Datasets(object):

    def __init__(self, path=None):
        self.path = PATH if path is None else path

    def prepare(self, ds):

        ds = alias(ds)

        if ds == 'Celegans':
            path_img = os.path.join(self.path[ds], 'BBBC010_v2_images')
            path_gt = os.path.join(self.path[ds], 'BBBC010_v1_foreground_eachworm')
            path_save = os.path.join(self.path[ds], 'Celegans')
            path_vis = os.path.join(self.path[ds], 'Celegans_uint8')
            bbx = BBX[ds]
            
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            if not os.path.exists(path_vis):
                os.makedirs(path_vis)

            for f in os.listdir(path_img):
                if not f.endswith('.tif'):
                    continue
                fns = f.split('_')
                if fns[7] == 'w1':
                    continue
                print('processing image: ', f)
                img = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                img = img[bbx[0]:bbx[1], bbx[2]:bbx[3]]
                cv2.imwrite(os.path.join(path_save, fns[6]+'.tif'), img)
                img = (img - img.min())/(img.max() - img.min()) * 255
                cv2.imwrite(os.path.join(path_vis, fns[6]+'.tif'), img.astype(np.uint8))
                
            for f in os.listdir(path_gt):
                if not f.endswith('.png'):
                    continue
                fns = f.split('_')
                print('processing gt: ', f)
                gt = cv2.imread(os.path.join(path_gt, f), cv2.IMREAD_UNCHANGED)
                gt = gt[bbx[0]:bbx[1], bbx[2]:bbx[3]]
                if not os.path.exists(os.path.join(path_save, fns[0])):
                    os.makedirs(os.path.join(path_save, fns[0]))
                cv2.imwrite(os.path.join(path_save, fns[0], 'worm'+fns[1])+'.png', gt)
        
        if ds == 'Cervical2014':
            path_data = os.path.join(self.path[ds], 'Dataset', 'Synthetic')
            path_save = os.path.join(self.path[ds], 'Cervical2014')
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            # save train images
            with h5py.File(os.path.join(path_data, 'trainset.mat'), 'r') as f:
                img_mat = f['trainset'][0]
                for idx, ref in enumerate(img_mat):
                    print('extracting train image: ', idx)
                    name = h5py.h5r.get_name(ref, f.id)
                    img = np.moveaxis(f[name][...], 0, 1)
                    cv2.imwrite(os.path.join(path_save, 'train{:03d}_img.png'.format(idx)), img)

            GT_train = scipy.io.loadmat(os.path.join(path_data, 'trainset_GT.mat'))
            for idx in range(len(GT_train['train_Nuclei'])):
                print('extracting train GT: ', idx)
                cv2.imwrite(os.path.join(path_save, 'train{:03d}_nucleus.png'.format(idx)), np.uint8(GT_train['train_Nuclei'][idx][0]>0)*255)
                subdir = 'train{:03d}_cytoplasm'.format(idx)
                if not os.path.exists(os.path.join(path_save, subdir)):
                    os.makedirs(os.path.join(path_save, subdir))
                for jdx, c in enumerate(GT_train['train_Cytoplasm'][idx][0]):
                    cv2.imwrite(os.path.join(path_save, subdir, 'cytoplasm_'+str(jdx)+'.png'), np.uint8(c[0]>0)*255)
            
            # save test images
            with h5py.File(os.path.join(path_data, 'testset.mat'), 'r') as f:
                img_mat = f['testset'][0]
                for idx, ref in enumerate(img_mat):
                    print('extracting test image: ', idx)
                    name = h5py.h5r.get_name(ref, f.id)
                    cv2.imwrite(os.path.join(path_save, 'test{:03d}_img.png'.format(idx)), f[name][...])

            with h5py.File(os.path.join(path_data, 'testset_GT.mat'), 'r') as f:
                for idx in range(len(f['test_Nuclei'][0])):
                    print('extracting test GT: ', idx)
                    Nname = h5py.h5r.get_name(f['test_Nuclei'][0][idx], f.id)
                    cv2.imwrite(os.path.join(path_save, 'test{:03d}_nucleus.png'.format(idx)), np.uint8(f[Nname][...]>0)*255)
                    
                    subdir = 'test{:03d}_cytoplasm'.format(idx)
                    if not os.path.exists(os.path.join(path_save, subdir)):
                        os.makedirs(os.path.join(path_save, subdir))
                    Cname = h5py.h5r.get_name(f['test_Cytoplasm'][0][idx], f.id)
                    for jdx, Cref in enumerate(f[Cname][...][0]):
                        cv2.imwrite(os.path.join(path_save, subdir, 'cytoplasm_'+str(jdx)+'.png'), np.uint8(f[h5py.h5r.get_name(Cref, f.id)][...]>0)*255)
        
        if ds == 'Neuroblastoma':
            path_img = os.path.join(self.path[ds], 'MP6843_img_full')
            path_vis = os.path.join(self.path[ds], 'MP6843_img_uint8')
            
            if not os.path.exists(path_vis):
                os.makedirs(path_vis)

            for f in os.listdir(path_img):
                if not f.endswith('.TIF'):
                    continue
                if f.split('w')[1].startswith('2'):
                    continue
                print('processing image: ', f)
                img = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                img = (img - img.min())/(img.max() - img.min()) * 255
                cv2.imwrite(os.path.join(path_vis, f), img.astype(np.uint8))

        if ds == 'BreastCancerCell':
            path_img = os.path.join(self.path[ds], 'Training dataset', 'Training_source')
            path_gt = os.path.join(self.path[ds], 'Training dataset', 'Training_target')
            path_save = os.path.join(self.path[ds], 'BreastCancerCell')
            path_vis = os.path.join(self.path[ds], 'BreastCancerCell_uint8')

            if not os.path.exists(path_save):
                os.makedirs(path_save)
            if not os.path.exists(path_vis):
                os.makedirs(path_vis)

            count = 0
            for f in sorted(os.listdir(path_img)):
                img = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                gt = Image.open(os.path.join(path_gt, f))
                gt = np.array(gt)
                
                print('processing image: ', f)
                if img.shape[0] != 1024 or img.shape[1] != 1024:
                    continue

                for i in range(2):
                    for j in range(2):
                        img_sub = img[512*i:512*(i+1), 512*j:512*(j+1)]
                        gt_sub = relabel(gt[512*i:512*(i+1), 512*j:512*(j+1)])
                        if np.sum(gt_sub > 0) > 10000:
                            cv2.imwrite(os.path.join(path_save, "image_{}.png".format(count)), img_sub)
                            save_indexed_png(gt_sub, os.path.join(path_save, "mask_{}.png".format(count)))
                            img_sub = (img_sub - img_sub.min())/(img_sub.max() - img_sub.min()) * 255
                            cv2.imwrite(os.path.join(path_vis, "image_{}.png".format(count)), img_sub.astype(np.uint8))
                            count += 1
        
        if ds == 'CVPPP-LSC':
            img_db = os.path.join(self.path[ds], 'CVPPP2017_training_images.h5')
            f_img = h5py.File(img_db, 'r')
            gt_db = os.path.join(self.path[ds], 'CVPPP2017_training_truth.h5')
            f_gt = h5py.File(gt_db, 'r')
            path_save = os.path.join(self.path[ds], 'CVPPP-LSC')
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            for subset in ['A1', 'A2', 'A3', 'A4']:
                for k in f_img[subset]:
                    print('processing training image: ', subset, k)
                    img = np.array(f_img[subset][k]['rgb'])[:,:,:3]
                    gt = relabel(np.array(f_gt[subset][k]['label']))
                    cv2.imwrite(os.path.join(path_save, 'train_{}-{}.png'.format(subset, k)), img)
                    save_indexed_png(gt, os.path.join(path_save, 'train_{}-{}_mask.png'.format(subset, k)))
            f_img.close()
            f_gt.close()

            img_db = os.path.join(self.path[ds], 'CVPPP2017_testing_images.h5')
            f_img = h5py.File(img_db, 'r')

            for subset in ['A1', 'A2', 'A3', 'A4', 'A5']:
                # path_subset = os.path.join(path_save, subset)
                # if not os.path.exists(path_subset):
                #     os.makedirs(path_subset)
                for k in f_img[subset]:
                    print('processing testing image: ', subset, k)
                    img = np.array(f_img[subset][k]['rgb'])[:,:,:3]
                    cv2.imwrite(os.path.join(path_save, 'test_{}-{}.png'.format(subset, k)), img)

        if ds == 'U2OS':
            path_gt = os.path.join(self.path[ds], 'masks')
            path_vis = os.path.join(self.path[ds], 'masks_vis')
            

            if not os.path.exists(path_vis):
                os.makedirs(path_vis)

            for f in os.listdir(path_gt):
                gt = relabel(cv2.imread(os.path.join(path_gt, f), cv2.IMREAD_UNCHANGED)[:,:,2])
                save_indexed_png(gt, os.path.join(path_vis, f))

        if ds == 'PanNuke':
            path_save = os.path.join(self.path[ds], 'PanNuke')

            if not os.path.exists(path_save):
                os.makedirs(path_save)

            for p, subD in enumerate(['Fold 1', 'Fold 2', 'Fold 3'], start=1):
                imgs = np.load(os.path.join(self.path[ds], subD, 'images', 'fold'+str(p), 'images.npy'))
                gts = np.load(os.path.join(self.path[ds], subD, 'masks', 'fold'+str(p), 'masks.npy'))
                for idx in range(len(imgs)):
                    print('processing image: ', subD, idx)
                    fname = 'p{}_{}'.format(p, idx)
                    cv2.imwrite(os.path.join(path_save, fname+'_img.png'), imgs[idx,:,:,::-1])
                    gt = relabel(np.max(gts[idx,:,:,0:5], axis=-1))
                    semantic = np.max((gts[idx,:,:,0:5] > 0)*np.expand_dims(np.arange(5)+1, axis=(0,1)), axis=-1)
                    assert len(np.unique(semantic)) <= 6
                    for r in regionprops(gt):
                        if r.area < 10:
                            gt[r.coords[:,0], r.coords[:,1]] = 0
                    semantic = semantic * (gt > 0)
                    save_indexed_png(gt, os.path.join(path_save, fname+'_mask.png'))
                    save_indexed_png(semantic, os.path.join(path_save, fname+'_semantic.png'))

        if ds == 'PNS-Cyst-p1' or ds == 'PNS-Cyst-p2':

            path_save = os.path.join(self.path[ds], ds)
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            count = 0

            if ds == 'PNS-Cyst-p1':
                subDs = ['cyst20180529', 'cyst20180618', 'cyst20180809', 'cyst20180904', 'cyst20190207']
                M, N = 2, 2
            elif ds == 'PNS-Cyst-p2':
                subDs = ['cyst20200917', 'cyst20200924']
                M, N = 6, 8

            for subD in subDs:
                path_data = os.path.join(self.path[ds], subD)
                for f in sorted(os.listdir(path_data)):
                    if f.startswith('gt'):
                        continue
                    print('processing image: ', subD, f)
                    img = cv2.imread(os.path.join(path_data, f), cv2.IMREAD_UNCHANGED)
                    gt = read_indexed_png(os.path.join(path_data, 'gt'+f[5:-3]+'png'))

                    for i in range(M):
                        for j in range(N):
                            img_sub = img[480*i:480*(i+1), 512*j:512*(j+1)]
                            gt_sub = relabel(gt[480*i:480*(i+1), 512*j:512*(j+1)])
                            if np.sum(gt_sub > 0) > 5000:
                                cv2.imwrite(os.path.join(path_save, "image_{}.png".format(count)), img_sub)
                                save_indexed_png(gt_sub, os.path.join(path_save, "mask_{}.png".format(count)))
                                count += 1
            print('Done')

    def _key(self, ds, f):

        sn, f_type = None, None
        if ds == 'Celegans':
            k, sn = f[:-4], int(f[1:-4])

        if ds == 'Cervical2014':
            k, f_type = f[:-4].split('_')

        if ds == 'Neuroblastoma':
            k = f[:-6]

        if ds == 'BreastCancerCell':
            k = int(f.split('_')[-1][:-4])

        if ds == 'CVPPP-LSC':
            k = f[:-4]
            fs = k.split('-')
            sn, f_type = int(fs[1][5:]), fs[0]

        if ds == 'U2OS':
            fs = f.split('_')
            k = fs[1] + '_' + fs[2] + '_' + fs[3]

        if ds == 'PanNuke':
            fs = f.split('_')
            k = fs[0] + '_' + fs[1]
            sn, f_type = int(fs[1]), fs[0]

        if ds == 'PNS-Cyst-p1' or ds == 'PNS-Cyst-p2':
            fs = f[:-4].split('_')
            sn = int(fs[1])
            k = ds[-2:] + fs[1]

        return k, sn, f_type

    def keys(self, ds, split='all'):

        ds = alias(ds)

        keys = set()
        if ds == 'Celegans':
            path_ds = os.path.join(self.path[ds], 'Celegans')
            assert os.path.exists(path_ds)
            for f in os.listdir(path_ds):
                if not f.endswith('tif'):
                    continue
                
                # k, sn = f[:-4], int(f[1:-4])
                k, sn, f_type = self._key(ds, f)
                
                if split == 'all':
                    keys.add(k)
                if split == 'test' and k.startswith('D'):
                    keys.add(k)
                if split == 'train' and (not k.startswith('D')) and sn % 10 != 0:
                    keys.add(k)
                if split == 'val' and (not k.startswith('D')) and sn % 10 == 0:
                    keys.add(k)
                if split == 'train_ext' and (not k.startswith('D')):
                    keys.add(k)

                # split in BMCV paper, randomly selected train examples for validation:
                if split == 'test_bmvc' and k.startswith('D'):
                    keys.add(k)
                if split == 'train_bmvc' and (not k.startswith('D')):
                    keys.add(k)

        if ds == 'Cervical2014':
            path_ds = os.path.join(self.path[ds], 'Cervical2014')
            assert os.path.exists(path_ds)
            for f in os.listdir(path_ds):
                if not f.endswith('png'):
                    continue
                
                # k, f_type = f[:-4].split('_')
                k, sn, f_type = self._key(ds, f)
                if f_type != 'img':
                    continue
                f_split, f_sn = k[:-3], int(k[-3:])

                if split == 'all':
                    keys.add(k)
                if split == 'test' and (f_split == 'test' and f_sn % 4 == 0):
                    keys.add(k)
                if split == 'train' and (f_split == 'test' and f_sn % 4 != 0):
                    keys.add(k)
                if split == 'val' and f_split == 'train':
                    keys.add(k)
                if split == 'train_ext' and (f_split == 'train' or (f_split == 'test' and f_sn % 4 != 0)):
                    keys.add(k)

                # split in BMCV paper, randomly selected train examples for validation:
                if split == 'test_bmvc' and (f_split == 'test' and f_sn % 2 != 0):
                    keys.add(k)
                if split == 'train_bmvc' and (f_split == 'train' or (f_split == 'test' and f_sn % 2 == 0)):
                    keys.add(k)

        if ds == 'Neuroblastoma':
            path_ds = os.path.join(self.path[ds], 'MP6843_img_full')
            assert os.path.exists(path_ds)

            fs = []
            for f in os.listdir(path_ds):
                if not f.endswith('.TIF'):
                    continue

                if not f.split('w')[1].startswith('2'):
                    fs.append(f)

            for idx, f in enumerate(sorted(fs)):
                
                k, sn, f_type = self._key(ds, f)
                # k = f[:-6]

                if split == 'all':
                    keys.add(k)
                if split == 'test' and idx % 5 == 0:
                    keys.add(k)
                if split == 'train' and idx % 5 != 0 and idx % 10 != 1:
                    keys.add(k)
                if split == 'val' and idx % 5 != 0 and idx % 10 == 1:
                    keys.add(k)
                if split == 'train_ext' and idx % 5 != 0:
                    keys.add(k)

                # split in BMCV paper, randomly selected train examples for validation:
                if split == 'test_bmvc' and k.startswith('F04'):
                    keys.add(k)
                if split == 'train_bmvc' and (not k.startswith('F04')):
                    keys.add(k)

        if ds == 'BreastCancerCell':
            path_ds = os.path.join(self.path[ds], 'BreastCancerCell')
            assert os.path.exists(path_ds)

            for f in os.listdir(path_ds):
                if f.startswith('mask'):
                    continue

                k, sn, f_type = self._key(ds, f)

                if split == 'all':
                    keys.add(k)
                if split == 'train' and k % 10 < 7:
                    keys.add(k)
                if split == 'val' and k % 10 == 7:
                    keys.add(k)
                if split == 'test' and k % 10 > 7:
                    keys.add(k)
                if split == 'train_ext' and k % 10 <= 7:
                    keys.add(k)

        if ds == 'CVPPP-LSC':

            path_ds = os.path.join(self.path[ds], 'CVPPP-LSC')
            assert os.path.exists(path_ds)

            for f in os.listdir(path_ds):
                if f[:-4].endswith('mask'):
                    continue

                k, sn, f_type = self._key(ds, f)

                if split == 'challenge_train' and f_type.startswith('train'):
                    keys.add(k)
                if split == 'challenge_test' and f_type.startswith('test'):
                    keys.add(k)
                if split == 'train' and f_type.startswith('train') and f_type != 'train_A3' and sn % 10 < 7:
                    keys.add(k)
                if split == 'val' and f_type.startswith('train') and f_type != 'train_A3' and sn % 10 == 7:
                    keys.add(k)
                if split == 'test' and f_type.startswith('train') and f_type != 'train_A3' and sn % 10 > 7:
                    keys.add(k)
                if split == 'all' and f_type.startswith('train') and f_type != 'train_A3':
                    keys.add(k)

        if ds == 'U2OS':
            path_ds = os.path.join(self.path[ds], 'images')
            assert os.path.exists(path_ds)

            for f in os.listdir(path_ds):

                k, _, _ = self._key(ds, f)

                if split == 'train' and k[0] <= 'G':
                    keys.add(k)
                if split == 'val' and k[0] in ['K', 'L']:
                    keys.add(k)
                if split == 'test' and k[0] >= 'M':
                    keys.add(k)
                if split == 'all':
                    keys.add(k)

        if ds == 'PanNuke':
            path_ds = os.path.join(self.path[ds], 'PanNuke')
            assert os.path.exists(path_ds)

            for f in os.listdir(path_ds):
                if not f[:-4].endswith('img'):
                    continue
                k, sn, f_type = self._key(ds, f)

                if split == 'train' and f_type != 'p3':
                    keys.add(k)
                if split == 'val' and f_type == 'p3' and sn % 2 == 0:
                    keys.add(k)
                if split == 'test' and f_type == 'p3' and sn % 2 == 1:
                    keys.add(k)
                if split == 'all':
                    keys.add(k)

        if ds == 'PNS-Cyst-p1' or ds == 'PNS-Cyst-p2':
            path_ds = os.path.join(self.path[ds], ds)
            assert os.path.exists(path_ds)

            for f in os.listdir(path_ds):
                if not f.startswith('mask'):
                    continue
                k, sn, _ = self._key(ds, f)
                
                if split == 'train' and sn % 10 < 7:
                    keys.add(k)
                if split == 'val' and sn % 10 == 7:
                    keys.add(k)
                if split == 'test' and sn % 10 > 7:
                    keys.add(k)
                if split == 'all':
                    keys.add(k)

        return keys



    def load_data(self, ds, split=None, num_read=None, verbose=True, label_map=False):

        ds = alias(ds)

        split = 'all' if split is None else split
        keys = self.keys(ds, split)

        imgs, masks, semantics = {}, {}, {}
        if ds == 'Celegans':
            path_ds = os.path.join(self.path[ds], 'Celegans')

            idx = 0
            for f in os.listdir(path_ds):
                if not f.endswith('tif'):
                    continue
                k, _, f_type = self._key(ds, f)
                if k not in keys:
                    continue
                

                img = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                worms = []
                for s in os.listdir(os.path.join(path_ds, k)):
                    w = cv2.imread(os.path.join(path_ds, k, s), cv2.IMREAD_UNCHANGED)
                    worms.append(w>0)
                masks[k] = worms

                if verbose:
                    print("image: ", f, 'worms: ', len(worms))
                idx += 1
                if num_read is not None and idx == num_read:
                    break
        
        if ds == 'Cervical2014':
            path_ds = os.path.join(self.path[ds], 'Cervical2014')

            idx = 0
            for f in os.listdir(path_ds):
                if not f.endswith('png'):
                    continue
                k, _, f_type = self._key(ds, f)
                if f_type == 'nucleus' or k not in keys:
                    continue
                
                img = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                cytos = []
                for fc in os.listdir(os.path.join(path_ds, k+'_cytoplasm')):
                    seg = cv2.imread(os.path.join(path_ds, k+'_cytoplasm', fc), cv2.IMREAD_UNCHANGED)
                    cytos.append(seg)
                masks[k] = cytos

                if verbose:
                    print("image: ", f, 'cytoplasm: ', len(cytos))
                idx += 1
                if num_read is not None and idx == num_read:
                    break

        if ds == "Neuroblastoma":

            path_img = os.path.join(self.path[ds], 'MP6843_img_full')
            path_gt = os.path.join(self.path[ds], 'MP6843_seg')

            idx = 0
            for f in os.listdir(path_img):
                if not f.endswith('.TIF'):
                    continue
                if f.split('w')[1].startswith('2'):
                    continue
                k, _, f_type = self._key(ds, f)
                if k not in keys:
                    continue
                
                img = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                gt = cv2.imread(os.path.join(path_gt, k+'_GT_01.tif'), cv2.IMREAD_UNCHANGED)
                gt = gt[:,:,0] if len(gt.shape) == 3 else gt
                cells = map2stack(label_instance(gt))
                if os.path.exists(os.path.join(path_gt, k+'_GT_02.tif')):
                    gt = cv2.imread(os.path.join(path_gt, k+'_GT_02.tif'), cv2.IMREAD_UNCHANGED) > 0
                    gt = gt[:,:,0] if len(gt.shape) == 3 else gt
                    cells = cells + map2stack(label_instance(gt))
                masks[k] = cells

                if verbose:
                    print("image: ", f, 'cells: ', len(cells))
                idx += 1
                if num_read is not None and idx == num_read:
                    break
        
        if ds == 'BreastCancerCell':
            path_ds = os.path.join(self.path[ds], 'BreastCancerCell')
            assert os.path.exists(path_ds)

            idx = 0
            for f in os.listdir(path_ds):
                if f.startswith('mask'):
                    continue
                k, _, f_type = self._key(ds, f)
                if k not in keys:
                    continue

                mask = read_indexed_png(os.path.join(path_ds, 'mask_{}.png'.format(k)))
                masks[k] = mask
                imgs[k] = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                
                if verbose:
                    print("image: ", f, 'cells: ', len(np.unique(masks[k])))

                idx += 1
                if num_read is not None and idx == num_read:
                    break
        
        if ds == 'CVPPP-LSC':
            path_ds = os.path.join(self.path[ds], 'CVPPP-LSC')
            assert os.path.exists(path_ds)

            idx = 0
            for f in os.listdir(path_ds):
                if f[:-4].endswith('mask'):
                    continue
                k, _, f_type = self._key(ds, f)
                if k not in keys:
                    continue
                
                imgs[k] = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                masks[k] = read_indexed_png(os.path.join(path_ds, k+'_mask.png'))
                
                if verbose:
                    print("image: ", f, 'leaves: ', len(np.unique(masks[k])))
                # print(imgs[k].shape, masks[k].shape)

                idx += 1
                if num_read is not None and idx == num_read:
                    break

        if ds == 'U2OS':
            path_img = os.path.join(self.path[ds], 'images')
            path_gt = os.path.join(self.path[ds], 'masks')
            idx = 0
            for f in os.listdir(path_img):
                k, _, _ = self._key(ds, f)
                # print(f, k)
                if k not in keys:
                    continue
                
                imgs[k] = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                masks[k] = relabel(cv2.imread(os.path.join(path_gt, f[:-3]+'png'), cv2.IMREAD_UNCHANGED)[:,:,2])

                if verbose:                
                    print("image: ", f, 'cells: ', len(np.unique(masks[k])))

                idx += 1
                if num_read is not None and idx == num_read:
                    break

        if ds == 'PanNuke':
            path_ds = os.path.join(self.path[ds], 'PanNuke')
            assert os.path.exists(path_ds)

            idx = 0
            for f in os.listdir(path_ds):
                if not f[:-4].endswith('img'):
                    continue
                k, _, _ = self._key(ds, f)
                if k not in keys:
                    continue
                
                imgs[k] = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                masks[k] = read_indexed_png(os.path.join(path_ds, k+'_mask.png'))
                semantics[k] = read_indexed_png(os.path.join(path_ds, k+'_semantic.png'))
                
                if verbose:
                    print("image: ", f, 'nuclei: ', len(np.unique(masks[k])))
                # print(imgs[k].shape, masks[k].shape)

                idx += 1
                if num_read is not None and idx == num_read:
                    break
        
        if ds == 'PNS-Cyst-p1' or ds == 'PNS-Cyst-p2':
            path_ds = os.path.join(self.path[ds], ds)
            assert os.path.exists(path_ds)

            idx = 0
            for f in os.listdir(path_ds):
                if f[:-4].startswith('mask'):
                    continue
                k, _, _ = self._key(ds, f)
                if k not in keys:
                    continue
                
                img = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img[:,:,:3]
                masks[k] = read_indexed_png(os.path.join(path_ds, 'mask'+f[5:]))
                
                if verbose:
                    print("image: ", f, 'cysts: ', len(np.unique(masks[k])))
                # print(imgs[k].shape, masks[k].shape)

                idx += 1
                if num_read is not None and idx == num_read:
                    break
        if ds == 'PNS-Cyst':
            if num_read is not None:
                imgs, masks, _ = self.load_data('PNS-Cyst-p1', split=split, num_read=num_read, verbose=verbose)
            else:
                imgs, masks, _ = self.load_data('PNS-Cyst-p1', split=split, num_read=None, verbose=verbose)
                imgs2, masks2, _ = self.load_data('PNS-Cyst-p2', split=split, num_read=None, verbose=verbose)
                imgs.update(imgs2)
                del imgs2
                masks.update(masks2)
                del masks2

        if ds.startswith('LIVECell'):

            splits = ['train', 'val', 'test'] if split == 'all' else [split]

            idx = 0
            for split in splits:

                if ds == 'LIVECell':
                    annFile = os.path.join(self.path['LIVECell'], 'annotations', 'LIVECell', 'livecell_coco_'+split+'.json')
                else:
                    ctype = ds.split('-')[1]
                    annFile = os.path.join(self.path['LIVECell'], 'annotations', 'LIVECell_single_cells', ctype.lower(), split+'.json')
                dataset = COCO(annFile)
                ids = dataset.getImgIds()
                subfolder = 'livecell_test_images' if split == 'test' else 'livecell_train_val_images'
                for item in dataset.loadImgs(ids):
                    print('loading image: ', item['file_name'])
                    # print(item)
                    # print('image loaded: ', item['id'], item['file_name'])
                    # fname = item['file_name']
                    imgs[item['id']] = cv2.imread(os.path.join(self.path['LIVECell'], 'images', subfolder, item['file_name']), cv2.IMREAD_UNCHANGED)

                    annoIds =  dataset.getAnnIds(imgIds=item['id'], iscrowd=None)
                    masks[item['id']] = []
                    for anno in dataset.loadAnns(annoIds):
                        masks[item['id']].append(dataset.annToMask(anno)>0)
                    
                    idx += 1
                    if num_read is not None and idx == num_read:
                        break
        if label_map:
            for k in list(masks.keys()):
                masks[k] = labeled_non_overlap(masks[k])

        return imgs, masks, semantics

    def info(self, ds):
        ds = alias(ds)

        for split in ['train', 'val', 'test', 'all']:
            imgs, masks, _ = self.load_data(ds, split=split, verbose=False)
            objs = 0
            for _, m in masks.items():
                if isinstance(m, list):
                    objs += len(m)
                else:
                    objs += (len(np.unique(m)) - 1)
            print('{} set: {} images, {} objects'.format(split, len(imgs), objs))


if __name__ == "__main__":
    ds = Datasets()
    # ds.prepare('Cervical2014')
    # ds.prepare('Neuroblastoma')
    # ds.prepare('breast')
    # ds.prepare('CVPPP-LSC')
    # ds.prepare('U2OS')
    # ds.prepare('PanNuke')
    ds.prepare('PNS-Cyst-p1')
    ds.prepare('PNS-Cyst-p2')
    
    # k = ds.keys('Celegans', split='val')
    # k = ds.keys('Cervical2014', split='test')
    # k = ds.keys('Neuroblastoma', split='train_bmvc')
    # k = ds.keys('breast', split='train_bmvc')
    # print(k)
    # print(len(k))

    # ds.info('U2OS')
    # ds.info('cvppp')
    # ds.info('Celegans')
    # ds.info('PanNuke')
    # ds.info('PNS-Cyst-p1')

    # imgs, masks = ds.load_data('Celegans', split='test', num_read=10)
    # imgs, masks = ds.load_data('Cervical2014', split='test', num_read=10)
    # imgs, masks = ds.load_data('breast', split='val', num_read=None)
    # imgs, masks = ds.load_data('CVPPP-LSC', split='train', num_read=None)
    # imgs, masks, _ = ds.load_data('LIVECell-a172', split='val', num_read=None)
    # for k, img in imgs.items():
    #     assert k in masks.keys()
    # print(len(imgs) == len(masks))
    # cv2.imwrite('./test.png', imgs[219512])
    # cv2.imwrite('./mask.png', (np.sum(masks[219512], axis=0)>0).astype(np.uint8)*255)
    # print(len(imgs))

    # imgs, masks = ds.load_data('Neuroblastoma', split='test', num_read=10)
    # for k in imgs.keys():
    #     print(imgs[k].shape, np.array(masks[k]).shape)
    #     assert k in masks.keys()
    # print(len(imgs), len(masks))

    # keys = list(imgs.keys())
    # img = imgs[keys[0]]
    # img = (img - img.min())/(img.max()-img.min())*255
    # cv2.imwrite('./test_image.png', img.astype(np.uint8))
    # mask = np.sum(masks[keys[0]], axis=0) > 0
    # # print(mask.shape)
    # cv2.imwrite('./test_gt.png', (mask * 255).astype(np.uint8))

    import instSeg
    
    radius = 15

    Ds = ['U2OS', 'BreastCancerCell', 'PanNuke', 'CVPPP', 'PNS-Cyst-p1', 'PNS-Cyst-p2', 'Neuroblastoma', 'Celegans', 'Cervical2014']
    Ds = ['LIVECell-a172', 'LIVECell-bt474', 'LIVECell-bv2', 'LIVECell-huh7', 'LIVECell-mcf7', 'LIVECell-shsy5y', 'LIVECell-skbr3', 'LIVECell-skov3']
    # Ds = ['LIVECell-a172']

    results = []

    for D in Ds:
        Nadj = []
        Nobj = []
        imgs, masks = {}, {}

        imgs, masks, _ = ds.load_data(D, split='all')


        for k, m in masks.items():
            if isinstance(m, list):
                labels = np.moveaxis(np.array(m), 0, -1)
                Nobj.append(len(m))
            else:
                labels = np.expand_dims(m, axis=-1)
                Nobj.append(len(np.unique(m))-1)
            print('counting', k, labels.shape)
            labels = np.expand_dims(labels, axis=0)
            
            adj_matrix = instSeg.utils.adj_matrix(labels, radius=radius, progress_bar=False, max_obj=1024)[0]

            objs = range(len(m)) if isinstance(m, list) else np.unique(m)
            for i in objs:
                adj = np.sum(adj_matrix[i+1,:])
                Nadj.append(adj)

        results.append('Dataset {}, images {}, objects {}, adjacents {}'.format(D, len(imgs), np.sum(Nobj), np.mean(Nadj)))

    for r in results:
        print(r)
