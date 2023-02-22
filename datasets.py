from skimage.measure import label as label_connected_component
from skimage.measure import regionprops
import os
import numpy as np
import h5py
import cv2
import scipy

#### Datasets ####
## Celegans: https://bbbc.broadinstitute.org/BBBC010
# required directory: BBBC010_v2_images, BBBC010_v1_foreground_eachworm
## CervicalCytology2014: https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html
# required directory: Dataset 
## Neuroblastoma: http://www.cellimagelibrary.org/images/CCDB_6843
# required directory: MP6843_img_full, MP6843_seg

PATH = {'Celegans': '/work/scratch/chen/Datasets/Celegans/',
        'Cervical2014': '/work/scratch/chen/Datasets/Cervical2014/',
        'Neuroblastoma': '/work/scratch/chen/Datasets/Neuroblastoma/'}
BBX = {'Celegans': (50, 50+448, 120, 120+448)}


#### utils ####

def alias(ds):
    if ds.lower() in {'bbbc010', 'celegans'}:
        return 'Celegans'
    if ds.lower() in {'occ2014', 'cervicalcytology2014', 'cervical2014'}:
        return 'Cervical2014'
    if ds.lower() in {'neuroblastoma', 'ccdb6843'}:
        return 'Neuroblastoma'

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

def labeled_non_overlap(masks):
    '''
    overlap region marked as zero
    '''
    masks = [m>0 for m in masks]
    S = np.sum(masks, axis=0)
    labeled = np.sum(masks * np.expand_dims(np.arange(len(masks))+1, axis=(1,2)), axis=0)
    non_overlap, overlap =  S == 1, S > 1
    labeled = labeled * non_overlap
    # labeled = labeled + overlap * overlap_label
    return labeled, overlap

#### Datasets ####

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
                
    def keys(self, ds, split='all'):

        ds = alias(ds)

        keys = set()
        if ds == 'Celegans':
            path_ds = os.path.join(self.path[ds], 'Celegans')
            assert os.path.exists(path_ds)
            for f in os.listdir(path_ds):
                if not f.endswith('tif'):
                    continue
                
                k, sn = f[:-4], int(f[1:-4])
                
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
                
                k, f_type = f[:-4].split('_')
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
                
                k = f[:-6]

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
        
        return keys



    def load_data(self, ds, split=None, keys=None, num_read=None):

        ds = alias(ds)

        assert split is None or keys is None
        if split is not None:
            keys = self.keys(ds, split)

        imgs, masks = {}, {}
        if ds == 'Celegans':
            path_ds = os.path.join(self.path[ds], 'Celegans')

            idx = 0
            for f in os.listdir(path_ds):
                if not f.endswith('tif'):
                    continue
                
                k = f[:-4]
                if keys is not None and k not in keys:
                    continue
                

                img = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                worms = []
                for s in os.listdir(os.path.join(path_ds, k)):
                    w = cv2.imread(os.path.join(path_ds, k, s), cv2.IMREAD_UNCHANGED)
                    worms.append(w>0)
                masks[k] = worms

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

                k, f_type = f[:-4].split('_')
                
                if f_type == 'nucleus':
                    continue

                if keys is not None and k not in keys:
                    continue
                
                img = cv2.imread(os.path.join(path_ds, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                cytos = []
                for fc in os.listdir(os.path.join(path_ds, k+'_cytoplasm')):
                    seg = cv2.imread(os.path.join(path_ds, k+'_cytoplasm', fc), cv2.IMREAD_UNCHANGED)
                    cytos.append(seg)
                masks[k] = cytos

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

                k = f[:-6]

                if keys is not None and k not in keys:
                    continue
                
                img = cv2.imread(os.path.join(path_img, f), cv2.IMREAD_UNCHANGED)
                imgs[k] = img

                print(k+'_GT_01.tif')
                gt = cv2.imread(os.path.join(path_gt, k+'_GT_01.tif'), cv2.IMREAD_UNCHANGED)
                gt = gt[:,:,0] if len(gt.shape) == 3 else gt
                cells = map2stack(label_instance(gt))
                if os.path.exists(os.path.join(path_gt, k+'_GT_02.tif')):
                    gt = cv2.imread(os.path.join(path_gt, k+'_GT_02.tif'), cv2.IMREAD_UNCHANGED) > 0
                    gt = gt[:,:,0] if len(gt.shape) == 3 else gt
                    cells = cells + map2stack(label_instance(gt))
                masks[k] = cells

                print("image: ", f, 'cells: ', len(cells))
                idx += 1
                if num_read is not None and idx == num_read:
                    break

        return imgs, masks



if __name__ == "__main__":
    ds = Datasets()
    # ds.prepare('Cervical2014')
    # ds.prepare('Neuroblastoma')
    
    # k = ds.keys('Celegans', split='val')
    # k = ds.keys('Cervical2014', split='test')
    # k = ds.keys('Neuroblastoma', split='train_bmvc')
    # print(k, len(k))

    # imgs, masks = ds.load_data('Celegans', split='test', num_read=10)
    # imgs, masks = ds.load_data('Cervical2014', split='test', num_read=10)
    imgs, masks = ds.load_data('Neuroblastoma', split='test', num_read=10)
    for k in imgs.keys():
        print(imgs[k].shape, np.array(masks[k]).shape)
        assert k in masks.keys()
    print(len(imgs), len(masks))

    keys = list(imgs.keys())
    img = imgs[keys[0]]
    img = (img - img.min())/(img.max()-img.min())*255
    cv2.imwrite('./test_image.png', img.astype(np.uint8))
    mask = np.sum(masks[keys[0]], axis=0) > 0
    # print(mask.shape)
    cv2.imwrite('./test_gt.png', (mask * 255).astype(np.uint8))
