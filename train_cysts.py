import instSeg
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
from skimage.measure import label as relabel
import argparse
from PNS import read_pns_cyst, read_patches
import random
import csv

random.seed(1)

palette = [255, 255, 255, 231, 10, 156, 231, 60, 26, 231, 129, 69, 231, 217, 30, 231, 30, 74, 231, 79, 200, 231, 148, 243, 231, 237, 204, 231, 49, 247, 68, 164, 255, 67, 252, 216, 67, 65, 3, 68, 115, 129, 68, 183, 172, 68, 116, 24, 68, 185, 68, 68, 234, 193, 68, 47, 237, 68, 135, 198, 73, 28, 217, 73, 97, 4, 73, 185, 221, 73, 254, 9, 73, 47, 134, 73, 116, 178, 73, 48, 30, 73, 117, 73, 73, 167, 199, 73, 236, 243, 72, 134, 200, 72, 184, 70, 72, 252, 113, 72, 85, 74, 72, 154, 118, 72, 203, 243, 72, 16, 31, 73, 204, 139, 73, 17, 182, 73, 67, 52, 71, 221, 9, 71, 34, 53, 71, 83, 179, 71, 152, 222, 71, 241, 183, 71, 54, 226, 71, 103, 96, 71, 172, 140, 72, 104, 248, 72, 173, 35, 71, 33, 157, 71, 121, 118, 71, 190, 162, 71, 239, 32, 71, 52, 75, 70, 141, 36, 70, 209, 79, 70, 3, 205, 70, 72, 249, 71, 4, 101, 76, 153, 120, 76, 221, 163, 76, 54, 124, 76, 123, 167, 76, 172, 37, 76, 241, 81, 76, 73, 42, 76, 142, 85, 76, 191, 211, 76, 4, 254, 75, 3, 103, 75, 52, 228, 75, 121, 16, 75, 210, 233, 75, 23, 20, 75, 72, 146, 75, 141, 189, 75, 229, 150, 75, 42, 194, 75, 91, 64, 74, 90, 168, 74, 159, 212, 74, 208, 81, 74, 21, 125, 74, 110, 86, 74, 178, 129, 74, 228, 255, 74, 41, 42, 74, 129, 3, 74, 198, 47, 74, 158, 60, 74, 246, 21, 74, 59, 64, 74, 108, 190, 74, 177, 234, 73, 9, 195, 73, 78, 238, 74, 128, 108, 73, 197, 151, 73, 29, 112, 126, 126, 200, 126, 175, 70, 126, 244, 113, 126, 77, 74, 126, 145, 118, 126, 195, 243, 126, 8, 31, 126, 96, 248, 126, 165, 35, 126, 214, 161, 125, 213, 10, 125, 26, 53, 126, 75, 179, 126, 144, 222, 125, 233, 183, 125, 45, 227, 125, 95, 96, 125, 164, 140, 125, 252, 101, 125, 65, 144, 125, 25, 157, 125, 113, 118, 125, 182, 162, 125, 231, 32, 125, 44, 75, 125, 132, 36, 125, 201, 79, 125, 251, 205, 125, 64, 249, 125, 152, 210, 130, 144, 120, 130, 213, 163, 130, 46, 124, 130, 114, 167, 130, 164, 37, 130, 233, 81, 130, 65, 42, 130, 134, 85, 130, 183, 211, 130, 252, 254, 129, 251, 103, 129, 44, 229, 129, 113, 16, 129, 201, 233, 129, 14, 20, 129, 64, 146, 129, 133, 190, 129, 221, 151, 129, 34, 194, 129, 83, 64, 129, 82, 168, 128, 151, 212, 129, 200, 81, 129, 13, 125, 128, 101, 86, 128, 170, 129, 128, 220, 255, 128, 33, 42, 128, 121, 3, 128, 190, 47, 129, 250, 207, 128, 238, 21, 128, 51, 65, 128, 100, 190, 128, 169, 234, 128, 1, 195, 128, 70, 238, 128, 120, 108, 128, 188, 151, 128, 21, 112, 128, 81, 17, 128, 149, 60, 133, 170, 27, 133, 239, 70, 133, 33, 196, 133, 102, 239, 133, 190, 200, 133, 3, 244, 133, 52, 114, 133, 121, 157, 133, 220, 153, 133, 13, 22, 133, 82, 66, 132, 70, 136, 132, 139, 179, 132, 189, 49, 132, 1, 92, 132, 90, 53, 132, 159, 97, 132, 208, 223, 132, 51, 218, 132, 120, 6, 132, 169, 131, 132, 238, 175, 131, 226, 245, 131, 39, 32, 132, 89, 158, 131, 157, 201, 131, 246, 162, 131, 59, 206, 192, 223, 252, 192, 36, 40, 192, 125, 1, 192, 193, 44, 192, 243, 170, 192, 56, 213, 192, 144, 174, 192, 213, 218, 192, 6, 87, 192, 75, 131, 197, 107, 132, 197, 156, 2, 197, 225, 45, 197, 57, 6, 197, 126, 50, 197, 175, 175, 197, 244, 219, 197, 77, 180, 197, 146, 223, 197, 195, 93, 197, 194, 197, 197, 6, 241, 197, 56, 111, 197, 125, 154, 197, 213, 115, 197, 26, 158, 197, 75, 28, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train_sparse_embedding', 'tune_overlap', 'train_embedding', 'tune_sparse'")
parser.add_argument('--ds_dir', default='/work/scratch/chen/Datasets/PNS-Cyst/patches_train_matting', help='ds_dir')
parser.add_argument('--ds_split', default='all', help='ds_split')
parser.add_argument('--test_negative', default=1, help='whether include negative examples or not')
parser.add_argument('--model_dir', default='./models_cyst_objAug/model_ce_pos', help='model_dir')
parser.add_argument('--mode', default='foreground', help='model_dir')
parser.add_argument('--epoches', default=300, help='training epoches')
parser.add_argument('--loss', default='ce', help='training loss')
parser.add_argument('--backbone', default='unet', help='')
parser.add_argument('--augmentation', default=1, help='')

parser.add_argument('--focal_loss_gamma', default=2, help='')
args = parser.parse_args()

model_dir = args.model_dir
sz = (512, 512)

# utils
def extract_patches(img, mask, sz, min_obj_size=0):
    # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[...,0:3]
    # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    patches_img, patches_mask = [], []
    patches_img_neg, patches_mask_neg = [], []

    patches_img_ = instSeg.split(img, (sz[0], sz[1]), (0,0), remainder='same')['patches']
    patches_mask_ = instSeg.split(mask, (sz[0], sz[1]), (0,0), remainder='same')['patches']
    for img_p, mask_p in zip(patches_img_, patches_mask_):
        assert img_p['position'][0] == mask_p['position'][0] and img_p['position'][1] == mask_p['position'][1]
        if np.sum(mask_p['data']>0) >= min_obj_size:
            patches_img.append(img_p['data'])
            patches_mask.append(mask_p['data'])
        else:
            patches_img_neg.append(img_p['data'])
            patches_mask_neg.append(mask_p['data'])

    return patches_img, patches_mask, patches_img_neg, patches_mask_neg

if args.command == 'train':
    ## basic setting
    config = instSeg.ConfigParallel(image_channel=3)
    config.H = sz[0]
    config.W = sz[1]
    config.backbone = args.backbone
    config.filters = 64
    config.nstage = 4
    if args.mode == 'foreground':
        config.modules = ['foreground']
    else:
        config.modules = ['foreground', 'contour']
    if args.loss == 'ce':
        config.loss['foreground'] = 'crossentropy' 
        config.loss['contour'] = 'crossentropy' 
    if args.loss == 'ce_random':
        config.loss['foreground'] = 'crossentropy_random' 
        config.loss['contour'] = 'crossentropy_random' 
    if args.loss == 'ce_hard':
        config.loss['foreground'] = 'crossentropy_hard' 
        config.loss['contour'] = 'crossentropy_hard' 
    if args.loss == 'dice':
        config.loss['foreground'] = 'dice' 
        config.loss['contour'] = 'dice' 
    if args.loss == 'ce_weighted':
        config.loss['foreground'] = 'crossentropy_weighted' 
        config.loss['contour'] = 'crossentropy_weighted' 
    if args.loss == 'focal_loss':
        config.loss['foreground'] = 'focal_loss' 
        config.loss['contour'] = 'focal_loss' 
    
    config.focal_loss_gamma = float(args.focal_loss_gamma)
    
    config.save_best_metric == 'AP'
    # config.save_best_metric == 'Dice'
    config.snapshots = [100, 200]
    config.validation_start_epoch = 1


    config.train_learning_rate = 1e-4
    config.lr_decay_rate = 0.9
    config.lr_decay_period = 10000
    config.input_normalization = 'per-image'
    config.net_upsample = 'deConv' # 'upConv', 'deConv'
    config.net_normalization = 'batch'
    config.dropout_rate = 0

    config.flip = True
    config.random_rotation = True
    config.random_gamma = True
    config.random_blur = False

    min_obj_size = 500
    config.obj_min_size = min_obj_size


    ## training data setting
    epoches = int(args.epoches)
    ds_dir = args.ds_dir
    use_all = False # if no test is needed, include the test data as training data 

    ## load dataset
    # images, masks = read_pns_cyst(ds_dir, part='part1', split='train', num_read=None)
    X_train, y_train = [], []
    images, masks = read_patches(ds_dir, split=args.ds_split, num_read=10)
    for k, img in images.items():
        X_train.append(img)
        y_train.append(masks[k])
    
    idx = 0
    for img, m in zip(X_train, y_train):
        cv2.imwrite("./img_{}.png".format(idx), img)
        cv2.imwrite("./mask_{}.png".format(idx), ((m>0)*255).astype(np.uint8))
        idx += 1

    # idx = list(range(len(X_train)))
    # random.shuffle(idx)
    # val_count = int(0.1*len(X_train))
    # X_val, y_val = [X_train[i] for i in idx[:val_count]], [y_train[i] for i in idx[:val_count]]
    # X_train, y_train = [X_train[i] for i in idx[val_count:]], [y_train[i] for i in idx[val_count:]]

    # X_train = np.array(X_train)
    # y_train = np.expand_dims(np.array(y_train), axis=-1)
    # print(X_train.shape, y_train.shape)
    # X_val = np.array(X_val)
    # y_val = np.expand_dims(np.array(y_val), axis=-1)
    # print(X_val.shape, y_val.shape)


    # train_ds = {'image': X_train, 'instance': y_train}
    # val_ds = {'image': X_val, 'instance': y_val}

    # # # # create model and train
    # keras.backend.clear_session()
    # model = instSeg.InstSegParallel(config=config, model_dir=model_dir)
    # model.train(train_ds, val_ds, batch_size=4, epochs=epoches, augmentation=(int(args.augmentation)==1))

if args.command == 'predict':
    model = instSeg.load_model(model_dir=model_dir)
    # img_path = "/work/scratch/chen/Datasets/Nema-Cyst/Part-2/test/RWE 2020 02_004_TopRGB_2020-09-16T10-06-42.png"
    img_path = "/work/scratch/chen/Datasets/PNS-Cyst/PNS-Cyst-part1/PNS-Cyst-Part1-Images/image_004.png"
    img = cv2.imread(img_path)[:,:,:3]
    # instance = instSeg.seg_in_tessellation(model, img, patch_sz=[512,512], margin=[128,128], overlap=[0,0], mode='bi', min_obj_size=model.config.obj_min_size)
    # instance = relabel(instance).astype(np.uint8)
    # instance_save = Image.fromarray(instance, mode='P')
    # instance_save.putpalette(palette)
    # instance_save.save('./test.png')

    mode = 'foreground' if args.mode == 'foreground' else 'bi'
    fg = instSeg.seg_in_tessellation(model, img, patch_sz=[512,512], margin=[128,128], overlap=[0,0], mode=mode, min_obj_size=model.config.obj_min_size)
    cv2.imwrite('./test.png', (fg*255).astype(np.uint8))


if args.command == 'evaluate':
    import os
    import csv
    f = open('/work/scratch/chen/instSeg/models_cyst_objAug/test.csv', 'w')
    writer = csv.writer(f)

    models = ['/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_all',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_pos',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_copy_all',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_copy_pos',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_copy_neg',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_matting_all',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_matting_pos',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_matting_neg',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_patch_all',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_patch_pos',
             '/work/scratch/chen/instSeg/models_cyst_objAug/model_ce_patch_neg']
    tolerence = [0, 1, 2, 3]
    for model_dir in models:
        writer.writerow([os.path.basename(model_dir)] + tolerence)
        model = instSeg.load_model(model_dir=model_dir, load_best=True)
        

        ds_dir = '/images/ACTIVE/2018_PheNeSens/PNS-Cyst/PNS-Cyst-part1'
        images, masks = read_pns_cyst(ds_dir, split='test', num_read=10)
        pred_patches, gt_patches = [], []
        for k, img in images.items():
            img_patches, mask_patches, imgs_neg_patches, mask_neg_patches = extract_patches(img, masks[k], sz=sz, min_obj_size=100)
            if int(args.test_negative) > 0:
                img_patches = img_patches + imgs_neg_patches
                mask_patches = mask_patches + mask_neg_patches
            for img_p, mask_p in zip(img_patches, mask_patches):
                pred_p = model.predict_raw(img_p)
                if args.mode == 'foreground':
                    pred_patches.append(pred_p['foreground']>0.5)
                    gt_patches.append(mask_p)
        
        JI, P, R = [], [], []
        for T in tolerence:
            evaluator = instSeg.Evaluator_Seg(tolerance=T) if args.mode == 'foreground' else instSeg.Evaluator(dimension=2, mode='area')
            for p, g in zip(pred_patches, gt_patches):
                evaluator.add_example(p, g)
        
            JI.append(evaluator.JI())
            P.append(evaluator.precision())
            R.append(evaluator.recall())
        writer.writerow(['JI'] + JI)
        writer.writerow(['Precision'] + P)
        writer.writerow(['Recall'] + R)
    
    f.close()



