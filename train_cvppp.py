
import instSeg
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import label
from instSeg.post_process import *
import os

config = instSeg.Config(semantic=False, dist=True, embedding=True)
config.module_order = ['dist', 'semantic', 'embedding']
config.feature_forward_dimension = 32
run_name = 'cvppp_dft_stagewise'
ds_dir = './ds_cvppp'
base_dir = './'

#### cross validation ####
for fold in range(5):
    # model = instSeg.InstSeg_Cascade(config=config, base_dir=base_dir, run_name=run_name + '_crossval_'+str(fold))
    model = instSeg.InstSeg_Mul(config=config, base_dir=base_dir, run_name=run_name + '_crossval_'+str(fold))
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    with open(os.path.join(ds_dir, 'crossval_partition.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if int(row[0]) == fold:
                X_val.append(os.path.join(ds_dir, row[1][2:]))
                Y_val.append(os.path.join(ds_dir, row[2][2:]))
            else:
                X_train.append(os.path.join(ds_dir, row[1][2:]))
                Y_train.append(os.path.join(ds_dir, row[2][2:]))

    train_data = {'image': list(map(cv2.imread,X_train)),
                'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_train))}
    val_data = {'image': list(map(cv2.imread,X_val)),
                'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_val))}

    # train_data = {'image': list(map(cv2.imread,X_train))[:40],
    #             'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_train))[:40]}
    # val_data = {'image': list(map(cv2.imread,X_val))[:40],
    #             'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y_val))[:40]}

    # model.train(train_data, val_data, batch_size=4, epochs=300, augmentation=False)
    model.stagewise_train(train_data, val_data, batch_size=4, epochs=50, augmentation=False)
    # model.save_model(stage_wise=True, save_best=False)

#### training ####

# dirs = ['A1', 'A2', 'A3', 'A4']

# X, Y = [], []
# for d in dirs:
#     X = X + sorted(glob.glob('./ds_cvppp/training_images/'+d+'/*.png'))
#     Y = Y + sorted(glob.glob('./ds_cvppp/training_truth/'+d+'/*.png'))
# train_data = {'image': list(map(cv2.imread,X)),
#               'object': list(map(lambda img: cv2.imread(img, cv2.IMREAD_GRAYSCALE),Y))}

# model.train(train_data, val_data, batch_size=4, epochs=500, augmentation=False)

#### test ####

# use_gt_fg = True
# model.load_weights()

# save_dir= './ds_cvppp/testing_seg/'
# vis_dir= './ds_cvppp/testing_vis/'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# if not os.path.exists(vis_dir):
#     os.mkdir(vis_dir)

# dirs = ['A1', 'A2', 'A3', 'A4', 'A5']
# # dirs = ['A2', 'A3']

# for d in dirs:
    
#     seg_save_dir = os.path.join(save_dir, d)
#     vis_save_dir = os.path.join(vis_dir, d)
#     if not os.path.exists(seg_save_dir):
#         os.mkdir(seg_save_dir)
#     if not os.path.exists(vis_save_dir):
#         os.mkdir(vis_save_dir)
    
#     files = sorted(glob.glob('./ds_cvppp/testing_images/'+d+'/*.png'))
#     if use_gt_fg:
#         files_fg = sorted(glob.glob('./ds_cvppp/testing_images_fg/'+d+'/*.png'))
    
#     for i in range(len(files)):
#         print(files[i])
#         img = cv2.imread(files[i])
#         if use_gt_fg:
#             fg = cv2.imread(files_fg[i], cv2.IMREAD_GRAYSCALE)
#             seg, pred = model.predict(img, thres=0.7, similarity_thres=0.7, semantic_mask=fg>0)
#         else:
#             seg, pred = model.predict(img, thres=0.7, similarity_thres=0.7)
#         seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
#         if np.sum(seg) == 0:
#             print('empty')
#             seg[0,0]=1
#         cv2.imwrite(os.path.join(seg_save_dir, os.path.basename(files[i])), seg)

#         vis = label2rgb(seg, bg_label=0)
#         vis = vis * 255 / vis.max()
#         cv2.imwrite(os.path.join(vis_save_dir, os.path.basename(files[i])), vis)

#         # cv2.imwrite(os.path.join(vis_save_dir, os.path.basename(files[i])[:-4]+'_dist.png'), pred['dist']*25)
#         # emb = pred['embedding'][:,:,0:3]
#         # emb = (emb - emb.min())/(emb.max()-emb.min()) * 255
#         # cv2.imwrite(os.path.join(vis_save_dir, os.path.basename(files[i])[:-4]+'_emb.png'), vis.astype(np.uint8))

# # plt.subplot(1,3,1)
# # plt.imshow(img)
# # plt.subplot(1,3,2)
# # plt.imshow(np.squeeze(np.argmax(pred['semantic'], axis=-1))*255)
# # plt.subplot(1,3,3)
# # plt.imshow(label2rgb(seg, bg_label=0))
# # # plt.imshow(seg)
# # plt.show()
