import instSeg
from skimage.io import imread, imsave
import numpy as np
import os
import csv
import time
import cv2
from random import shuffle

config = instSeg.ConfigParallel()
config.loss_semantic = 'dice' 
config.loss_contour = 'dice'
config.contour_radius = 2
config.train_learning_rate = 1e-5
config.backbone = 'uNet'
config.filters = 32
run_name = 'model_DCAN_cyst'
base_dir = './'
splits = [0]

config.train_batch_size = 3
supepoches = 20
image_load_split = 20
subepoches = 5

ds_dir = 'D:/Datasets/PheNeSensCyst/cysts_split'
ds_csv = os.path.join(ds_dir, 'patches.csv')

# training 
for s in splits:
    # load dataset
    with open(ds_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        X_train, y_train = [], []
        X_val, y_val = [], []
        for row in csv_reader:
            if int(row[0]) == s:
                X_val.append(os.path.join(ds_dir, row[1]))
                y_val.append(os.path.join(ds_dir, row[2]))
            else:
                X_train.append(os.path.join(ds_dir, row[1]))
                y_train.append(os.path.join(ds_dir, row[2]))
    
    # X_train = X_train[:5000]
    # y_train = y_train[:5000]
    # X_val = X_val[:500]
    # y_val = y_val[:500]


    epoch = 0
    while epoch < supepoches * image_load_split * subepoches:

        index_train = list(range(len(X_train)))
        shuffle(index_train)
        X_train = [X_train[i] for i in index_train]
        y_train = [y_train[i] for i in index_train]
        index_val = list(range(len(X_val)))
        shuffle(index_val)
        X_val = [X_val[i] for i in index_val]
        y_val = [y_val[i] for i in index_val]

        for ss in range(image_load_split):

            X_train_ee = [example for idx, example in enumerate(X_train) if idx % image_load_split == ss]
            y_train_ee = [example for idx, example in enumerate(y_train) if idx % image_load_split == ss]
            X_val_ee = [example for idx, example in enumerate(X_val) if idx % image_load_split == ss]
            y_val_ee = [example for idx, example in enumerate(y_val) if idx % image_load_split == ss]

            train_ds = {'image': np.array(list(map(imread,X_train_ee))),
                        'instance': np.expand_dims(np.array(list(map(imread,y_train_ee))), axis=-1)}
            val_ds = {'image': np.array(list(map(imread,X_val_ee))),
                      'instance': np.expand_dims(np.array(list(map(imread,y_val_ee))), axis=-1)}
            print(train_ds['image'].shape, val_ds['image'].shape)
            # create model and train

            model = instSeg.InstSegParallel(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
            model.train(train_ds, val_ds, batch_size=4, epochs=epoch+subepoches, augmentation=False)
            epoch = model.training_epoch


# evalutation

# eval_dir = 'eval'+run_name[5:]
# if not os.path.exists(eval_dir):
#     os.makedirs(eval_dir)

# e = instSeg.Evaluator(dimension=2, mode='area', verbose=False)
# model_load_time = 0
# image_load_time = 0
# image_seg_time = 0

# with open(ds_csv) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=';')
#     idx_spilt, img_path, gt_path = [], [], []
#     for row in csv_reader:
#         idx_spilt.append(int(row[0]))
#         img_path.append(row[1])
#         gt_path.append(row[2])

# t = time.time()

# for s in splits:
    
#     tt = time.time()
#     model = instSeg.InstSegParallel(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
#     model.load_weights(load_best=True)
#     model_load_time += (time.time()-tt)

#     for i, ss in enumerate(idx_spilt):
#         if ss != s:
#             continue
#         print(i, img_path[i])
#         # load image
#         tt = time.time()
#         img = imread(img_path[i])
#         image_load_time += (time.time()-tt)
#         gt = imread(gt_path[i])
#         # process
#         tt = time.time()
#         instance, raw = model.predict(img)
#         image_seg_time += (time.time()-tt)
#         # eval
#         e.add_example(instance, gt)
#         # vis instance
#         vis = instSeg.vis.vis_instance_area(img, instance)
#         vis = instSeg.vis.vis_instance_contour(vis, gt, color='red')
#         imsave(os.path.join(eval_dir, 'example_{:04d}.png'.format(i)), vis)

#         semantic = np.squeeze(np.argmax(raw[0],-1))
#         semantic = cv2.resize(semantic, (1240, 1000), interpolation=cv2.INTER_NEAREST)
#         contour = np.squeeze(raw[1]>0.5).astype(np.int16)
#         contour = cv2.resize(contour, (1240, 1000), interpolation=cv2.INTER_NEAREST)

#         vis = instSeg.vis.vis_semantic_area(img, semantic, colors=['blue'])
#         imsave(os.path.join(eval_dir, 'example_{:04d}_seg.png'.format(i)), vis)

#         vis = instSeg.vis.vis_semantic_area(img, contour, colors=['blue'])
#         imsave(os.path.join(eval_dir, 'example_{:04d}_contour.png'.format(i)), vis)


# with open(os.path.join(eval_dir, "eval_"+run_name+".txt"), "w") as f:
#     f.write("model loading time {:} \r\n".format(model_load_time/len(splits)))
#     f.write("image loading time {:} \r\n".format(image_load_time/len(img_path)))
#     f.write("image processing time {:} \r\n".format(image_seg_time/len(img_path)))
#     f.write("mAP {:} \r\n".format(e.mAP()))
#     f.write("AP {:} \r\n".format(e.AP()))
#     f.write("mP 0.5thres: {:}, 0.6thres: {:}, 0.7thres: {:}, 0.8thres: {:}, 0.9thres: {:} \r\n".format(e.P(0.5), e.P(0.6), e.P(0.7), e.P(0.8), e.P(0.9)))
#     f.write("detection Sensitivity 0.5thres: {:}, 0.6thres: {:}, 0.7thres: {:}, 0.8thres: {:}, 0.9thres: {:} \r\n".format(e.detectionSensitivity(0.5), e.detectionSensitivity(0.6), e.detectionSensitivity(0.7), e.detectionSensitivity(0.8), e.detectionSensitivity(0.9)))
#     f.write("detection Accuracy 0.5thres: {:}, 0.6thres: {:}, 0.7thres: {:}, 0.8thres: {:}, 0.9thres: {:} \r\n".format(e.detectionAccuracy(0.5), e.detectionAccuracy(0.6), e.detectionAccuracy(0.7), e.detectionAccuracy(0.8), e.detectionAccuracy(0.9)))
#     f.write("mAJ {:} \r\n".format(e.mAJ()))
#     f.write("AJ {:} \r\n".format(e.aggregatedJaccard()))
        
