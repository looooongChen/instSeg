
import instSeg
from skimage.io import imsave
import cv2
import numpy as np
import os
import csv
import time

config = instSeg.ConfigCascade()
config.modules = ['embedding']
config.validation_start_epoch = 150
config.loss_embedding = 'cos'
config.filters = 32
run_name = 'model_EMBl2_cvppp'
base_dir = './'
splits = [0]

ds_csv = 'D:/instSeg/ds_cvppp/cvppp.csv'
ds_dir = os.path.dirname(ds_csv)

# training 
for s in splits:
    # load dataset
    with open(ds_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        X_train, y_train = [], []
        X_val, y_val = [], []
        for row in csv_reader:
            if int(row[0]) == s:
                X_val.append(os.path.join(ds_dir,row[1]))
                y_val.append(os.path.join(ds_dir,row[2]))
            else:
                X_train.append(os.path.join(ds_dir,row[1]))
                y_train.append(os.path.join(ds_dir,row[2]))
    

    imgs_train, gts_train = [], []
    for i in range(len(X_train)):
        img = cv2.imread(X_train[i], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (config.W, config.H), interpolation=cv2.INTER_LINEAR)
        imgs_train.append(img)
        gt = cv2.imread(y_train[i], cv2.IMREAD_UNCHANGED)
        gt = cv2.resize(gt, (config.W, config.H), interpolation=cv2.INTER_NEAREST)
        gts_train.append(gt)
    imgs_train, gts_train = np.array(imgs_train), np.array(gts_train)
    train_ds = {'image': imgs_train,
                'instance': gts_train}
    
    # imgs_val, gts_val = [], []
    # for i in range(len(X_val)):
    #     img = cv2.imread(X_val[i], cv2.IMREAD_UNCHANGED)
    #     img = cv2.resize(img, (config.W, config.H), interpolation=cv2.INTER_LINEAR)
    #     imgs_val.append(img)
    #     gt = cv2.imread(y_val[i], cv2.IMREAD_UNCHANGED)
    #     gt = cv2.resize(gt, (config.W, config.H), interpolation=cv2.INTER_NEAREST)
    #     gts_val.append(gt)
    # imgs_val, gts_val = np.array(imgs_val), np.array(gts_val)
    # val_ds = {'image': imgs_val,
    #           'instance': gts_val}
    
    # create model and train
    model = instSeg.InstSegCascade(config=config, base_dir=base_dir, run_name=run_name+'_'+str(s))
    # model.train(train_ds, val_ds, batch_size=4, epochs=300, augmentation=False)
    model.train(train_ds, batch_size=4, epochs=300, augmentation=False)


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
#     model.load_weights(load_best=False)
#     model_load_time += (time.time()-tt)

#     for i, ss in enumerate(idx_spilt):
#         if ss != s:
#             continue
#         print(i, img_path[i])
#         # load image
#         tt = time.time()
#         img = cv2.imread(os.path.join(ds_dir,img_path[i]), cv2.IMREAD_UNCHANGED)
#         image_load_time += (time.time()-tt)
#         gt = cv2.imread(os.path.join(ds_dir,gt_path[i]), cv2.IMREAD_UNCHANGED)
#         # process
#         tt = time.time()
#         instance, raw = model.predict(img)
#         image_seg_time += (time.time()-tt)
#         # eval
#         # e.add_example(instance, gt)
#         # vis instance
#         # vis = instSeg.vis.vis_instance_area(img, instance)
#         # vis = instSeg.vis.vis_instance_contour(vis, gt, color='red')
#         # imsave(os.path.join(eval_dir, 'example_{:04d}.png'.format(i)), vis)
#         print(raw[0].shape, raw[1].shape)
#         imsave(os.path.join(eval_dir, 'img_{:04d}.png'.format(i)), img)
#         imsave(os.path.join(eval_dir, 'dist_{:04d}.png'.format(i)), raw[1][0,:,:,0])
#         imsave(os.path.join(eval_dir, 'dist_t_{:04d}.png'.format(i)), np.array(raw[1][0,:,:,0])>0.5)
#         imsave(os.path.join(eval_dir, 'inst_{:04d}.png'.format(i)), instance)
#         instance2 = np.squeeze(np.argmax(raw[0], axis=-1))
#         imsave(os.path.join(eval_dir, 'inst2_{:04d}.png'.format(i)), instance2)
#         break


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
        
