import instSeg
import numpy as np
import os
import cv2
from random import shuffle
from tensorflow import keras
from sklearn.decomposition import PCA
from skimage.morphology import binary_closing, square
import shutil
from data_loader import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='bbbc010', help='experiment dataset')
parser.add_argument('--net', default='layered', help='experiment architecture')
args = parser.parse_args()

pd_dir = './prediction_'+args.ds.lower()
if not os.path.exists(pd_dir):
    os.makedirs(pd_dir)

if args.ds.lower() == 'bbbc010':
    sz = (448, 448)
    obj_min_size = 500
if args.ds.lower() == 'occ2014':
    sz = (320, 320)
    obj_min_size = 500
if args.ds.lower() == 'ccdb6843':
    sz = (512, 512)
    obj_min_size = 500

imgs, masks = load_data_test(args.ds.lower(), sz)

predictions = {}
predictions_layers = {}
# ## unet2
# if args.net.lower() in ['unet2', 'unet3']:
#     model_dir = './model_'+args.net.lower()+'_'+args.ds.lower()
#     model = instSeg.load_model(model_dir, load_best=True)
#     model.config.obj_min_size = obj_min_size
#     for k, img in imgs.items():
#         print('segmenting: ', k)
#         instances = model.predict(img)[0]
#         predictions[k] = instances

# if args.net.lower() in ['layered']:
#     # model_dir = './model_'+args.net.lower()+'_'+args.ds.lower()
#     # model_dir = './model_embedding/model_embedding200'+'_'+args.ds.lower()
#     # model_dir = 'model_LayeredParallelS_' + args.ds.lower()
#     model_dir = './model_base/bbbc010'
#     model = instSeg.load_model(model_dir, load_best=True)
#     model.config.obj_min_size = obj_min_size
#     for k, img in imgs.items():
#         print('segmenting: ', k)
#         instances, raw = model.predict(img)
#         instances = np.moveaxis(instances, -1, 0)
#         predictions[k] = instances
#         predictions_layers[k] = raw['embedding'] > 0.5

# mrcnn
if args.net.lower() == 'mrcnn':
    if args.ds.lower() == 'bbbc010':
        dir_mrcnn = '/images/innoretvision/MICCAI2022/BBBC010/pred/'
        for k, img in imgs.items():
            M = []
            for f in os.listdir(os.path.join(dir_mrcnn,k)):
                m = cv2.imread(os.path.join(dir_mrcnn,k,f), cv2.IMREAD_UNCHANGED)
                m = cv2.resize(m, (masks[k][0].shape[1], masks[k][0].shape[0]), interpolation=cv2.INTER_NEAREST)
                M.append(m)
            predictions[k] = M
    if args.ds.lower() == 'occ2014':
        dir_mrcnn = '/images/innoretvision/MICCAI2022/OCC2014/pred/'
        for k, img in imgs.items():
            f = 'cytoplasm'+str(k)
            print('reading: ', f)
            M = []
            for f_m in os.listdir(os.path.join(dir_mrcnn,f)):
                m = cv2.imread(os.path.join(dir_mrcnn,f,f_m), cv2.IMREAD_UNCHANGED)
                m = cv2.resize(m, (masks[k][0].shape[1], masks[k][0].shape[0]), interpolation=cv2.INTER_NEAREST)
                M.append(m)
            predictions[k] = M
    if args.ds.lower() == 'ccdb6843':
        dir_mrcnn = '/images/innoretvision/MICCAI2022/CCDB6843/pred/'
        for k, img in imgs.items():
            f = k+'w1'
            print('reading: ', f)
            M = []
            for f_m in os.listdir(os.path.join(dir_mrcnn,f)):
                m = cv2.imread(os.path.join(dir_mrcnn,f,f_m), cv2.IMREAD_UNCHANGED)
                m = cv2.resize(m, (masks[k][0].shape[1], masks[k][0].shape[0]), interpolation=cv2.INTER_NEAREST)
                M.append(m)
            predictions[k] = M


# stardist
if args.net.lower() == 'stardist':
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    model_dir = './model_stardist'+'_'+args.ds.lower()
    model = StarDist2D(None, name='stardist', basedir=model_dir)
    for k, img in imgs.items():
        print('segmenting: ', k)
        img_n = normalize(img, 1, 99.8, axis=(0,1))
        instances, _ = model.predict_instances(img_n)
        print(instances.shape)
        predictions[k] = instances

## save visulization

# save_dir = os.path.join(pd_dir, args.net.lower())
# if os.path.exists(save_dir):
#     shutil.rmtree(save_dir)
# for k, masks_pd in predictions.items():
#     example_dir = os.path.join(save_dir, k)
#     os.makedirs(example_dir)
#     # # save predicted mask
#     # for i in range(masks_pd.shape[0]):
#     #     cv2.imwrite(os.path.join(example_dir, 'instance_'+str(i)+'.png'), np.uint8(masks_pd[i]>0)*255)
#     if len(predictions_layers) > 0:
#         layers = predictions_layers[k]
#         for i in range(layers.shape[-1]):
#             cv2.imwrite(os.path.join(example_dir, 'layer_'+str(i)+'.png'), np.uint8(layers[...,i]>0)*255)
#     # save visualization of predicted mask
#     vis_pd = instSeg.vis.vis_instance_contour(imgs[k], masks_pd)
#     cv2.imwrite(os.path.join(example_dir, 'pred.png'), vis_pd)
#     # save visualization of ground truth
#     vis_gt = instSeg.vis.vis_instance_contour(imgs[k], np.array(masks[k]))
#     cv2.imwrite(os.path.join(example_dir, 'gt.png'), vis_gt)

## evaluation
thres = [0.5, 0.6, 0.7, 0.8, 0.9]
# thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
e = instSeg.Evaluator(dimension=2, mode='area', verbose=True)
for k, masks_pd in predictions.items():
    e.add_example(masks_pd, masks[k])
print("Evaluation of ", args.net.lower(), 'on data set', args.ds.lower(), ': ')
e.AP_DSB(thres=thres)
e.AJI()
for t in thres:
    r = e.detectionRecall(t, metric='dice')
    p = e.detectionPrecision(t, metric='dice')
    # f1 = 2*p*r/(p+r)
    # print('(thres '+str(t)+') precision: ', p, 'recall: ', r, 'f1: ', f1)


