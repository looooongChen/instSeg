import instSeg
import os
import openslide 
import numpy as np
import cv2
from skimage.morphology import dilation, disk
from skimage.measure import regionprops
import csv
from PIL import Image
from skimage.segmentation import mark_boundaries

eval_dir = './eval_Endometrium'

img_dir = '/images/ACTIVE/2020_Endometrium/ds_endometrium/images'
gt_dir = '/images/ACTIVE/2020_Endometrium/ds_endometrium/masks'
cv_list = '/work/scratch/chen/Datasets/Endometrium/patches_gland/files.csv'

model_dir = './model_endometrium/model_Endometrium'
level = 3
min_size = 200


if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

with open(cv_list, newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')
    examples = [e for e in csv_reader]
fname_dict = {}
for e in examples:
    s = int(e[0])
    if s not in fname_dict:
        fname_dict[s]=e[1]
print(fname_dict)

with open(os.path.join(eval_dir, 'evaluation.csv'), mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['-', 'DICE', 0.5, 0.6, 0.7, 0.8, 0.9])
    model = instSeg.load_model(model_dir+'_0', load_best=True)
    for idx, fname in fname_dict.items():
        # fname = fname_dict[idx+1]
        # model = instSeg.load_model(model_dir+'_'+str(idx), load_best=True)
        csv_writer.writerow([fname])

        e = instSeg.evaluation.Evaluator(dimension=2, mode='area')

        gt = Image.open(os.path.join(gt_dir, fname+'.png'))
        gt = np.array(gt)
        slide = openslide.OpenSlide(os.path.join(img_dir, fname+'.ndpi'))
        img = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))
        
        rr, cc = np.nonzero(gt)
        R_min, R_max = max(int(np.min(rr))-model.config.H, 0), min(int(np.max(rr))+model.config.H, img.shape[0])
        C_min, C_max = max(int(np.min(cc))-model.config.W, 0), min(int(np.max(cc))+model.config.W, img.shape[1])
        if R_max - R_min < 512:
            D = R_max - R_min
            R_min = R_min - D//2
            R_max = R_max + (D - D//2)
        if C_max - C_min < 512:
            D = C_max - C_min
            C_min = C_min - D//2
            C_max = C_max + (D - D//2)

        gt = gt[R_min:R_max, C_min:C_max]
        img = img[R_min:R_max, C_min:C_max, 0:3][:,:,::-1]

        instance = instSeg.seg_in_tessellation(model, img, patch_sz=[512,512], margin=[64,64], overlap=[64,64], mode='lst')
        for p in regionprops(instance):
            if p.area < min_size:
                instance[p.coords[:,0], p.coords[:,1]] = 0

        vis = mark_boundaries(img, gt, color=(0, 0, 1))
        vis = mark_boundaries(vis, instance, color=(0, 1, 1))
        cv2.imwrite(os.path.join(eval_dir,fname+'.tif'), (vis*255).astype(np.uint8))

        e.add_example(instance, gt)
        precision = [e.detectionPrecision(t, metric='dice') for t in [0.5,0.6,0.7,0.8,0.9]]
        csv_writer.writerow(['', 'Precision']+precision)
        recall = [e.detectionRecall(t, metric='dice') for t in [0.5,0.6,0.7,0.8,0.9]]
        csv_writer.writerow(['', 'Recall']+recall)
        csv_writer.writerow(['', 'mAP']+[e.AP_DSB(thres=[0.5,0.6,0.7,0.8,0.9], metric='dice')])
        csv_writer.writerow(['', 'mAJ']+[e.AJI()])


