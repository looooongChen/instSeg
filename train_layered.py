import instSeg
import numpy as np
import os
from datasets import Datasets, labeled_non_overlap
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train_sparse_embedding', 'tune_overlap', 'train_embedding', 'tune_sparse'")

parser.add_argument('--model_dir', default='/work/scratch/chen/models/models_release/model_Celegans_unetL', help='model_dir')
parser.add_argument('--steps', default=100000, help='training epoches')
parser.add_argument('--ds', default='Celegans', help='experiment dataset')
parser.add_argument('--backbone', default='unet', help='')
parser.add_argument('--unet_stage', default=5, help='more conv stages enlarge receptive field')
parser.add_argument('--D_embedding', default=8, help='')
parser.add_argument('--augmentation', default=1, help='')
# parser.add_argument('--net', default='unet2', help='experiment architecture')
args = parser.parse_args()


batch_size = 8
steps = int(args.steps)
if args.ds == 'Celegans':
    ds_repeat = 5
    H, W = 448, 448
elif args.ds == 'Cervical2014':
    ds_repeat = 1
    H, W = 320, 320
elif args.ds == 'Neuroblastoma':
    ds_repeat = 5
    H, W = 448, 448
obj_min_size = 250


# load dataset
ds = Datasets()
if args.command.startswith('train') or args.command.startswith('tune'):

    imgs_train, masks_train = ds.load_data(args.ds, split='train')
    keys_train = list(imgs_train.keys())
    imgs_train = [imgs_train[k] for k in keys_train]
    masks_train = [masks_train[k] for k in keys_train]

    imgs_val, masks_val = ds.load_data(args.ds, split='val')
    keys_val = list(imgs_val.keys())
    imgs_val = [imgs_val[k] for k in keys_val]
    masks_val = [masks_val[k] for k in keys_val]

    ds_train = {'image': imgs_train, 'instance': masks_train}
    ds_val = {'image': imgs_val, 'instance': masks_val}
elif args.command.startswith('detect') or args.command.startswith('evaluate'):
    imgs_test, masks_test = ds.load_data(args.ds, split='test')

if args.command.startswith('train') or args.command.startswith('tune'):
    epoches = int(steps / (len(imgs_train) * ds_repeat / batch_size))
    epoches = epoches // 2 if args.command == 'tune' else epoches


if args.command.startswith('train'):
    model = None
    if os.path.exists(args.model_dir):
        model = instSeg.load_model(model_dir=args.model_dir, load_best=False)
    if model is None:
        config = instSeg.Config(image_channel=1)
        config.post_processing = 'layered_embedding'
        config.H = H 
        config.W = W
        config.nstage = int(args.unet_stage)
        config.filters = 64
        config.modules = ['layered_embedding', 'foreground']
        if args.command == 'train_embedding':
            config.loss['layered_embedding'] = 'cos' 
        elif args.command == 'train_sparse_embedding':
            config.loss['layered_embedding'] = 'sparse_cos' 
        config.loss['foreground'] = 'crossentropy' 
        config.neighbor_distance = 15
        config.embedding_dim = int(args.D_embedding)
        config.embedding_include_bg = False

        config.ds_repeat = ds_repeat
        config.flip = True
        config.random_rotation = True
        config.random_rotation_p = 0.5
        config.random_shift = True
        config.random_shift_p = 0.5
        config.max_shift = 64 # maximal shift
        config.random_gamma = True
        config.random_gamma_p = 0.2
        config.random_blur = False
        config.random_blur_p = 0.1
        config.random_saturation = False
        config.random_saturation_p = 0.5
        config.random_hue = False
        config.random_hue_p = 0.5
        config.elastic_deform = True
        config.elastic_deform_p = 0.8
        config.deform_scale = [16,256]
        config.deform_strength = 0.25

        config.train_learning_rate = 1e-4
        config.lr_decay_rate = 0.9
        config.lr_decay_period = 10000
        config.backbone = args.backbone
        config.input_normalization = 'per-image'
        config.up_scaling = 'bilinear'
        config.net_normalization = 'batch'
        config.dropout_rate = 0

        config.obj_min_size = obj_min_size
        config.save_best_metric = 'loss'
        config.validation_start_epoch = 10

        # create model and train
        model = instSeg.Model(config=config, model_dir=args.model_dir)
    model.train(ds_train, ds_val, batch_size=batch_size, epochs=epoches, augmentation=(int(args.augmentation)==1))

if args.command.startswith('tune'):
    model = instSeg.load_model(model_dir=args.model_dir, load_best=True)
    model.config.validation_start_epoch = 1
    # if args.command == 'tune_sparse': 
    #     suffix = "_sparseTuned" 
    #     model.config.loss['layered_embedding'] = 'sparse_cos' 
    # if args.command == 'tune_overlap': 
    suffix = "_overlapTuned" 
    model.config.loss['layered_embedding'] = 'overlap' 
    model.config.save_best_metric = 'AP'
    if not args.model_dir.endswith(suffix):
        model.save_as(os.path.join(os.path.dirname(args.model_dir), os.path.basename(args.model_dir)+suffix))
        model.config.train_learning_rate = model.lr()
        model.reset_training()
    model.train(ds_train, ds_val, batch_size=batch_size, epochs=epoches, augmentation=(int(args.augmentation)==1))

if args.command == 'detect':
    import shutil
    import cv2

    model = instSeg.load_model(model_dir=args.model_dir, load_best=True)
    result_dir = os.path.join(args.model_dir, 'results')
    print(result_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    for k, img in imgs_test.items():
        print(k)
        if args.ds == 'Neuroblastoma':
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            imgs_test[k] = img
        raw = model.predict_raw(img)
        embedding = raw['layered_embedding'] if 'layered_embedding' in raw.keys() else raw['embedding']
        embedding = np.squeeze(embedding)
        instances, layered = model.postprocess(raw, post_processing='layered_embedding')
        # save vis of layering
        save_dir = os.path.join(result_dir, k)
        os.makedirs(save_dir)
        # prediction visualization
        vis = instSeg.vis.vis_instance_contour(imgs_test[k], np.array(instances))
        cv2.imwrite(os.path.join(save_dir, 'pred.png'), vis)
        # ground truth visualization
        vis = instSeg.vis.vis_instance_contour(imgs_test[k], np.array(masks_test[k]))
        cv2.imwrite(os.path.join(save_dir, 'gt.png'), vis)
        # foreground
        fg = np.squeeze(raw['foreground']) > 0.5 
        cv2.imwrite(os.path.join(save_dir, 'foreground.png'), np.uint8(fg*255))
        # binary layering results
        # for i in range(layered.shape[-1]):
        #     cv2.imwrite(os.path.join(save_dir, 'layering_binary_'+str(i)+'.png'), np.uint8(layered[:,:,i])*255)
        # layering results
        for i in range(embedding.shape[-1]):
            cv2.imwrite(os.path.join(save_dir, 'layering_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*255))
            cv2.imwrite(os.path.join(save_dir, 'layering_masked_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*fg*255))
            img_vis = img/img.max() * 255
            if len(img.shape) == 2:
                img_vis = np.stack((img_vis, img_vis, img_vis), axis=-1)
            mask = embedding[:,:,i]*fg*255
            mask = np.stack((mask*0, mask*0, mask), axis=-1)
            img_vis = img_vis * 0.7 + mask*0.3
            cv2.imwrite(os.path.join(save_dir, 'layering_vis_'+str(i)+'.png'), np.uint8(img_vis))
        layering_gt = np.copy(embedding)*0
        overlap_free = np.sum([m>0 for m in masks_test[k]], axis=0) == 1
        for m in masks_test[k]:
            rr,cc = np.nonzero(m*overlap_free)
            idx = np.argmax(np.sum(embedding[rr,cc,:], axis=0))
            rr,cc = np.nonzero(m)
            layering_gt[rr,cc,idx]=1
        for i in range(layering_gt.shape[-1]):
            img_vis = img/img.max() * 255
            if len(img.shape) == 2:
                img_vis = np.stack((img_vis, img_vis, img_vis), axis=-1)
            mask = layering_gt[:,:,i]*fg*255
            mask = np.stack((mask*0, mask*0, mask), axis=-1)
            img_vis = img_vis * 0.75 + mask*0.25
            cv2.imwrite(os.path.join(save_dir, 'layering_gt_vis_'+str(i)+'.png'), np.uint8(img_vis))
        # raw layering visualization

if args.command == 'evaluate':
    import cv2

    thres = [0.5, 0.6, 0.7, 0.8, 0.9]

    e = instSeg.Evaluator(dimension=2, mode='area', verbose=True)
    model = instSeg.load_model(model_dir=args.model_dir, load_best=True)
    for k, img in imgs_test.items():
        if args.ds == 'Neuroblastoma':
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        raw = model.predict_raw(img)
        instances, layered = model.postprocess(raw, post_processing='layered_embedding')
        e.add_example(instances, masks_test[k])
    print("Evaluation of ", args.model_dir.lower(), 'on data set', args.ds.lower(), ': ')
    e.AP_DSB(thres=thres)
    e.AJI()
    for t in thres:
        r = e.detectionRecall(t, metric='dice')
        p = e.detectionPrecision(t, metric='dice')
