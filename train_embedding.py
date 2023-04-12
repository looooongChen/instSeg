import instSeg
import numpy as np
import os
from datasets import Datasets, labeled_non_overlap, map2stack
import argparse
# import calendar
# import time

parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train', 'tune_overlap'")

parser.add_argument('--model_dir', default='/work/scratch/chen/modelZoo/', help='model_dir')

parser.add_argument('--optimizer', default='Adam', help='')
parser.add_argument('--steps', default=150000, help='training epoches')
parser.add_argument('--lr', default=0.0001, help='')
parser.add_argument('--lr_decay', default=1, help='')
parser.add_argument('--augmentation', default='geo', help='geometric transform augmentation')

parser.add_argument('--suffix', default='', help='model name suffix')
parser.add_argument('--ds', default='Celegans', help='experiment dataset')
parser.add_argument('--backbone', default='unet', help='')
parser.add_argument('--unet_stage', default=5, help='more conv stages enlarge receptive field')
parser.add_argument('--filters', default=32, help='number of filters')
parser.add_argument('--up_scaling', default='deConv', help='number of filters')
parser.add_argument('--D_embedding', default=8, help='')
parser.add_argument('--embedding_loss', default='cosine', help='')
parser.add_argument('--embedding_activation', default='sigmoid', help='')
parser.add_argument('--embedding_regularization', default=0, help='')
parser.add_argument('--dynamic_weighting', action=argparse.BooleanOptionalAction)
parser.add_argument('--overlap_free', action=argparse.BooleanOptionalAction)
parser.add_argument('--mode', default='normal', help='mode of the embedding')
parser.add_argument('--coord', action=argparse.BooleanOptionalAction, help='coord')
parser.add_argument('--margin', action=argparse.BooleanOptionalAction, help='embedding loss margin')

parser.add_argument('--globalCons', action=argparse.BooleanOptionalAction, help='global constraint')
parser.add_argument('--embeddingOnly', action=argparse.BooleanOptionalAction, help='coord')
parser.add_argument('--useGtForeground', action=argparse.BooleanOptionalAction, help='coord')

args = parser.parse_args()
dynamic_weighting = False if args.dynamic_weighting is None else True
coord = False if args.coord is None else True
margin = False if args.margin is None else True
embeddingOnly = False if args.embeddingOnly is None else True
useGtForeground = False if args.useGtForeground is None else True
allow_overlap = False if args.overlap_free is None else True
suffix = args.suffix

# current_GMT = time.gmtime()
# time_stamp = calendar.timegm(current_GMT)
# suffix = args.suffix + '_' + str(time_stamp)

batch_size = 4
obj_min_size = 200

# load dataset
ds = Datasets()
if args.command.startswith('train') or args.command.startswith('tune'):
    label_map = False if args.command.startswith('tune') else True

    imgs_train, masks_train, _ = ds.load_data(args.ds, split='train', label_map=label_map)
    keys_train = list(imgs_train.keys())
    imgs_train = [imgs_train[k] for k in keys_train]
    masks_train = [masks_train[k] for k in keys_train]

    imgs_val, masks_val, _ = ds.load_data(args.ds, split='val', label_map=label_map)
    keys_val = list(imgs_val.keys())
    imgs_val = [imgs_val[k] for k in keys_val]
    masks_val = [masks_val[k] for k in keys_val]

    ds_train = {'image': imgs_train, 'instance': masks_train}
    ds_val = {'image': imgs_val, 'instance': masks_val}
elif args.command.startswith('detect') or args.command.startswith('evaluate'):
    imgs_test, masks_test, _ = ds.load_data(args.ds, split='test')

# training length
steps = int(args.steps)
if args.ds == 'Celegans':
    ds_repeat = 20
    H, W = 448, 448
    image_channel = 1
    guide_function_period = [32, 64, 128]
elif args.ds == 'Cervical2014':
    ds_repeat = 1
    H, W = 320, 320
    image_channel = 1
elif args.ds == 'Neuroblastoma':
    ds_repeat = 20
    H, W = 512, 512
    image_channel = 1
elif args.ds == 'BreastCancerCell':
    ds_repeat = 20
    H, W = 512, 512
    image_channel = 1
    guide_function_period = [32, 64, 128]
elif args.ds == 'CVPPP':
    ds_repeat = 1
    H, W = 448, 448
    image_channel = 3
    guide_function_period = [32, 64, 128]
elif args.ds == 'U2OS':
    ds_repeat = 20
    H, W = 512, 512
    image_channel = 1
elif args.ds.startswith('PNS-Cyst'):
    ds_repeat = 1
    H, W = 480, 512
    image_channel = 3
elif args.ds == 'PanNuke':
    ds_repeat = 1
    H, W = 256, 256
    image_channel = 3

if args.command.startswith('train') or args.command.startswith('tune'):
    epoches = int(steps / (len(imgs_train) * ds_repeat / batch_size))
    epoches = epoches // 2 if args.command == 'tune' else epoches

post_processing = 'layering' if args.mode == 'sparse' else 'meanShift'

config = instSeg.Config(image_channel=image_channel)
config.post_processing = post_processing
config.H = H 
config.W = W
config.nstage = int(args.unet_stage)
config.filters = int(args.filters)
if embeddingOnly:
    config.modules = ['embedding']
else:
    config.modules = ['embedding', 'foreground']
config.loss['embedding'] = args.embedding_loss
config.loss['foreground'] = 'CE' 
config.neighbor_distance = 15 if args.globalCons is None else None
config.embedding_dim = int(args.D_embedding)
config.embedding_activation = args.embedding_activation
config.embedding_include_bg = False
config.dynamic_weighting = dynamic_weighting
config.embedding_regularization = float(args.embedding_regularization)
if margin:
    if args.embedding_loss == 'cosine':
        config.margin_attr = 1 - np.cos(10/180*np.pi)
        config.margin_rep = np.cos(60/180*np.pi) ** 2
    if args.embedding_loss == 'euclidean':
        config.margin_attr = 0.5
        config.margin_rep = 3
if coord:
    config.guide_function_type = 'cosine'
    config.guide_function_period = guide_function_period

config.allow_overlap = allow_overlap

config.ds_repeat = ds_repeat
if args.augmentation=='all' or args.augmentation=='geo':
    config.flip = True
    config.random_rotation = True
    config.random_rotation_p = 0.5
    config.random_shift = True
    config.random_shift_p = 0.5
    config.max_shift = 64 # maximal shift
    config.elastic_deform = True
    config.elastic_deform_p = 0.5
    config.deform_scale = [16,256]
    config.deform_strength = 0.25
if args.augmentation=='all' or args.augmentation=='non_geo':
    config.random_gamma = True
    config.random_gamma_p = 0.2
    config.random_blur = True
    config.random_blur_p = 0.2
    if image_channel == 3:
        config.random_saturation = True
        config.random_saturation_p = 0.2
    else:
        config.random_saturation = False
        # config.random_hue = False
if embeddingOnly:
    suffix = 'embeddingOnly' if len(suffix) == 0 else 'embeddingOnly' + '_' + suffix
suffix_aug = None
if args.augmentation=='all':
    suffix_aug = 'aug'
elif args.augmentation=='geo':
    suffix_aug = 'augGeo'
if suffix_aug is not None:
    suffix = suffix_aug if len(suffix) == 0 else suffix_aug + '_' + suffix

config.optimizer = args.optimizer
config.learning_rate = float(args.lr)
config.lr_decay_rate = float(args.lr_decay)
config.lr_decay_period = 10000
config.backbone = args.backbone
config.input_normalization = 'per-image'
config.up_scaling = args.up_scaling
config.net_normalization = 'batch'
config.dropout_rate = 0

config.obj_min_size = obj_min_size
config.save_best_metric = 'loss'
config.validation_start_epoch = 1


if args.command.startswith('train'):
    # model = None
    
    model_name = config.exp_name(preffix=args.ds, suffix=suffix)
    model_dir = os.path.join(args.model_dir, model_name)
    model = None
    if os.path.exists(model_dir):
        model = instSeg.load_model(model_dir=model_dir, load_best=False)
    if model is None:
        model = instSeg.Model(config=config, model_dir=model_dir)
    model.config.info()
    augmentation = False if args.augmentation.lower() == 'none' else True
    model.train(ds_train, ds_val, batch_size=batch_size, epochs=epoches, augmentation=augmentation, image_summary=False)

if args.command.startswith('tune'):
    model = instSeg.load_model(model_dir=args.model_dir, load_best=True)
    model.config.validation_start_epoch = 1
    # if args.command == 'tune_sparse': 
    #     suffix = "_sparseTuned" 
    #     model.config.loss['embedding'] = 'sparse_cos' 
    # if args.command == 'tune_overlap': 
    suffix = "_overlapTuned" 
    model.config.loss['embedding'] = 'overlap' 
    model.config.save_best_metric = 'AP'
    if not args.model_dir.endswith(suffix):
        model.save_as(os.path.join(os.path.dirname(args.model_dir), os.path.basename(args.model_dir)+suffix))
        model.config.learning_rate = model.lr()
        model.reset_training()
    model.train(ds_train, ds_val, batch_size=batch_size, epochs=epoches, augmentation=augmentation, image_summary=False)

if args.command == 'detect':
    import shutil
    import cv2

    model_name = config.exp_name(preffix=args.ds, suffix=suffix)
    model_dir = os.path.join(args.model_dir, model_name)
    # print(model_dir)
    # model = instSeg.load_model(model_dir=model_dir, load_best=True)
    model = instSeg.Model(config=config, model_dir=model_dir)
    model.load_weights(load_best=True)
    result_dir = os.path.join(model_dir, 'results')
    print(result_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    for k, img in imgs_test.items():
        print(k)
        if args.ds == 'Neuroblastoma':
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            imgs_test[k] = img
        
        raw = model.predict_raw(img)
        if embeddingOnly or useGtForeground:
            fg = np.sum(masks_test[k], axis=0) if isinstance(masks_test[k], list) else masks_test[k]
            fg = cv2.resize((fg>0).astype(np.uint8), (model.config.W, model.config.H), interpolation=cv2.INTER_NEAREST)
            raw['foreground'] = fg.astype(np.float32)
        embedding = raw['embedding']
        embedding = np.squeeze(embedding)
        # save vis of layering
        save_dir = os.path.join(result_dir, str(k))
        os.makedirs(save_dir)
        # foreground
        fg = np.squeeze(raw['foreground']) > 0.5 
        cv2.imwrite(os.path.join(save_dir, 'foreground.png'), np.uint8(fg*255))
        if args.mode == 'sparse':
            pass
            # instances, layered = model.postprocess(raw, post_processing=post_processing)
            # # layering results
            # for i in range(embedding.shape[-1]):
            #     cv2.imwrite(os.path.join(save_dir, 'layering_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*255))
            #     cv2.imwrite(os.path.join(save_dir, 'layering_masked_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*fg*255))
            #     img_vis = img/img.max() * 255
            #     if len(img.shape) == 2:
            #         img_vis = np.stack((img_vis, img_vis, img_vis), axis=-1)
            #     mask = embedding[:,:,i]*fg*255
            #     mask = np.stack((mask*0, mask*0, mask), axis=-1)
            #     img_vis = img_vis * 0.7 + mask*0.3
            #     cv2.imwrite(os.path.join(save_dir, 'layering_vis_'+str(i)+'.png'), np.uint8(img_vis))
            # layering_gt = np.copy(embedding)*0
            # if args.ds in ['Celegans', 'Cervical2014', 'Neuroblastoma']:
            #     mask_gt = masks_test[k]
            # else:
            #     mask_gt = map2stack(masks_test[k])
            # overlap_free = np.sum([m>0 for m in mask_gt], axis=0) == 1
            # for m in mask_gt:
            #     rr,cc = np.nonzero(m*overlap_free)
            #     idx = np.argmax(np.sum(embedding[rr,cc,:], axis=0))
            #     rr,cc = np.nonzero(m)
            #     layering_gt[rr,cc,idx]=1
            # for i in range(layering_gt.shape[-1]):
            #     img_vis = img/img.max() * 255
            #     if len(img.shape) == 2:
            #         img_vis = np.stack((img_vis, img_vis, img_vis), axis=-1)
            #     mask = layering_gt[:,:,i]*fg*255
            #     mask = np.stack((mask*0, mask*0, mask), axis=-1)
            #     img_vis = img_vis * 0.75 + mask*0.25
            #     cv2.imwrite(os.path.join(save_dir, 'layering_gt_vis_'+str(i)+'.png'), np.uint8(img_vis))
            # # raw layering visualization
        else:
            instances = model.postprocess(raw, post_processing=post_processing)
            clusters = raw['clusters']
            for i in np.unique(clusters):
                if i == 0:
                    continue
                # cv2.imwrite(os.path.join(save_dir, 'layering_'+str(i)+'.png'), np.uint8((clusters == i)*255))
                # cv2.imwrite(os.path.join(save_dir, 'layering_masked_'+str(i)+'.png'), np.uint8((clusters == i)*fg*255))
                img_vis = img/img.max() * 255
                if len(img.shape) == 2:
                    img_vis = np.stack((img_vis, img_vis, img_vis), axis=-1)
                if args.ds == 'CVPPP':
                    C = cv2.resize(((clusters == i)*fg).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    rr, cc = np.nonzero(C)
                    img_vis[rr, cc, 2] = img_vis[rr, cc, 2] * 0.4 + 255 * 0.6
                else:
                    mask = (clusters == i)*fg*255
                    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = np.stack((mask*0, mask*0, mask), axis=-1)
                    img_vis = img_vis * 0.7 + mask*0.3
                cv2.imwrite(os.path.join(save_dir, 'layering_vis_'+str(i)+'.png'), np.uint8(img_vis))
        
        # prediction visualization
        instances = cv2.resize(np.array(instances), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis = instSeg.vis.vis_instance_contour(img, instances)
        cv2.imwrite(os.path.join(save_dir, 'pred.png'), vis)
        # ground truth visualization
        vis = instSeg.vis.vis_instance_contour(img, np.array(masks_test[k]))
        cv2.imwrite(os.path.join(save_dir, 'gt.png'), vis)

if args.command == 'evaluate':
    import cv2

    thres = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    lines = []

    e = instSeg.Evaluator(dimension=2, mode='area', verbose=True)
    model = instSeg.load_model(model_dir=args.model_dir, load_best=True)

    # imgs_test = {'D02': imgs_test['D02']}
    # masks_test = {'D02': masks_test['D02']}

    for k, img in imgs_test.items():
        # print(k)
        if args.ds == 'Neuroblastoma':
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        raw = model.predict_raw(img)
        if post_processing == 'layering':
            instances, layered = model.postprocess(raw, post_processing=post_processing)
        elif post_processing == 'meanShift':
            instances = model.postprocess(raw, post_processing='meanShift')
        e.add_example(instances, masks_test[k])
    lines.append("Evaluation of {} on data set {}".format(args.model_dir.lower(), args.ds.lower()))
    print(lines[0])
    ap = e.AP_DSB(thres=thres)
    lines.append("AP (Data Scient Bowl 2018) over the whole dataset: {}".format(ap))
    aji = e.AJI()
    lines.append("aggregated Jaccard: {}".format(aji))
    for t in thres:
        r = e.detectionRecall(t, metric='Jaccard')
        p = e.detectionPrecision(t, metric='Jaccard')
        lines.append("detectionRecall over the whole dataset under 'Jaccard' {}: {}".format(t, r))
        lines.append("detectionPrecision over the whole dataset under 'Jaccard' {}: {}".format(t, p))
    
    with open(os.path.join(args.model_dir, 'evaluation.txt'), 'w') as f:
            f.write('\n'.join(lines))
