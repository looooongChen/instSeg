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



parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train_sparse_embedding', 'tune_overlap', 'train_embedding', 'tune_sparse'")
parser.add_argument('--exp', default='semantic', help='experiments')                    
parser.add_argument('--model_dir', default='./models_elastic/model_scale128_strength128', help='model_dir')
parser.add_argument('--backbone', default='unet', help='')
parser.add_argument('--nstage', default=4, help='nstage')
parser.add_argument('--filters', default=32, help='nstage')
parser.add_argument('--coord_type', default=None, help='coord type')
parser.add_argument('--coord_period', default=64)
parser.add_argument('--dynamic_weighting', action=argparse.BooleanOptionalAction)
parser.add_argument('--steps', default=5000, help='training steps')
parser.add_argument('--upscale', default='deConv', help='training steps')
parser.add_argument('--training_examples', default=100, help='training steps')
parser.add_argument('--test_examples', default=10, help='training steps')
parser.add_argument('--augmentation', action=argparse.BooleanOptionalAction)
parser.add_argument('--grid', default=64, help='grid')
parser.add_argument('--radius', default=18, help='radius')

batch_size = 4
mode = 'layered_embedding'
args = parser.parse_args()
model_dir = args.model_dir
radius = int(args.radius)
training_examples = int(args.training_examples)
test_examples = int(args.test_examples)
grid = int(args.grid)
augmentation = False if args.augmentation is None else True

if args.command == 'train':
    random.seed(1)
else:
    random.seed(2)

g_config = instSeg.GeneratorConfig()

# g_config.shape = ['circle', 'triangle', 'square']
g_config.obj_color = (254, 208, 73)
g_config.bg_color = (0,0,0)

g_config.img_sz = (64*8, 64*8)
g_config.grid_sz = (grid, grid)
if args.exp == 'grid' or args.exp == 'grid_offset' or args.exp == 'grid_translate' or args.exp == 'grid_rotation':
    if args.exp == 'grid':
        eval_count = 1 
    else:
        eval_count = test_examples
    g_config.obj_spec = {'circle': {'radius': radius}}
    g_config.obj_shape = 'circle'
    g_config.type = 'grid'
    
    if args.exp == 'grid_translate':
        g_config.grid_translation = True
    else:    
        g_config.grid_translation = False
    
    if args.exp == 'grid_rotation':
        g_config.grid_rotation = True
    else:    
        g_config.grid_rotation = False

    if args.exp == 'grid_offset':
        g_config.obj_translation = 4
elif args.exp == 'grid_ellipse':
    g_config.obj_spec = {'ellipse': {'major': radius, 'minor': round(radius/3)}}
    g_config.obj_shape = 'ellipse'
    eval_count = test_examples
    g_config.type = 'grid'
    g_config.obj_rotation = 30
elif args.exp == 'pair' or args.exp == 'quartet':
    eval_count = test_examples
    g_config.obj_spec = {'circle': {'radius': radius}}
    g_config.obj_shape = 'circle'
    g_config.type = args.exp
    g_config.obj_rotation = False


g = instSeg.GridGenerator(g_config)

if args.command == 'debug':
    for i in range(5):
        img = g.generate()
        cv2.imwrite('test_{}.png'.format(i), img[:,:,::-1].astype(np.uint8))

if args.command == 'train':
    ## basic setting
    config = instSeg.Config(image_channel=3)
    config.H = g_config.img_sz[0]
    config.W = g_config.img_sz[1]
    config.backbone = args.backbone
    config.filters = int(args.filters)
    config.nstage = int(args.nstage)
    config.modules = [mode]
    config.embedding_dim = 8
    if args.exp == 'quartet':
        config.neighbor_distance = g_config.grid_sz[0] * 3
    else:
        config.neighbor_distance = int(g_config.grid_sz[0] * 1.4)
    
    config.loss[mode] = 'cos'
    config.embedding_include_bg = False
    config.dynamic_weighting = False if args.dynamic_weighting is None else True

    config.train_learning_rate = 1e-4
    config.lr_decay_rate = 0.9
    config.lr_decay_period = 10000
    # config.input_normalization = 'per-image'
    config.input_normalization = None
    config.up_scaling = args.upscale
    config.net_normalization = 'batch'
    config.dropout_rate = 0
    config.random_shift = False

    if args.coord_type is not None:
        if args.coord_type == 'bandS':
            config.coord_type = 'cosine'
            config.coord_period = [32, 64, 128]
        elif args.coord_type == 'bandM':
            config.coord_type = 'cosine'
            config.coord_period = [32, 48, 64, 96, 128]
        else:
            config.coord_type = args.coord_type
            config.coord_period = int(args.coord_period)

    ## training data setting

    ## load dataset
    X_train, y_train = [], []
    for idx in range(training_examples):
        img = g.generate()
        X_train.append(img)
        mask = relabel(img[:,:,0] > 0)
        y_train.append(mask)
    ds_train = {'image': X_train, 'instance': y_train}

    epoches = int(int(args.steps)/(len(X_train)/batch_size)) + 1
    
    model = instSeg.Model(config=config, model_dir=args.model_dir)
    model.train(ds_train, None, batch_size=batch_size, epochs=epoches, augmentation=augmentation)


if args.command == 'eval':
    # from sklearn.decomposition import PCA
    import umap
    import time
    import os


    model_dirs = [args.model_dir]

    # model_dirs = ['./models_exp/embedding_Grid_unet2', './models_exp/embedding_Grid_unet3', './models_exp/embedding_Grid_unet4', './models_exp/embedding_Grid_unet5',
    # './models_exp/embedding_RotatedGrid_unet2', './models_exp/embedding_RotatedGrid_unet3', './models_exp/embedding_RotatedGrid_unet4', './models_exp/embedding_RotatedGrid_unet5']

    for model_dir in model_dirs:
        print(model_dir)
        model = instSeg.load_model(model_dir=model_dir, load_best=False)
        # model.config.H = g_config.img_sz[0]
        # model.config.W = g_config.img_sz[1]
        if not os.path.exists(os.path.join(model_dir, 'visulization')):
            os.makedirs(os.path.join(model_dir, 'visulization'))

        Consistency, Distinctiveness, error, total = [], [], [], []
        for idx in range(eval_count):
            img = g.generate()
            # cv2.imwrite(os.path.join(model_dir, 'visulization', 'img_{}.png'.format(idx)), img.astype(np.uint8))
            embedding = model.predict_raw(img)[mode]
            embedding = np.squeeze(embedding)

            consistency, distinctiveness = instSeg.result_analyse.embedding_analysis(relabel(img[:,:,0] > 0), embedding, metric='cos', radius=int(int(g_config.grid_sz[0] * 1.4)))
            N_false = instSeg.result_analyse.false_merge(distinctiveness, thres=45/180*3.1415)
            Consistency = Consistency + list(consistency.values())
            rr, cc = np.nonzero(distinctiveness != float('inf'))
            Distinctiveness = Distinctiveness + list(distinctiveness[rr, cc])
            error.append(N_false)
            total.append(len(distinctiveness))

            print("test{}: {}".format(idx, N_false))
            if idx < 10:
                vis = instSeg.result_analyse.visualize_embedding(embedding, mask=img[:,:,0] > 0, metric='cos')
                cv2.imwrite(os.path.join(model_dir, 'visulization', 'vis_{}.png'.format(idx)), vis.astype(np.uint8))


        lines = ['Consistency: {}'.format(np.mean(Consistency)), 'Distinctiveness: {}'.format(np.mean(Distinctiveness)), 'errors: {}/{}'.format(np.sum(error), np.sum(total))]
        with open(os.path.join(model_dir, 'eval.txt'), 'w') as f:
            f.write('\n'.join(lines))





