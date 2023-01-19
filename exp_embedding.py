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

parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train_sparse_embedding', 'tune_overlap', 'train_embedding', 'tune_sparse'")
parser.add_argument('--model_dir', default='./models_elastic/model_scale128_strength128', help='model_dir')
parser.add_argument('--backbone', default='unet', help='')
parser.add_argument('--nstage', default=4, help='nstage')
parser.add_argument('--filters', default=32, help='nstage')
parser.add_argument('--epoches', default=300, help='training epoches')
parser.add_argument('--rotation', default=False, help="")
parser.add_argument('--translation', default=True, help="")
parser.add_argument('--neightborhood', default=64, help="") # 32: 4-N; 64: 8-N; 128: 21-N (24-3)

args = parser.parse_args()

model_dir = args.model_dir
sz = 512
distance = 64

g_config = instSeg.Generator_Config()
g_config.img_sz = sz
g_config.obj_sz = distance # interger or 'random'
g_config.max_degree = 10
g_config.pattern = 'grid' # 'grid' or 'random'
g_config.rotation = args.rotation
g_config.translation = args.translation

# g_config.shape = ['circle', 'triangle', 'square']
g_config.shape = 'circle'
g_config.obj_color = (254, 208, 73)
g_config.bg_color = (0,0,0)


g = instSeg.Pattern_Generator(g_config)

# for i in range(5):
#     img = g.generate()
#     cv2.imwrite('test_{}.png'.format(i), img.astype(np.uint8))

if args.command == 'train':
    ## basic setting
    config = instSeg.Config(image_channel=3)
    config.H = sz
    config.W = sz
    config.backbone = args.backbone
    config.filters = int(args.filters)
    config.nstage = int(args.nstage)
    config.modules = ['layered_embedding']
    config.embedding_dim = 8
    config.neighbor_distance = int(args.neightborhood)
    
    config.loss['layered_embedding'] = 'cos' 


    config.train_learning_rate = 1e-4
    config.lr_decay_rate = 0.9
    config.lr_decay_period = 10000
    # config.input_normalization = 'per-image'
    config.input_normalization = None
    # config.net_upsample = 'deConv' # 'upConv', 'deConv'
    config.up_scaling = 'upConv'
    # config.net_normalization = 'batch'
    config.net_normalization = None
    config.dropout_rate = 0

    config.random_shift = False
    config.flip = False
    config.random_rotation = False
    config.random_gamma = False
    config.random_blur = False
    config.elastic_deform = True
    config.deform_scale = [128,128]
    config.deform_strength_max = 32

    ## training data setting
    epoches = int(args.epoches)

    ## load dataset
    X_train, y_train = [], []
    for idx in range(10):
        img = g.generate()
        # cv2.imwrite('img_{}.png'.format(idx), img)
        X_train.append(img)
        mask = relabel(img[:,:,0] > 0)
        y_train.append(mask)
    ds_train = {'image': X_train, 'instance': y_train}
    
    model = instSeg.Model(config=config, model_dir=args.model_dir)
    model.train(ds_train, None, batch_size=4, epochs=epoches, augmentation=True)


if args.command == 'eval':
    from sklearn.decomposition import PCA
    import umap
    import time
    import os


    model_dirs = [args.model_dir]

    # model_dirs = ['./models_exp/embedding_Grid_unet2', './models_exp/embedding_Grid_unet3', './models_exp/embedding_Grid_unet4', './models_exp/embedding_Grid_unet5',
    # './models_exp/embedding_RotatedGrid_unet2', './models_exp/embedding_RotatedGrid_unet3', './models_exp/embedding_RotatedGrid_unet4', './models_exp/embedding_RotatedGrid_unet5']

    for model_dir in model_dirs:
        print(model_dir)
        model = instSeg.load_model(model_dir=model_dir, load_best=False)
        if not os.path.exists(os.path.join(model_dir, 'visulization')):
            os.makedirs(os.path.join(model_dir, 'visulization'))

        for idx in range(10):
            img = g.generate()
            embedding = model.predict_raw(img)['layered_embedding']
            embedding = np.squeeze(embedding)
            embedding = embedding[1::2,1::2,:]
            img = img[1::2,1::2,:]
            
            H, W = embedding.shape[0], embedding.shape[1]
            rr, cc = np.nonzero(img[:,:,0] > 0)
            embedding = embedding[rr, cc, :]
            embedding = embedding/np.sqrt(np.sum(embedding**2, axis=1, keepdims=True))

            # pca = PCA(n_components=3)
            # pca.fit(embedding)
            # embedding = pca.fit_transform(embedding)
            # embedding = embedding.reshape(H,W,3)

            t = time.time()
            # embedding = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=100).fit_transform(embedding)
            # embedding = Isomap(n_components=3).fit_transform(embedding)
            reducer = umap.UMAP(n_neighbors=50, min_dist=0, n_components=3)
            embedding = reducer.fit_transform(embedding)
            embedding = (embedding - embedding.min())/(embedding.max() - embedding.min()) * 255
            print(time.time() - t)

            vis = np.zeros((H, W, 3))
            vis[rr, cc, :] = embedding
            
            cv2.imwrite(os.path.join(model_dir, 'visulization', 'img_{}.png'.format(idx)), img.astype(np.uint8))
            cv2.imwrite(os.path.join(model_dir, 'visulization', 'vis_{}.png'.format(idx)), vis.astype(np.uint8))







