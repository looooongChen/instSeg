import instSeg
import numpy as np
import cv2
from tensorflow import keras
import argparse
import tensorflow as tf
import random
import os


parser = argparse.ArgumentParser()

parser.add_argument("command",
                    metavar="<command>",
                    help="'train_sparse_embedding', 'tune_overlap', 'train_embedding', 'tune_sparse'")
parser.add_argument('--net', default='unet', help='experiments')
parser.add_argument('--coord_sz', default=16, help='experiments')
parser.add_argument('--pt_num', default=300, help='experiments')                    
parser.add_argument('--model_dir', default='./models_elastic/model_scale128_strength128', help='model_dir')
parser.add_argument('--steps', default=10000, help='training steps')
parser.add_argument('--upscale', default='deConv', help='training steps')


batch_size = 8
args = parser.parse_args()
model_dir = args.model_dir
coord_sz = int(args.coord_sz)
pt_num = int(args.pt_num)

g_config = instSeg.GeneratorConfig()
g_config.type = 'stain' 

# g_config.shape = ['circle', 'triangle', 'square']
g_config.obj_color = (254, 208, 73)
g_config.bg_color = (0,0,0)

g_config.img_sz = (512, 512)
g_config.coord_sz = (coord_sz, coord_sz)
if args.net == 'encoder':
    g_config.coord_resolution = (16,16)
    encoder_only = True
    grid_sz = 16
else:
    g_config.coord_resolution = (1,1)
    encoder_only = False
    grid_sz = 1
g_config.grid_sz = (grid_sz, grid_sz)

g = instSeg.GridGenerator(g_config)

if args.command == 'debug':
    for i in range(1):
        print('===')
        img, coord_gird, coord_global = g.generate()
        cv2.imwrite('test_{}.png'.format(i), img[:,:,::-1].astype(np.uint8))
        xx, yy = np.nonzero(coord_gird[:,:,0])
        print(coord_global[xx,yy,0])
        print(coord_global[xx,yy,1])
        xx, yy = np.nonzero(img[:,:,0])
        print(xx)
        print(yy)

if args.command == 'train':

    inputs = tf.keras.layers.Input((g_config.img_sz[0], g_config.img_sz[1], 3), name='input_img')

    model, fts, _ = instSeg.nets.unet(inputs,
                                      filters=32,
                                      stages=4,
                                      normalization=None,
                                      up_scaling=args.upscale,
                                      concatenate=True,
                                      encoder_only=encoder_only)

    output = tf.keras.layers.Conv2D(2,1)(fts)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    def masked_mse(y_true, y_pred):
        mask = tf.stop_gradient(tf.cast(y_true != 0, y_pred.dtype))
        loss = tf.reduce_sum((y_true - y_pred) ** 2 * mask) / tf.reduce_sum(mask)
        # loss = tf.reduce_mean((y_true - y_pred) ** 2) 
        return loss

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
              loss=masked_mse, metrics=[masked_mse])

    ## load dataset
    X_train, y_train = [], []
    for idx in range(100):
        img, coord = g.generate(num=pt_num)
        X_train.append(img)
        y_train.append(coord)
    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(np.float32)

    epochs = int(int(args.steps)/(len(X_train)/batch_size)) + 1

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    model.save_weights(args.model_dir, overwrite=True)

if args.command == 'eval':

    thres = 1

    inputs = tf.keras.layers.Input((g_config.img_sz[0], g_config.img_sz[1], 3), name='input_img')

    model, fts, _ = instSeg.nets.unet(inputs,
                                      filters=32,
                                      stages=4,
                                      normalization=None,
                                      up_scaling=args.upscale,
                                      concatenate=True,
                                      encoder_only=encoder_only)

    output = tf.keras.layers.Conv2D(2,1)(fts)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.load_weights(args.model_dir).expect_partial()

    save_dir = os.path.join(os.path.dirname(args.model_dir), 'visualization')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    error = []

    for i in range(10):
        img, coord_gt = g.generate(num=pt_num)
        img_vis = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        xx_img, yy_img = np.nonzero(img[:,:,0])
        xx_out, yy_out = xx_img // grid_sz, yy_img // grid_sz
        # xx_out, yy_out = np.nonzero(coord_gt[:,:,0])

        coord_gt = coord_gt[xx_out, yy_out, :]
        coord_pred = model(np.expand_dims(img, axis=0))
        coord_pred = np.squeeze(coord_pred)
        coord_pred = coord_pred[xx_out, yy_out, :]
        # print(pred)

        D = np.sqrt(np.sum((coord_pred - coord_gt) ** 2, axis=-1)+1e-12)
        # D = np.sqrt((coord_pred[:,0] - coord_gt[:,0]) ** 2 + (coord_pred[:,1] - coord_gt[:,1]) ** 2+1e-12)

        img_vis[xx_img, yy_img, 2] = 255 # ground truth, red
        xx_pred = xx_img - xx_img % coord_sz + np.round(coord_pred[:,0]) - 1
        yy_pred = yy_img - yy_img % coord_sz + np.round(coord_pred[:,1]) - 1
        xx_pred, yy_pred = xx_pred.astype(np.int32), yy_pred.astype(np.int32)

        idx_plot = np.logical_and(xx_pred < img.shape[0], yy_pred < img.shape[1])

        img_vis[xx_pred[idx_plot], yy_pred[idx_plot], 1] = 255 # ground truth, green

        for j in range(len(xx_img)):
            if idx_plot[j] is False:
                continue
            img_vis = cv2.line(img_vis, (yy_img[j], xx_img[j]), (yy_pred[j], xx_pred[j]), (0,255,255), thickness=1) 


        # img_vis[xx[D > thres], yy[D > thres], 2] = 255
        # img_vis[xx[D <= thres], yy[D <= thres], 1] = 255 

        cv2.imwrite(os.path.join(save_dir, 'img_{}.png'.format(i)), img_vis)

        error.append(np.mean(D))
    
    print('=== average error ====', np.mean(error))

        






