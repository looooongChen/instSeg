from distutils.command.config import config
from pyexpat import model
import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.utils import *
import instSeg.loss as L
from instSeg.enumDef import *
from instSeg.post_process import *
from instSeg.evaluation import Evaluator
from instSeg.data_augmentation import *
import numpy as np
from abc import abstractmethod
from collections.abc import Iterable 
from keras.layers import Conv2D
import os
import cv2

def input_process(x, config):
    if config.input_normalization == 'per-image':
        x = tf.image.per_image_standardization(x)
    elif config.input_normalization == 'constant':
        x = (x - config.input_normalization_bias)/config.input_normalization_scale

    # if self.config.positional_embedding is True:
    #     self.normalized_img = PosConv(filters=self.config.filters)(self.normalized_img)
    return x

def module_output(x, module, config):
    if module == 'semantic':
        outlayer = Conv2D(filters=config.classes, kernel_size=1, activation='softmax')
    if module == 'contour' or module == 'foreground':
        outlayer = Conv2D(filters=1, kernel_size=1, activation='sigmoid')
    if module == 'edt':
        outlayer = Conv2D(filters=1, kernel_size=1, activation='linear')
    if module == 'flow':
        outlayer = Conv2D(filters=2, kernel_size=1, activation='linear')
    if module == 'embedding':
        outlayer = Conv2D(filters=config.embedding_dim, kernel_size=1, activation='linear')
    if module == 'layered_embedding':
        outlayer = Conv2D(filters=config.embedding_dim, kernel_size=1, activation='sigmoid')
    
    return outlayer(x)

class ModelBase(object):

    def __init__(self, config, model_dir='./'):
        self.config = config
        config.model_type = MODEL_BASE
        # model handle
        self.backone = None
        self.features = None
        # model saving
        self.create_model_dir(model_dir)
        # training
        self.training_prepared = False
        self.training_epoch = 0
        self.training_step = 0
        # validation
        self.modules = config.modules
        self.best_score = None
        self.build_model()

    def save_as(self, model_dir):
        self.create_model_dir(model_dir)
    
    def reset_training(self):
        self.best_score = None
        self.training_epoch = 0
        self.training_step = 0

        self.training_prepared = False

    def create_model_dir(self, model_dir):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.weights_latest = os.path.join(self.model_dir, 'weights_latest')
        self.weights_best = os.path.join(self.model_dir, 'weights_best')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.model_dir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(self.model_dir, 'val'))

    @abstractmethod
    def build_model(self):
        ''' build the model self.self.model'''
        self.model = None

    def lr(self):
        p = self.training_step // self.config.lr_decay_period
        return self.config.train_learning_rate * (self.config.lr_decay_rate ** p)
    
    def prepare_training(self):

        ''' prepare loss functions and optimizer'''

        if self.config.lr_decay_period != 0:
            # self.optimizer = keras.optimizers.Adam(learning_rate=lambda : self.lr())
            self.optimizer = keras.optimizers.RMSprop(learning_rate=lambda : self.lr())
        else:
            # self.optimizer = keras.optimizers.Adam(lr=self.config.train_learning_rate)
            self.optimizer = keras.optimizers.RMSprop(lr=self.config.train_learning_rate)

        loss_fns = {}
        # foreground loss
        loss_fns['foreground'] = {'crossentropy': L.bce, 
                                  'crossentropy_random': lambda y_true, y_pred: L.bce_random(y_true, y_pred, N=self.config.ce_neg_ratio),
                                  'crossentropy_hard': lambda y_true, y_pred: L.bce_hard(y_true, y_pred, N=self.config.ce_neg_ratio),
                                  'crossentropy_weighted': L.bce_weighted,
                                  'dice': L.dice,
                                  'focal_loss': lambda y_true, y_pred: L.bfl(y_true, y_pred, gamma=self.config.focal_loss_gamma)}
        # edt regression loss
        loss_fns['edt'] = {'mse': L.mse}
        # contour loss
        loss_fns['contour'] = loss_fns['foreground']
        # embedding loss
        loss_fns['embedding'] = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=self.config.embedding_include_bg),
                                 'sparse_cos': lambda y_true, y_pred, adj_indicator: L.sparse_cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=self.config.embedding_include_bg)}
        loss_fns['layered_embedding'] = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=self.config.embedding_include_bg),
                                         'sparse_cos': lambda y_true, y_pred, adj_indicator: L.sparse_cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=self.config.embedding_include_bg),
                                         'overlap': lambda y_true, y_pred, adj_indicator:  L.overlap_embedding_loss(y_true, y_pred, adj_indicator, include_background=self.config.embedding_include_bg)}

        self.loss_fns = {}
        for m in self.modules:
            self.loss_fns[m] = loss_fns[m][self.config.loss[m]]

        self.training_prepared = True
    
    def data_loader(self, data, modules):

        '''prepare training dataset'''

        data_to_keep = ['image']
        keep_instance = False

        if self.validation_mode() in ['AP', 'AJI']:
            keep_instance = True

        # resize images
        data['image'] = trim_images(data['image'], (self.config.H, self.config.W), interpolation='bilinear').astype(np.int32)
        if 'instance' in data.keys():
            data['instance'] = trim_instance_label(data['instance'], (self.config.H, self.config.W)).astype(np.int32)
        if 'foreground' in data.keys():
            data['foreground'] = (trim_images(data['foreground'], (self.config.H, self.config.W), interpolation='nearest') > 0).astype(np.uint8)
        if 'contour' in data.keys():
            data['contour'] = (trim_images(data['contour'], (self.config.H, self.config.W), interpolation='nearest') > 0).astype(np.uint8)
        if 'edt' in data.keys():
            data['edt'] = trim_images(data['edt'], (self.config.H, self.config.W), interpolation='bilinear').astype(np.float32)

        # computer gt if not existing
        for m in modules:
            if m == 'foreground':
                data_to_keep.append('foreground')
                if 'foreground' not in data.keys():
                    data['foreground'] = (np.sum(data['instance'], axis=-1, keepdims=True)>0).astype(np.uint8) 

            if m == 'contour':
                data_to_keep.append('contour')
                if 'contour' not in data.keys():
                    data['contour'] = (contour(data['instance'], mode=self.config.contour_mode, radius=self.config.contour_radius)> 0).astype(np.uint8)

            # keep edt in ram may cause error, if augmentation (for transformations that does not preserve distances, like elastic tranform) used
            if m == 'edt': 
                data_to_keep.append('edt')
                if 'edt' not in data.keys():
                    data['edt'] = edt(data['instance'], normalize=self.config.edt_normalize, process_disp=True).astype(np.float32)
                    data['edt'] = data['edt'] * 10 if self.config.edt_normalize else data['edt']

            if m in ['embedding', 'layered_embedding']:
                keep_instance = True
                data_to_keep.append('adj_matrix')
                if 'adj_matrix' not in data.keys():
                    data['adj_matrix'] = adj_matrix(data['instance'], self.config.neighbor_distance)

        # delete unused data
        if keep_instance:
            data_to_keep.append('instance')
        for k in list(data.keys()):
            if k not in data_to_keep:
                del data[k]

        # return data
        return tf.data.Dataset.from_tensor_slices(data)

    def get_ds_item(self, ds_frame, m):
        if m in ds_frame.keys():
            return ds_frame[m]
        else:
            if m == 'foreground':
                return (np.sum(ds_frame['instance'], axis=-1, keepdims=True)>0).astype(np.bool) 
            if m == 'contour':
                return contour(ds_frame['instance'], mode=self.config.contour_mode, radius=self.config.contour_radius, process_disp=False)
            if m == 'edt':
                E = edt(ds_frame['instance'], normalize=self.config.edt_normalize, process_disp=False)
                return 10 * E if self.config.edt_normalize else E

    def ds_frame_for_loss(self, ds_frame):
        '''
        check whether required data (for training) is in ds_frame or not, if not try to compute from existing data
        ''' 
        for m in self.modules:
            ds_frame[m] = self.get_ds_item(ds_frame, m)
        return ds_frame
    
    def ds_augment(self, ds):

        img_modification = ['image', 'instance', 'foreground', 'contour', 'edt']
        img_bilinear = ['image', 'edt']
        img_nearest = ['instance', 'foreground', 'contour']
        
        def map_func(d):
            # flipping
            if self.config.flip:
                flip_p = tf.random.uniform((), minval=0, maxval=1)
                for k, im in d.items():
                    if k not in img_modification:
                        continue
                    if flip_p < 0.25:
                        d[k] = tf.image.flip_left_right(im)
                    elif flip_p < 0.5:
                        d[k] = tf.image.flip_up_down(im)
                    elif flip_p < 0.75:
                        d[k] = tf.image.flip_left_right(tf.image.flip_up_down(im))
                    else:
                        pass
            # random rotation
            if self.config.random_rotation:
                angle = tf.cast(tf.random.uniform(shape=[], maxval=359, dtype=tf.int32), tf.float32)
                angle = angle / 360 * (2*3.1415926)
                random_rotation_p = tf.random.uniform((), minval=0, maxval=1)
                for k, im in d.items():
                    if random_rotation_p < self.config.random_rotation_p:
                        if k in img_bilinear:
                            d[k] = tfa.image.rotate(d[k], angle, interpolation='bilinear')
                        if k in img_nearest:
                            d[k] = tfa.image.rotate(d[k], angle, interpolation='nearest')
            # random shift
            if self.config.random_shift:
                shift = tf.cast(tf.random.uniform(shape=[2], maxval=self.config.max_shift, dtype=tf.int32), tf.float32)
                random_shift_p = tf.random.uniform((), minval=0, maxval=1)
                for k, im in d.items():
                    if random_shift_p < self.config.random_shift_p:
                        if k in img_bilinear:
                            d[k] = tfa.image.translate(d[k], shift, interpolation='bilinear')
                        if k in img_nearest:
                            d[k] = tfa.image.translate(d[k], shift, interpolation='nearest')
            # random gamma
            if self.config.random_gamma:
                gamma = tf.random.uniform((), minval=1, maxval=2)
                if tf.random.uniform((), minval=0, maxval=1) < 0.5:
                    gamma = 1/gamma
                random_gamma_p = tf.random.uniform((), minval=0, maxval=1)
                if random_gamma_p < self.config.random_gamma_p:
                    Vmax = tf.cast(tf.reduce_max(d['image']), tf.float32) + 1
                    # gain = Vmax / Vmax ** gamma
                    d['image'] = tf.cast(tf.cast(d['image'], tf.float32) ** gamma / Vmax ** gamma * Vmax, d['image'].dtype)
                    # d['image'] = tf.image.adjust_gamma(d['image'], gamma=gamma, gain=gain)
                    # d['image'] = tf.image.adjust_gamma(d['image'], gamma=gamma, gain=1)
            # random satuation
            if self.config.random_saturation:
                random_saturation_p = tf.random.uniform((), minval=0, maxval=1)
                if random_saturation_p < self.config.random_saturation_p:
                    d['image'] = tf.image.random_saturation(d['image'], 0.5, 1.5)
            # random hue
            if self.config.random_hue:
                random_hue_p = tf.random.uniform((), minval=0, maxval=1)
                if random_hue_p < self.config.random_hue_p:
                    d['image'] = tf.image.random_hue(d['image'], 0.05)
            # random blur
            if self.config.random_blur:
                # sigma = tf.cast(tf.random.uniform((), minval=1, maxval=3), tf.float32)
                random_blur_p = tf.random.uniform((), minval=0, maxval=1)
                if random_blur_p < self.config.random_blur_p:
                    d['image'] = tfa.image.gaussian_filter2d(d['image'], (10, 10), sigma=2)
            # elastic deform
            if self.config.elastic_deform:
                deform_scale = self.config.deform_scale if isinstance(self.config.deform_scale, Iterable) else [self.config.deform_scale, self.config.deform_scale+1] 
                deform_strength = self.config.deform_strength if isinstance(self.config.deform_strength, Iterable) else [self.config.deform_strength, self.config.deform_strength + 0.0001]
                scale = tf.random.uniform(shape=[], minval=deform_scale[0], maxval=deform_scale[1])
                strength = tf.random.uniform(shape=[], minval=deform_strength[0]*scale, maxval=deform_strength[1]*scale)
                # strength = tf.random.uniform(shape=[], minval=deform_strength[0], maxval=deform_strength[1])
                # scale = self.config.deform_scale
                # strength = self.config.deform_strength_max
                flow = elastic_flow(tf.shape(d['image']), scale, strength)
                elastic_deform_p = tf.random.uniform((), minval=0, maxval=1)
                for k, im in d.items():
                    if elastic_deform_p < self.config.elastic_deform_p:
                        if k in img_bilinear:
                            d[k] = warp_image(d[k], flow, interpolation='bilinear')
                        if k in img_nearest:
                            d[k] = warp_image(d[k], flow, interpolation='nearest')
            
            return d

        ds = ds.map(map_func)
        return ds

    def load_weights(self, load_best=False, weights_only=False):

        if load_best:
            if os.path.exists(self.weights_best):
                weights_path = self.weights_best 
            else:
                print(" ==== Weights Best not found, Weights Latest loaded ==== ")
                weights_path = self.weights_latest
        else:
            weights_path = self.weights_latest
        cp_file = tf.train.latest_checkpoint(weights_path)
        
        if cp_file is not None:
            self.model.load_weights(cp_file)
            parsed = os.path.basename(cp_file).split('_')
            disp = 'Model restored from'
            for i in range(1, len(parsed)):
                if parsed[i][:3] == 'epo':
                    disp = disp + ' Epoch {:d}, '.format(int(parsed[i][5:]))
                    self.training_epoch = int(parsed[i][5:]) if not weights_only else self.training_epoch 
                if parsed[i][:3] == 'ste':
                    disp = disp + 'Step {:d}'.format(int(parsed[i][4:]))
                    self.training_step = int(parsed[i][4:]) if not weights_only else self.training_step
                if not weights_only: 
                    self.best_score = self.best_score
                    cp_best = tf.train.latest_checkpoint(self.weights_best) 
                    if cp_best is not None:
                        parsed_best = os.path.basename(cp_best).split('_')
                        for i in range(1, len(parsed_best)):
                            if parsed_best[i][:3] == 'val':
                                self.best_score = float(parsed_best[i][3:])
                # print('current best score: ', self.best_score)
            print(disp)
        else:
            print("==== Model not found! ====")
    
    def save_weights(self, model_type='latest'):
        
        # save_name = 'weights_stage' + str(self.training_stage+1) if stage_wise else 'weights'
        save_name = 'weights' + '_epoch'+str(self.training_epoch)+'_step'+str(self.training_step)
        if model_type == 'latest':
            save_dir = self.weights_latest
        elif model_type == 'best':
            save_name = save_name + '_val'+'{:.5f}'.format(float(self.best_score))
            save_dir = self.weights_best
        elif model_type == 'snapshot':
            save_dir = os.path.join(self.model_dir, 'weights_epoch'+str(self.training_epoch))

        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
        self.model.save_weights(os.path.join(save_dir, save_name))
        print('Model saved at Step {:d}, Epoch {:d}'.format(self.training_step, self.training_epoch))
        self.config.save(os.path.join(self.model_dir, 'config.pkl'))


    def module_loss(self, module, out, ds_frame):
        if module == 'edt':
            loss = self.loss_fns[module](ds_frame['edt'], out)
        # elif module == 'flow':
        #     flow_gt = ds_frame['flow'] if self.config.flow_mode == 'offset' else 10 * ds_frame['flow']
        #     if self.config.flow_loss.startswith('masked'):
        #         mask = np.expand_dims((flow_gt[...,0]**2 + flow_gt[...,1]**2) > 1e-5, axis=-1)
        #         return self.loss_fns['flow'](flow_gt, out, mask) * self.config.flow_weight
        #     else:
        #         return self.loss_fns['flow'](flow_gt, out) * self.config.flow_weight
        elif module in ['embedding', 'layered_embedding']:
            loss = self.loss_fns[module](ds_frame['instance'], out, ds_frame['adj_matrix'])
        else:
            loss = self.loss_fns[module](ds_frame[module], out)

        loss_weight = self.config.loss_weights[module] if module in self.config.loss_weights else 1
        return loss * loss_weight


    def model_loss(self, ds_frame, training=False):
        outs = self.model(ds_frame['image'], training=training)
        if len(self.modules) == 1:
            outs = [outs]

        module_losses, loss = {}, 0
        for m, out in zip(self.modules, outs):
            module_losses[m] = self.module_loss(m, out, ds_frame)
            loss += module_losses[m]
        loss += sum(self.model.losses)

        return loss, module_losses, outs


    def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
              augmentation=True, image_summary=True, clear_best_val=False):
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'instance': ..., 'foreground': ...} 
                image (required): numpy array of size N x H x W x C 
                instance: numpy array of size N x H x W x 1, 0 indicated background, or list of list of object masks of size H x W:  [[img1_obj1, img1_obj2, ...], [img2_obj1, img2_obj2, ...], ...]
                foreground: numpy array of size N x H x W x 1
        '''
        # prepare network
        if not self.training_prepared:
            self.prepare_training()
            # if self.config.transfer_training == True and self.config.backbone.lower().startswith('resnet'):
            #     self.backbone.trainable = False
        epochs = self.config.train_epochs if epochs is None else epochs
        batch_size = self.config.train_batch_size if batch_size is None else batch_size

        # prepare data
        train_ds = self.data_loader(train_data, self.modules)
        if augmentation:
            train_ds = self.ds_augment(train_ds)
        if self.config.ds_repeat > 1:
            train_ds = train_ds.repeat(self.config.ds_repeat)
        train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size)
        if validation_data is None or len(validation_data['image']) == 0:
            val_ds = None
        else:
            val_ds = self.data_loader(validation_data, self.modules).batch(1)
            
        # load model
        self.load_weights()
        if clear_best_val:
            self.best_score = None

        # train
        for _ in range(epochs-self.training_epoch):
            for ds_frame in train_ds:
                ds_frame = self.ds_frame_for_loss(ds_frame)
                with tf.GradientTape() as tape:
                    loss, module_losses, outs = self.model_loss(ds_frame, training=True)

                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # display trainig loss
                    self.training_step += 1
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}".format(self.training_epoch+1, self.training_step, float(loss))
                    for m, l in module_losses.items():
                        disp += ', ' + m + ' loss: {:.5f}'.format(float(l))
                    print(disp)
                    # summary training loss
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.training_step)
                        for m, l in module_losses.items():
                            tf.summary.scalar('loss_'+m, l, step=self.training_step)
                    # summary output
                    if self.training_step % self.config.summary_step == 0 and image_summary:
                        with self.train_summary_writer.as_default():
                            vis_image = ds_frame['image']/tf.math.reduce_max(ds_frame['image'])*255
                            tf.summary.image('input_img', tf.cast(vis_image, tf.uint8), step=self.training_step, max_outputs=1)
                            outs_dict = {k: v for k, v in zip(self.modules, outs)}
                            # foreground
                            if 'foreground' in outs_dict.keys():
                                vis_foreground = outs_dict['foreground']
                                tf.summary.image('foreground', vis_foreground*255/tf.reduce_max(vis_foreground), step=self.training_step, max_outputs=1)
                                gt = tf.cast(ds_frame['foreground'], tf.int32)
                                tf.summary.image('foreground_gt', tf.cast(gt*255/tf.reduce_max(gt), tf.uint8), step=self.training_step, max_outputs=1)
                            # contour
                            if 'contour' in outs_dict.keys():
                                vis_contour = tf.cast(outs_dict['contour']*255, tf.uint8)
                                tf.summary.image('contour', vis_contour, step=self.training_step, max_outputs=1)
                                vis_contour_gt = tf.cast(ds_frame['contour'], tf.uint8) * 255
                                tf.summary.image('contour_gt', vis_contour_gt, step=self.training_step, max_outputs=1)
                            # edt regression
                            if 'edt' in outs_dict.keys():
                                vis_edt = tf.cast(outs_dict['edt']*255/tf.reduce_max(outs_dict['edt']), tf.uint8)
                                tf.summary.image('edt', vis_edt, step=self.training_step, max_outputs=1)
                                vis_edt_gt = tf.cast(ds_frame['edt']*255/tf.reduce_max(ds_frame['edt']), tf.uint8)
                                tf.summary.image('edt_gt', vis_edt_gt, step=self.training_step, max_outputs=1)
                            # embedding
                            if 'embedding' in outs_dict.keys():
                                for i in range(self.config.embedding_dim//3):
                                    vis_embedding = outs_dict['embedding'][:,:,:,3*i:3*(i+1)]
                                    tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                    if not self.config.embedding_include_bg:
                                        vis_embedding = vis_embedding * tf.cast(tf.reduce_sum(tf.cast(ds_frame['instance'], tf.int32), axis=-1, keepdims=True) > 0, vis_embedding.dtype)
                                        tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                            if 'layered_embedding' in outs_dict.keys():
                                for i in range(self.config.embedding_dim//3):
                                    vis_embedding = outs_dict['layered_embedding'][:,:,:,3*i:3*(i+1)]
                                    tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                    if not self.config.embedding_include_bg:
                                        vis_embedding = vis_embedding * tf.cast(tf.reduce_sum(tf.cast(ds_frame['instance'], tf.int32), axis=-1, keepdims=True) > 0, vis_embedding.dtype)
                                        tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                            # if 'layered_instances' in outs_dict.keys():
                            #     for i in range(self.config.instance_layers):
                            #         vis_instance = tf.expand_dims(outs_dict['layered_instances'][:,:,:,i]>0.5, axis=-1)
                            #         vis_instance = tf.cast(vis_instance, tf.uint8) * 255
                            #         tf.summary.image('layerd_instance_{}'.format(i), vis_instance, step=self.training_step, max_outputs=1)
            self.training_epoch += 1

            if self.training_epoch in self.config.snapshots:
                self.save_weights(model_type='snapshot')


            self.save_weights(model_type='latest')
            if self.training_epoch >= self.config.validation_start_epoch and val_ds is not None:
                self.validate(val_ds, save_best=True)

    def predict_raw(self, image, training=False):
        
        img = np.squeeze(image)
        img = image_resize_np([img], (self.config.H, self.config.W))
        img = K.cast_to_floatx(img)
        raw = self.model(img, training=training)
        raw = {m: np.squeeze(o) for m, o in zip(self.modules, raw)}

        return raw

    def postprocess(self, raw, post_processing=None):
        ''' return a tuple, the first item will be used by validate() for validation
        parametes could be set in config:
            DCAN:
                - dcan_thres_contour
            Embedding + edt:
                - emb_thres
                - emb_max_step
                - edt_thres_upper
                - edt_intensity
            EDT:
                - di
            All:
                - min_size
                - max_size
        '''

        instances = None
        post_processing = post_processing if post_processing is not None else self.config.post_processing

        if post_processing == 'layered_embedding':
            if 'layered_embedding' in raw.keys() and 'foreground' in raw.keys():
                instances, layered = instance_from_layered_embedding(raw, self.config) # size screening has been done insde 'instance_from_layered_embedding'
                return instances, layered
        else:
            pass
        
        # if self.config.post_processing is not set, auto selection
        if instances is None: 
            if len(self.modules) == 1:
                if 'foreground' in raw.keys():
                    instances = instance_from_foreground(raw, self.config)
                if 'embedding' in raw.keys():
                    instances = instance_from_embedding(raw, self.config)
                if 'edt' in raw.keys():
                    instances = instance_from_edt(raw, self.config)
            
            if len(self.modules) == 2:
                if 'foreground' in raw.keys() and 'contour' in raw.keys():
                    instances = instance_from_foreground_and_contour(raw, self.config)
                elif 'embedding' in raw.keys() and 'edt' in raw.keys():
                    instances = instance_from_emb_and_edt(raw, self.config)
                elif 'foreground' in raw.keys() and 'edt' in raw.keys():
                    instances = instance_from_edt(raw, self.config)
        
        instances = size_screening(instances, self.config.obj_min_size, self.config.obj_max_size)
        if instances is not None and len(instances.shape) == 3:
            instances = instances.astype(np.uint8)
    
        return instances

    def validation_mode(self):
        validation_mode = self.config.save_best_metric
        if validation_mode == 'loss':
            return 'loss'
        if len(self.modules) == 1 and 'foreground' in self.modules:
            validation_mode = 'Dice'
        return validation_mode      

    def validate(self, val_ds, save_best=True):
        '''
        Args:
            val_ds: validation dataset
        '''
        print('Running validation in mode: ', self.validation_mode())
        if self.validation_mode() == 'loss':
            losses = []
            for ds_frame in val_ds:
                loss, module_losses, _ = self.model_loss(ds_frame, training=False)
                disp = "Validation example with loss: {:.5f}".format(float(loss))
                for m, l in module_losses.items():
                    disp += ', ' + m + ' loss: {:.5f}'.format(float(l))
                print(disp)
                losses.append(loss)
        elif self.validation_mode() == 'Dice':
            Agg_I, Agg_S = 0, 0
            for ds_frame in val_ds:
                outs = self.model(ds_frame['image'])
                for m, out in zip(self.modules, outs):
                    if m == 'foreground':
                        pd = np.squeeze(out) > 0.5
                        gt = self.get_ds_item(ds_frame, 'foreground')
                        I, S = np.sum((pd * np.squeeze(gt)) >0), np.sum(pd) + np.sum(gt>0)
                        Agg_I, Agg_S = Agg_I + I, Agg_S + S 
                        D = 2 * I / (S+1e-6)
                        disp = "Validation example with dice: {:.5f}".format(float(D))
                        print(disp)
        else:
            e = Evaluator(dimension=2, mode='area')
            for ds_frame in val_ds:
                outs = self.model(ds_frame['image'])
                instances = self.postprocess({m: o for m, o in zip(self.modules, outs)})
                if isinstance(instances, tuple):
                    instances = instances[0]
                instances = np.array(instances)
                gt = np.squeeze(ds_frame['instance'], axis=0)
                if ds_frame['instance'].shape[-1] == 1:
                    gt = np.squeeze(gt, axis=-1)
                else:
                    gt = np.moveaxis(gt, -1, 0)

                e.add_example(instances, gt)


        with self.val_summary_writer.as_default():
            if self.validation_mode() == 'loss':
                score = np.mean(losses)
                tf.summary.scalar('validation loss', score, step=self.training_step)
                disp = 'validation loss: {:.5f}'.format(score)
            elif self.validation_mode() == 'Dice':
                score = 2 * Agg_I / (Agg_S + 1e-6)
                tf.summary.scalar('validation dice', score, step=self.training_step)
                disp = 'validation loss: {:.5f}'.format(score)
            else:
                AP, AJI = e.AP_DSB(), e.AJI()
                score = AP if self.config.save_best_metric == 'AP' else AJI
                tf.summary.scalar('validation AP', AP, step=self.training_step)
                tf.summary.scalar('validation AJI', AJI, step=self.training_step)
                disp = 'validation AP: {:.5f}, AJI: {:.5f}'.format(AP, AJI)

        if self.best_score is None:
            self.best_score = score

        if self.validation_mode() == 'loss':
            score_cp, best_score_cp = - score, - self.best_score
        else:
            score_cp, best_score_cp = score, self.best_score

        if score_cp > best_score_cp:
            self.best_score = score
            disp = "Validation Score Improved! " + disp
            if save_best:
                self.save_weights(model_type='best')
        else:
            disp = "Validation Score Not Improved! " + disp

        print(disp)
        
    def predict(self, image):
        
        raw = self.predict_raw(image)
        # post processing
        instances = self.postprocess(raw)

        return instances