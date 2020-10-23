# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.uNet import *
from instSeg.utils import *
import instSeg.loss as L
from instSeg.post_process import *
import os

try:
    import tfAugmentor as tfaug 
    augemntor_available = True
except:
    augemntor_available = False

class InstSegBase(object):

    def __init__(self, config, base_dir='./', run_name=''):
        self.config = config
        self.base_dir = os.path.join(base_dir, run_name)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.weights_latest = os.path.join(self.base_dir, 'weights_latest')
        self.weights_best = os.path.join(self.base_dir, 'weights_best')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.base_dir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(self.base_dir, 'val'))
        self._build()
        self.training_prepared = False
        self.training_stage = 0
        self.training_epoch = 0
        self.training_step = 0
        # self.val_best = float('inf')
        self.val_best = 0
    
    def lr(self):
        p = self.training_step // self.config.lr_decay_period
        return self.config.train_learning_rate * (self.config.lr_decay_rate ** p)
    
    def prepare_training(self,
                         semantic=False,
                         dist=False,
                         embedding=False, 
                         contour=False):
        if self.config.lr_decay_period != 0:
            self.optimizer = keras.optimizers.Adam(learning_rate=lambda : self.lr())
        else:
            self.optimizer = keras.optimizers.Adam(lr=self.config.train_learning_rate)

        # if 'semantic' in self.module_config:
        if semantic:
            loss_semantic = {'crossentropy': L.crossentropy, 
                             'focal_loss': lambda y_true, y_pred: L.focal_loss(y_true, y_pred, gamma=2.0),
                             'dice_loss': L.dice_loss}
            self.loss_fn_semantic = loss_semantic[self.config.loss_semantic] 
        # if 'dist' in self.module_config:
        if dist:
            loss_dist = {'binary_crossentropy': lambda y_true, y_pred: L.weighted_binary_crossentropy(y_true, y_pred, neg_weight=self.config.dist_neg_weight), 
                         'mse': lambda y_true, y_pred: L.mse(y_true, y_pred, neg_weight=self.config.dist_neg_weight)} 
            self.loss_fn_dist = loss_dist[self.config.loss_dist]
        # if 'embedding' in self.module_config:
        if embedding:
            loss_embedding = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, self.config.max_obj, include_background=self.config.embedding_include_bg)}
            self.loss_fn_embedding = loss_embedding[self.config.loss_embedding]
        if contour:
            loss_contour = loss_semantic = {'crossentropy': L.binary_crossentropy, 
                                            'focal_loss': lambda y_true, y_pred: L.binary_focal_loss(y_true, y_pred, gamma=2.0)}
            self.loss_fn_contour = loss_contour[self.config.loss_contour]

        self.training_prepared = True
    
    def ds_from_np(self, data,
                   instance=False, 
                   semantic=False,
                   dist=False,
                   embedding=False, 
                   contour=False):

        for k in data.keys():
            if k == 'image':
                data[k] = image_resize_np(data[k], (self.config.H, self.config.W))
                data[k] = K.cast_to_floatx(data[k])
            else:
                data[k] = image_resize_np(data[k], (self.config.H, self.config.W), method='nearest')

        required = ['image']

        if instance:
            required.append('instance')

        if semantic:
            required.append('semantic')
            if 'semantic' in data.keys():
                data['semantic'] = data['semantic']
            else:
                data['semantic'] = tf.cast(data['instance']>0, tf.uint16)

        if dist:
            required.append('dist')
            data['dist']  = edt_np(data['instance'], normalize=True)

        if embedding:
            required.append('instance')
            required.append('adj_matrix')
            data['adj_matrix'] = adj_matrix_np(data['instance'], self.config.neighbor_distance, self.config.max_obj)

        if contour:
            required.append('contour')
            data['contour'] = contour_np(data['instance'], radius=self.config.contour_radius)

        for k in list(data.keys()):
            if k not in required:
                del data[k]

        # return data
        return tf.data.Dataset.from_tensor_slices(data)
    
    def ds_augment(self, ds):
        if augemntor_available:
            image_list = ['image']
            label_list = []
            if self.config.dist:
                image_list.append('dist')
            if self.config.semantic:
                label_list.append('semantic')
            if self.config.embedding:
                label_list.append('instance')
            aug_ds = []
            if self.config.flip:
                aug_flip_lr = tfaug.Augmentor(image=image_list, label=label_list)
                aug_flip_lr.flip_left_right(probability=1)
                aug_ds.append(aug_flip_lr(ds))
                aug_flip_ud = tfaug.Augmentor(image=image_list, label=label_list)
                aug_flip_ud.flip_up_down(probability=1)
                aug_ds.append(aug_flip_ud(ds))
            if self.config.elastic_strength != 0 and self.config.elastic_scale != 0:
                print('elastic')
                aug_elas = tfaug.Augmentor(image=image_list, label=label_list)
                aug_elas.elastic_deform(strength=self.config.elastic_strength, scale=self.config.elastic_scale, probability=1)
                aug_ds.append(aug_elas(ds))
            if self.config.rotation:
                aug_rotate90 = tfaug.Augmentor(image=image_list, label=label_list)
                aug_rotate90.rotate90(probability=1)
                aug_ds.append(aug_rotate90(ds))
                aug_rotate180 = tfaug.Augmentor(image=image_list, label=label_list)
                aug_rotate180.rotate180(probability=1)
                aug_ds.append(aug_rotate180(ds))
                aug_rotate270 = tfaug.Augmentor(image=image_list, label=label_list)
                aug_rotate270.rotate270(probability=1)
                aug_ds.append(aug_rotate270(ds))
            if self.config.random_crop:
                aug_crop = tfaug.Augmentor(image=image_list, label=label_list)
                aug_crop.random_crop(scale_range=self.config.random_crop_range, probability=1)
                aug_ds.append(aug_crop(ds))
            for d in aug_ds:
                ds = ds.concatenate(d)
        return ds
    

    def load_weights(self, load_best=False, weights_only=False):
        weights_path = self.weights_best if load_best else self.weights_latest
        cp_file = tf.train.latest_checkpoint(weights_path)
        
        if cp_file is not None:
            self.model.load_weights(cp_file)
            parsed = os.path.basename(cp_file).split('_')
            disp = 'Model restored from'
            for i in range(1, len(parsed)):
                if parsed[i][:3] == 'sta':
                    # assert parsed[i][6:] == self.module_config[int(parsed[i][5])-1], "inconsistent module order configuration"
                    # disp = disp + ' Stage {:d} '.format(int(parsed[i][5])+1) + self.module_config[int(parsed[i][5])]
                    disp = disp + ' Stage {:d},'.format(int(parsed[i][5]))
                    self.training_stage = int(parsed[i][5])-1 if not weights_only else self.training_stage 
                if parsed[i][:3] == 'epo':
                    disp = disp + ' Epoch {:d}, '.format(int(parsed[i][5:]))
                    self.training_epoch = int(parsed[i][5:]) if not weights_only else self.training_epoch 
                if parsed[i][:3] == 'ste':
                    disp = disp + 'Step {:d}'.format(int(parsed[i][4:]))
                    self.training_step = int(parsed[i][4:]) if not weights_only else self.training_step 
                if parsed[i][:3] == 'val':
                    self.val_best = float(parsed[i][3:]) if not weights_only else self.val_best
            print(disp)
        else:
            print("Model not found!")
    
    def save_weights(self, stage_wise=False, save_best=False):
        
        # save_name = 'weights_stage' + str(self.training_stage+1) + self.module_config[self.training_stage] if self.training_stage else 'weights'
        save_name = 'weights_stage' + str(self.training_stage+1) if stage_wise else 'weights'
        save_name = save_name + '_epoch'+str(self.training_epoch)+'_step'+str(self.training_step)
        if save_best:
            save_name = save_name + '_val'+'{:.5f}'.format(float(self.val_best))
            if os.path.exists(self.weights_best):
                for f in os.listdir(self.weights_best):
                    os.remove(os.path.join(self.weights_best, f))
            self.model.save_weights(os.path.join(self.weights_best, save_name))
            print('Model saved at Stage {:d}, Step {:d}, Epoch {:d}'.format(self.training_stage+1, self.training_step, self.training_epoch))
        else:
            if os.path.exists(self.weights_latest):
                for f in os.listdir(self.weights_latest):
                    os.remove(os.path.join(self.weights_latest, f))
            self.model.save_weights(os.path.join(self.weights_latest, save_name))
            print('Model saved at Step {:d}, Epoch {:d}'.format(self.training_step, self.training_epoch))
    
    def predict_raw(self, image):

        img = np.squeeze(image)
        img = image_resize_np([img], (self.config.H, self.config.W))
        img = K.cast_to_floatx(img)

        pred = self.model(img)

        return [np.squeeze(np.array(p)) for p in pred]

