# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.uNet import *
from instSeg.utils import *
import instSeg.loss as L
from instSeg.post_process import *
from instSeg.evaluation import Evaluator
from skimage.measure import regionprops
import os
import numpy as np
from abc import abstractmethod

try:
    import tfAugmentor as tfaug 
    augemntor_available = True
except:
    augemntor_available = False

class ModelBase(object):

    def __init__(self, config, base_dir='./', run_name=''):
        self.config = config
        # model saving
        self.base_dir = os.path.join(base_dir, run_name)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.weights_latest = os.path.join(self.base_dir, 'weights_latest')
        self.weights_best = os.path.join(self.base_dir, 'weights_best')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.base_dir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(self.base_dir, 'val'))
        # training
        self.training_prepared = False
        self.training_stage = 0
        self.training_epoch = 0
        self.training_step = 0
        # augmentation
        self.augemntor_available = augemntor_available
        # validation
        self.best_score = None
        # build model
        self.build_model()

    @abstractmethod
    def build_model(self):
        ''' build the model self.self.model'''
        self.model = None

    def lr(self):
        p = self.training_step // self.config.lr_decay_period
        return self.config.train_learning_rate * (self.config.lr_decay_rate ** p)
    
    def prepare_training(self,
                         semantic=False,
                         dist=False,
                         embedding=False, 
                         contour=False):

        ''' prepare loss functions and optimizer'''

        if self.config.lr_decay_period != 0:
            self.optimizer = keras.optimizers.Adam(learning_rate=lambda : self.lr())
        else:
            self.optimizer = keras.optimizers.Adam(lr=self.config.train_learning_rate)

        self.loss_fns = {}
        # semantic loss
        if semantic:
            loss_semantic = {'crossentropy': L.ce, 
                             'binary_crossentropy': L.bce,
                             'weighted_binary_crossentropy': L.wbce,
                             'balanced_binary_corssentropy': L.bbce,
                             'dice': L.dice_union,
                             'object_dice': L.object_dice,
                             'dice_union': L.dice_union,
                             'dice_square': L.dice_square,
                             'generalised_dice': L.gdice,
                             'focal_loss': lambda y_true, y_pred: L.focal_loss(y_true, y_pred, gamma=self.config.focal_loss_gamma),
                             'sensitivity_specificity': lambda y_true, y_pred: L.sensitivity_specificity_loss(y_true, y_pred, beta=self.config.sensitivity_specificity_loss_beta)}
            self.loss_fns['semantic'] = loss_semantic[self.config.loss_semantic] 
        # distance regression loss
        if dist:
            loss_dist = {'binary_crossentropy': L.bce, 
                         'mse': lambda y_true, y_pred: L.mse(y_true, y_pred),
                         'huber': None,
                         'logcosh': None} 
            self.loss_fns['dist'] = loss_dist[self.config.loss_dist]
        # embedding loss
        if embedding:
            loss_embedding = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, self.config.max_obj, include_background=self.config.embedding_include_bg)}
            self.loss_fns['embedding'] = loss_embedding[self.config.loss_embedding]
        # contour loss
        if contour:
            loss_contour = {'binary_crossentropy': L.bce,
                            'weighted_binary_crossentropy': L.wbce,
                            'balanced_binary_corssentropy': L.bbce,
                            'dice': L.dice_union,
                            'focal_loss': lambda y_true, y_pred: L.binary_focal_loss(y_true, y_pred, gamma=2.0)}
            self.loss_fns['contour'] = loss_contour[self.config.loss_contour]

        self.training_prepared = True
    
    def ds_from_np(self, data,
                   instance=False, 
                   semantic=False,
                   dist=False,
                   embedding=False, 
                   contour=False):

        '''prepare training dataset'''

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

        '''augmentation if necessary'''

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
                    disp = disp + ' Stage {:d},'.format(int(parsed[i][5]))
                    self.training_stage = int(parsed[i][5])-1 if not weights_only else self.training_stage 
                if parsed[i][:3] == 'epo':
                    disp = disp + ' Epoch {:d}, '.format(int(parsed[i][5:]))
                    self.training_epoch = int(parsed[i][5:]) if not weights_only else self.training_epoch 
                if parsed[i][:3] == 'ste':
                    disp = disp + 'Step {:d}'.format(int(parsed[i][4:]))
                    self.training_step = int(parsed[i][4:]) if not weights_only else self.training_step 
                if parsed[i][:3] == 'val':
                    self.best_score = float(parsed[i][3:]) if not weights_only else self.best_score
            print(disp)
        else:
            print("Model not found!")
    
    def save_weights(self, stage_wise=False, save_best=False):
        
        save_name = 'weights_stage' + str(self.training_stage+1) if stage_wise else 'weights'
        save_name = save_name + '_epoch'+str(self.training_epoch)+'_step'+str(self.training_step)
        if save_best:
            save_name = save_name + '_val'+'{:.5f}'.format(float(self.best_score))
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
    

class InstSegMul(ModelBase):

    '''instance segmetation achieved by multi-tasking'''

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)

    @abstractmethod
    def build_model(self):
        ''' build the model self.self.model, the model output should be consistent with self.config.modules '''
        self.model = None

    def postprocess(self, raw, thres_contour=0.5, thres_emb=0.7, thres_dist=0.5, min_size=20):
        ''' return a tuple, the first item will be used by validate() for validation'''

        instances = None
        if len(raw) == 2:
            if 'semantic' in raw.keys() and 'contour' in raw.keys():
                instances = instance_from_semantic_and_contour(raw, thres_contour=thres_contour)
            if 'embedding' in raw.keys() and 'dist' in raw.keys():
                instances = instance_from_emb_and_dist(raw, thres_emb=thres_emb, thres_dist=thres_dist)
            if 'semantic' in raw.keys() and 'dist' in raw.keys():
                pass
        if len(self.config.modules) == 1:
            if 'semantic' in raw.keys():
                instances = np.squeeze(np.argmax(raw['semantic'], axis=-1)).astype(np.uint8)
            if 'embedding' in raw.keys():
                if self.embedding_cluster == 'argmax':
                    pass
                if self.embedding_cluster == 'meanshift':
                    pass
                if self.embedding_cluster == 'mws':
                    pass
            if 'dist' in raw.keys():
                pass
            
        if instances is not None and min_size > 0:
            for r in regionprops(instances):
                if r.area < min_size:
                    instances[r.coords[:,0], r.coords[:,1]] = 0

        return instances

    def validate(self, val_ds, save_best=True):
        '''
        Args:
            val_ds: validation dataset
        '''
        if val_ds is not None:
            print('Running validation: ')
            e, losses = Evaluator(dimension=2, mode='area'), []
            for ds_item in val_ds:
                outs = self.model(ds_item['image'])
                if self.config.save_best_metric == 'loss':
                    loss = 0
                    outs = [outs] if len(self.config.modules) == 1 else outs
                    for m, out in zip(self.config.modules, outs):
                        l = self._modules_loss(m, out, ds_item)
                        print('validation example with loss: {:5f}'.format(l))
                        loss += l
                    losses.append(loss)
                else:
                    instances = self.postprocess(outs)
                    if isinstance(instances, tuple):
                        instances = instances[0]
                    if instances is not None:
                        e.add_example(instances, np.squeeze(ds_item['instance']))
            with self.val_summary_writer.as_default():
                if self.config.save_best_metric == 'loss':
                    score = np.mean(losses)
                    tf.summary.scalar('validation loss', score, step=self.training_step)
                    disp = 'validation loss: {:.5f}'.format(score)
                else:
                    mAP, mAJ = e.mAP(), e.mAJ()
                    # summary training loss
                    tf.summary.scalar('validation mAP', mAP, step=self.training_step)
                    tf.summary.scalar('validation mAJ', mAJ, step=self.training_step)
                    # best score
                    disp = 'validation mAP: {:.5f}, mAJ: {:.5f}'.format(mAP, mAJ)
                    if self.config.save_best_metric == 'mAP':
                        score = mAP
                    else: # use mAJ
                        score= mAJ
                print(disp)

            if self.best_score is None:
                self.best_score = score

            if self.config.save_best_metric == 'loss':
                score_cp, best_score_cp = - score, - self.best_score
            else:
                score_cp, best_score_cp = score, self.best_score

            if score_cp > best_score_cp:
                self.best_score = score
                print("Validation Score Improved: " + disp)
                if save_best:
                    self.save_weights(save_best=True)
            else:
                print("Validation Score Not Improved: " + disp)

    def _modules_loss(self, module, out, ds_item):
        if module == 'semantic':
            gt = ds_item['instance'] if self.config.loss_semantic == 'object_dice' else ds_item['semantic']
            return self.loss_fns['semantic'](gt, out) * self.config.weight_semantic
        elif module == 'contour':
            return self.loss_fns['contour'](ds_item['contour'], out) * self.config.weight_contour
        elif module == 'dist':
            return self.loss_fns['dist'](ds_item['dist'], out) * self.config.weight_dist
        elif module == 'embedding':
            return self.loss_fns['embedding'](ds_item['instance'], out, ds_item['adj_matrix']) * self.config.weight_embedding


    def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
              augmentation=True, image_summary=True):
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'instance': ..., 'semantic': ...} 
                image (required): numpy array of size N x H x W x C 
                instance (reuqired): numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        '''
        modules_dict = {m: True for m in self.config.modules}
        # prepare network
        if not self.training_prepared:
            self.prepare_training(**modules_dict)
        epochs = self.config.train_epochs if epochs is None else epochs
        batch_size = self.config.train_batch_size if batch_size is None else batch_size

        # prepare data
        if 'semantic' in self.config.modules and self.config.loss_semantic == 'object_dice':
            modules_dict['semantic'] = False
            modules_dict['instance'] = True
        train_ds = self.ds_from_np(train_data, **modules_dict)
        if augmentation:
            train_ds = self.ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
        if validation_data is None or len(validation_data['image']) == 0:
            val_ds = None
        else:
            modules_dict['instance'] = True
            if self.config.save_best_metric == 'loss':
                val_ds = self.ds_from_np(validation_data, **modules_dict).batch(1)
            else:
                val_ds = self.ds_from_np(validation_data, instance=True).batch(1)
            
        # load model
        self.load_weights()

        # train
        for _ in range(epochs-self.training_epoch):
            for ds_item in train_ds:
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'])
                    if len(self.config.modules) == 1:
                        outs = [outs]

                    losses, loss = {}, 0
                    for m, out in zip(self.config.modules, outs):
                        losses[m] = self._modules_loss(m, out, ds_item)
                        loss += losses[m]
                    
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # display trainig loss
                    self.training_step += 1
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}".format(self.training_epoch+1, self.training_step, float(loss))
                    for m, l in losses.items():
                        disp += ', ' + m + ' loss: {:.5f}'.format(float(l))
                    print(disp)
                    # summary training loss
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.training_step)
                        for m, l in losses.items():
                            tf.summary.scalar('loss_'+m, l, step=self.training_step)
                    # summary output
                    if self.training_step % 200 == 0 and image_summary:
                        with self.train_summary_writer.as_default():
                            tf.summary.image('input_img', tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                            outs_dict = {k: v for k, v in zip(self.config.modules, outs)}
                            # semantic
                            if 'semantic' in outs_dict.keys():
                                vis_semantic = tf.expand_dims(tf.argmax(outs_dict['semantic'], axis=-1), axis=-1)
                                tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                                gt = ds_item['instance'] if self.config.loss_semantic == 'object_dice' else ds_item['semantic']
                                tf.summary.image('semantic_gt', tf.cast(gt*255/tf.reduce_max(gt), tf.uint8), step=self.training_step, max_outputs=1)
                            # contour
                            if 'contour' in outs_dict.keys():
                                vis_contour = tf.cast(outs_dict['contour']*255, tf.uint8)
                                tf.summary.image('contour', vis_contour, step=self.training_step, max_outputs=1)
                                vis_contour_gt = tf.cast(ds_item['contour'], tf.uint8) * 255
                                tf.summary.image('contour_gt', vis_contour_gt, step=self.training_step, max_outputs=1)
                            # dist regression
                            if 'dist' in outs_dict.keys():
                                vis_dist = tf.cast(outs_dict['dist']*255/tf.reduce_max(outs_dict['dist']), tf.uint8)
                                tf.summary.image('dist', vis_dist, step=self.training_step, max_outputs=1)
                                vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
                                tf.summary.image('dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
                            # embedding
                            if 'embedding' in outs_dict.keys():
                                for i in range(self.config.embedding_dim//3):
                                    vis_embedding = outs_dict['embedding'][:,:,:,3*i:3*(i+1)]
                                    tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                    if not self.config.embedding_include_bg:
                                        vis_embedding = vis_embedding * tf.cast(ds_item['object'] > 0, vis_embedding.dtype)
                                        tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
 
            self.training_epoch += 1

            self.save_weights(stage_wise=False)
            if self.training_epoch >= self.config.validation_start_epoch:
                self.validate(val_ds, save_best=True)

    def predict(self, image, keep_size=True):
        
        sz = image.shape
        # model inference
        img = np.squeeze(image)
        img = image_resize_np([img], (self.config.H, self.config.W))
        img = K.cast_to_floatx(img)
        raw = self.model(img)
        raw = {m: o for m, o in zip(self.config.modules, raw)}
        # post processing
        instances = self.postprocess(raw)
        # resize to original resolution
        if keep_size:
            instances = cv2.resize(instances, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        return instances, raw