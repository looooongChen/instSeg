import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.uNet import *
from instSeg.utils import *
import instSeg.loss as L
from instSeg.enumDef import *
from instSeg.post_process import *
from instSeg.evaluation import Evaluator
from skimage.measure import regionprops
import os
import copy
import numpy as np
from abc import abstractmethod
from instSeg.tfAugmentor.Augmentor import Augmentor 

try:
    augmntor_available = True
except:
    augmntor_available = False

class ModelBase(object):

    def __init__(self, config, model_dir='./'):
        self.config = config
        config.model_type = MODEL_BASE
        # model saving
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.weights_latest = os.path.join(self.model_dir, 'weights_latest')
        self.weights_best = os.path.join(self.model_dir, 'weights_best')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.model_dir, 'train'))
        self.val_summary_writer = tf.summary.create_file_writer(os.path.join(self.model_dir, 'val'))
        # training
        self.training_prepared = False
        self.training_epoch = 0
        self.training_step = 0
        # augmentation
        self.augmntor_available = augmntor_available
        # validation
        self.best_score = None
        # input positional aug
        coordX, coordY = tf.meshgrid(tf.range(0,self.config.W), tf.range(0,self.config.H))
        coordX, coordY = coordX / self.config.W - 0.5, coordY / self.config.H - 0.5
        coords = tf.expand_dims(tf.stack([coordX, coordY], axis=-1), axis=0)
        if self.config.positional_input == 'globalCoords':
            self.coords = coords
        elif self.config.positional_input == 'octave0':
            self.coords = tf.math.sin(coords)
        elif self.config.positional_input == 'octave1':
            self.coords = tf.math.sin(2*coords)
        elif self.config.positional_input == 'octave2':
            self.coords = tf.math.sin(4*coords)
        elif self.config.positional_input == 'octave3':
            self.coords = tf.math.sin(8*coords)
        elif self.config.positional_input == 'octave4':
            self.coords = tf.math.sin(16*coords)
        # build model
        self.build_model()

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

        self.loss_fns = {}
        # semantic loss
        loss_semantic = {'crossentropy': L.ce, 
                         'binary_crossentropy': L.bce,
                         'weighted_binary_crossentropy': L.wbce,
                         'balanced_binary_corssentropy': L.bbce,
                         'dice': L.mdice,
                         'binary_dice': L.binary_dice,
                         'generalised_dice': L.gdice,
                         'mean_dice': L.mdice,
                         'focal_loss': lambda y_true, y_pred: L.focal_loss(y_true, y_pred, gamma=self.config.focal_loss_gamma),
                         'sensitivity_specificity': lambda y_true, y_pred: L.sensitivity_specificity_loss(y_true, y_pred, beta=self.config.sensitivity_specificity_loss_beta)}
        self.loss_fns['semantic'] = loss_semantic[self.config.semantic_loss] 
        # distance regression loss
        loss_reg = {'binary_crossentropy': L.bce, 
                    'mse': L.mse,
                    'masked_mse': L.masked_mse,
                    'huber': None,
                    'logcosh': None} 
        self.loss_fns['edt'] = loss_reg[self.config.edt_loss]
        # flow regression
        self.loss_fns['edt_flow'] = loss_reg[self.config.edt_flow_loss]
        # embedding loss
        loss_embedding = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, self.config.max_obj, include_background=self.config.embedding_include_bg)}
        self.loss_fns['embedding'] = loss_embedding[self.config.embedding_loss]
        # contour loss
        loss_contour = {'binary_crossentropy': L.bce,
                        'weighted_binary_crossentropy': L.wbce,
                        'balanced_binary_corssentropy': L.bbce,
                        'dice': L.binary_dice,
                        'focal_loss': lambda y_true, y_pred: L.binary_focal_loss(y_true, y_pred, gamma=2.0)}
        self.loss_fns['contour'] = loss_contour[self.config.contour_loss]

        self.training_prepared = True
    
    def data_loader(self, data, modules, keep_instance=False):

        '''prepare training dataset'''

        for k in data.keys():
            if k == 'image':
                data[k] = image_resize_np(data[k], (self.config.H, self.config.W))
                data[k] = K.cast_to_floatx(data[k])
            else:
                data[k] = image_resize_np(data[k], (self.config.H, self.config.W), method='nearest')

        data['instance'] = relabel_instance(data['instance']).astype(np.int32)
        data_to_keep = ['image']

        for m in modules:

            if m == 'semantic':
                if 'semantic' in data.keys():
                    data_to_keep.append('semantic')
                elif self.config.semantic_in_ram:
                    data_to_keep.append('semantic')
                    data['semantic'] = (data['instance']>0).astype(np.uint8) 
                else:
                    keep_instance = True

            # keep edt in ram may cause error, if augmentation used
            if m == 'edt': 
                if self.config.edt_in_ram:
                    data_to_keep.append('edt')
                    data['edt'] = edt(data['instance'], normalize=self.config.edt_normalize, process_disp=True)
                    if self.config.edt_normalize:
                        data['edt'] = data['edt'] * 10
                    if self.config.edt_loss.startswith('masked'):
                        keep_instance = True
                else:
                    keep_instance = True

            # keep edt in ram may cuase error, if augmentation used
            if m == 'edt_flow':
                if self.config.edt_flow_in_ram:
                    data_to_keep.append('edt_flow')
                    data['edt_flow'] = 10 * edt_flow(data['instance'], normalize=self.config.edt_normalize, bg=False, process_disp=True)
                    if self.config.edt_flow_loss.startswith('masked'):
                        keep_instance = True
                else:
                    keep_instance = True
                
            if m == 'embedding':
                data_to_keep.append('adj_matrix')
                data['adj_matrix'] = adj_matrix(data['instance'], self.config.neighbor_distance, self.config.max_obj)
                keep_instance = True

            if m == 'contour':
                if self.config.contour_in_ram:
                    data_to_keep.append('contour')
                    data['contour'] = contour(data['instance'], radius=self.config.contour_radius)
                else:
                    keep_instance = True
        
        if keep_instance:
            data_to_keep.append('instance')

        for k in list(data.keys()):
            if k not in data_to_keep:
                del data[k]

        # return data
        return tf.data.Dataset.from_tensor_slices(data)

    def get_training_batch(self, ds_item):
        for m in self.config.modules:
            if m not in ds_item.keys():
                if m == 'semantic':
                    ds_item['semantic'] = tf.cast(ds_item['instance']>0, tf.int8)
                if m == 'contour':
                    ds_item['contour'] = contour(ds_item['instance'], radius=self.config.contour_radius, process_disp=False)
                if m == 'edt':
                    ds_item['edt'] = edt(ds_item['instance'], normalize=self.config.edt_normalize, process_disp=False)
                    if self.config.edt_normalize:
                        ds_item['edt'] = 10 * ds_item['edt']
                if m == 'edt_flow':
                    ds_item['edt_flow'] =  10 * edt_flow(ds_item['instance'], normalize=self.config.edt_flow_normalize, bg=False, process_disp=False)
        return ds_item
    
    def ds_augment(self, ds):

        '''augmentation if necessary'''

        if augmntor_available:
            image_list = ['image']
            label_list = []
            for m in ds.element_spec.keys():
                if m == 'instance':
                    label_list.append('instance')
                if m == 'semantic':
                    label_list.append('semantic')
                if m == 'edt':
                    print('WARNING: augment euclidean distance transform may be problematic!')
                    image_list.append('edt')
                if m == 'edt_flow':
                    assert False, 'Directly augmenting flow is not supported, set config.edt_flow_in_ram to False'
                if m == 'contour':
                    label_list.append('contour')

            # keys = image_list + label_list
            keys = copy.deepcopy(ds.element_spec)
            for k, _ in keys.items():
                keys[k] = k
            aug_ds = []
            if self.config.flip:
                # aug_flip = tfaug.Augmentor(signature=keys, image=image_list, label=label_list)
                aug_flip = Augmentor(signature=keys, image=image_list, label=label_list)
                aug_flip.flip_left_right(probability=0.8)
                aug_flip.flip_up_down(probability=0.8)
                aug_ds.append(aug_flip(ds))
            if self.config.elastic_deform:
                # aug_elas = tfaug.Augmentor(signature=keys, image=image_list, label=label_list)
                aug_elas = Augmentor(signature=keys, image=image_list, label=label_list)
                aug_elas.elastic_deform(strength=self.config.elastic_strength, scale=self.config.elastic_scale, probability=1)
                aug_ds.append(aug_elas(ds))
            if self.config.random_rotation:
                # aug_rotation = tfaug.Augmentor(signature=keys, image=image_list, label=label_list)
                aug_rotation = Augmentor(signature=keys, image=image_list, label=label_list)
                aug_rotation.random_rotate(probability=1)
                aug_ds.append(aug_rotation(ds))
            if self.config.random_crop:
                # aug_crop = tfaug.Augmentor(signature=keys, image=image_list, label=label_list)
                aug_crop = Augmentor(signature=keys, image=image_list, label=label_list)
                aug_crop.random_crop(scale_range=self.config.random_crop_range, probability=1)
                aug_ds.append(aug_crop(ds))
            if self.config.random_gamma:
                # aug_gamma = tfaug.Augmentor(signature=keys, image=image_list, label=label_list)
                aug_gamma = Augmentor(signature=keys, image=image_list, label=label_list)
                aug_gamma.random_gamma(gamma_range=self.config.random_gamma_range, probability=1)
                aug_ds.append(aug_gamma(ds))

            for d in aug_ds:
                ds = ds.concatenate(d)
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
    
    # def save_weights(self, stage_wise=False, save_best=False):
    def save_weights(self, save_best=False):
        
        # save_name = 'weights_stage' + str(self.training_stage+1) if stage_wise else 'weights'
        save_name = 'weights' + '_epoch'+str(self.training_epoch)+'_step'+str(self.training_step)
        if save_best:
            save_name = save_name + '_val'+'{:.5f}'.format(float(self.best_score))
            if os.path.exists(self.weights_best):
                for f in os.listdir(self.weights_best):
                    os.remove(os.path.join(self.weights_best, f))
            self.model.save_weights(os.path.join(self.weights_best, save_name))
            # print('Model saved at Stage {:d}, Step {:d}, Epoch {:d}'.format(self.training_stage+1, self.training_step, self.training_epoch))
            print('Model saved at Step {:d}, Epoch {:d}'.format(self.training_step, self.training_epoch))
        else:
            if os.path.exists(self.weights_latest):
                for f in os.listdir(self.weights_latest):
                    os.remove(os.path.join(self.weights_latest, f))
            self.model.save_weights(os.path.join(self.weights_latest, save_name))
            print('Model saved at Step {:d}, Epoch {:d}'.format(self.training_step, self.training_epoch))
        
        self.config.save(os.path.join(self.model_dir, 'config.pkl'))

    def predict_raw(self, image, keep_size=True):
        
        sz = image.shape
        # model inference
        img = np.squeeze(image)
        img = image_resize_np([img], (self.config.H, self.config.W))
        img = K.cast_to_floatx(img)
        raw = self.model(img)
        raw = {m: np.squeeze(o) for m, o in zip(self.config.modules, raw)}
        # resize to original resolution
        if keep_size:
            for k in raw.keys():
                if len(raw[k].shape) == 3:
                    resized = np.zeros((sz[0], sz[1], raw[k].shape[-1]))
                    for i in range(raw[k].shape[-1]):
                        resized[:,:,i] = cv2.resize(raw[k][:,:,i], (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)
                    raw[k] = resized
                else: 
                    raw[k] = cv2.resize(raw[k], (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)

        return raw
    

class InstSegMul(ModelBase):

    '''instance segmetation achieved by multi-tasking'''

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_INST

    @abstractmethod
    def build_model(self):
        ''' build the model self.self.model, the model output should be consistent with self.config.modules '''
        self.model = None

    def postprocess(self, raw):
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
        if len(raw) == 2:
            if 'semantic' in raw.keys() and 'contour' in raw.keys():
                instances = instance_from_semantic_and_contour(raw, self.config)
            if 'embedding' in raw.keys() and 'edt' in raw.keys():
                instances = instance_from_emb_and_edt(raw, self.config)
            if 'semantic' in raw.keys() and 'edt' in raw.keys():
                instances = instance_from_edt_and_semantic(raw, self.config)
            if 'edt_flow' in raw.keys() and 'semantic' in raw.keys():
                mask = np.squeeze(np.argmax(raw['semantic'], axis=-1)).astype(np.uint16)
                flow = np.squeeze(raw['edt_flow'])/10
                instances = instance_from_flow(flow, mask, self.config)

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
            if 'edt' in raw.keys():
                instances = instance_from_edt(raw, self.config)
            
        if self.config.obj_min_size > 0 or self.config.obj_max_size < float('inf'):
            for r in regionprops(instances):
                if r.area < self.config.obj_min_size or r.area > self.config.obj_max_size:
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
                        ds_item = self.get_training_batch(ds_item)
                        l = self._module_loss(m, out, ds_item)
                        print('validation example with loss: {:5f}'.format(l))
                        loss += l
                    losses.append(loss)
                else:
                    instances = self.postprocess({m: o for m, o in zip(self.config.modules, outs)})
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

    def _module_loss(self, module, out, ds_item):
        if module == 'semantic':
            return self.loss_fns['semantic'](ds_item['semantic'], out) * self.config.semantic_weight
        elif module == 'contour':
            return self.loss_fns['contour'](ds_item['contour'], out) * self.config.contour_weight
        elif module == 'edt':
            if self.config.edt_loss.startswith('masked'):
                return self.loss_fns['edt'](ds_item['edt'], out, ds_item['instance']) * self.config.edt_weight
            else:
                return self.loss_fns['edt'](ds_item['edt'], out) * self.config.edt_weight
        elif module == 'edt_flow':
            if self.config.edt_flow_loss.startswith('masked'):
                return self.loss_fns['edt_flow'](ds_item['edt_flow'], out, ds_item['instance']) * self.config.edt_flow_weight
            else:
                return self.loss_fns['edt_flow'](ds_item['edt_flow'], out) * self.config.edt_flow_weight
        elif module == 'embedding':
            return self.loss_fns['embedding'](ds_item['instance'], out, ds_item['adj_matrix']) * self.config.embedding_weight


    def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
              augmentation=True, image_summary=True):
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'instance': ..., 'semantic': ...} 
                image (required): numpy array of size N x H x W x C 
                instance (reuqired): numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        '''
        # prepare network
        if not self.training_prepared:
            self.prepare_training()
        epochs = self.config.train_epochs if epochs is None else epochs
        batch_size = self.config.train_batch_size if batch_size is None else batch_size

        # prepare data
        train_ds = self.data_loader(train_data, self.config.modules)
        if augmentation:
            train_ds = self.ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size)
        if validation_data is None or len(validation_data['image']) == 0:
            val_ds = None
        else:
            if self.config.save_best_metric == 'loss':
                val_ds = self.data_loader(validation_data, self.config.modules).batch(1)
            else:
                val_ds = self.data_loader(validation_data, ['image'], keep_instance=True).batch(1)
            
        # load model
        self.load_weights()

        # train
        for _ in range(epochs-self.training_epoch):
            for ds_item in train_ds:
                ds_item = self.get_training_batch(ds_item)
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'], training=True)
                    if len(self.config.modules) == 1:
                        outs = [outs]

                    losses, loss = {}, 0
                    for m, out in zip(self.config.modules, outs):
                        losses[m] = self._module_loss(m, out, ds_item)
                        loss += losses[m]
                    loss += sum(self.model.losses)
                    
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
                            vis_image = ds_item['image']/tf.math.reduce_max(ds_item['image'], axis=[1,2], keepdims=True)*255
                            tf.summary.image('input_img', tf.cast(vis_image, tf.uint8), step=self.training_step, max_outputs=1)
                            outs_dict = {k: v for k, v in zip(self.config.modules, outs)}
                            # semantic
                            if 'semantic' in outs_dict.keys():
                                vis_semantic = tf.expand_dims(tf.argmax(outs_dict['semantic'], axis=-1), axis=-1)
                                tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                                gt = tf.cast(ds_item['semantic'], tf.int32)
                                tf.summary.image('semantic_gt', tf.cast(gt*255/tf.reduce_max(gt), tf.uint8), step=self.training_step, max_outputs=1)
                            # contour
                            if 'contour' in outs_dict.keys():
                                vis_contour = tf.cast(outs_dict['contour']*255, tf.uint8)
                                tf.summary.image('contour', vis_contour, step=self.training_step, max_outputs=1)
                                vis_contour_gt = tf.cast(ds_item['contour'], tf.uint8) * 255
                                tf.summary.image('contour_gt', vis_contour_gt, step=self.training_step, max_outputs=1)
                            # edt regression
                            if 'edt' in outs_dict.keys():
                                vis_edt = tf.cast(outs_dict['edt']*255/tf.reduce_max(outs_dict['edt']), tf.uint8)
                                tf.summary.image('edt', vis_edt, step=self.training_step, max_outputs=1)
                                vis_edt_gt = tf.cast(ds_item['edt']*255/tf.reduce_max(ds_item['edt']), tf.uint8)
                                tf.summary.image('edt_gt', vis_edt_gt, step=self.training_step, max_outputs=1)
                            # embedding
                            if 'embedding' in outs_dict.keys():
                                for i in range(self.config.embedding_dim//3):
                                    vis_embedding = outs_dict['embedding'][:,:,:,3*i:3*(i+1)]
                                    tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                    if not self.config.embedding_include_bg:
                                        vis_embedding = vis_embedding * tf.cast(ds_item['instance'] > 0, vis_embedding.dtype)
                                        tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
 
            self.training_epoch += 1

            self.save_weights()
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