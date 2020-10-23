# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.model_base import InstSegBase
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

class InstSegCascade(InstSegBase):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)

    def _build(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        self.module_config = []
        output_list = []

        for m in self.config.module_order:
            if m == 'semantic' and self.config.semantic: 
                self.module_config.append('semantic')
            if m == 'dist' and self.config.dist: 
                self.module_config.append('dist')
            if m == 'embedding' and self.config.embedding: 
                self.module_config.append('embedding')
        
        self.feature_suppression = {}
        self.nets = {}
        self.outlayers = {}
        for i, m in enumerate(self.module_config):
            if i != 0:   
                self.feature_suppression[m] = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', 
                                                                     activation='linear', kernel_initializer='he_normal', 
                                                                     name='input_feature_'+m)
                if self.config.stop_gradient:
                    features = tf.stop_gradient(tf.identity(features))
                features = self.feature_suppression[m](features)
                input_list = [self.normalized_img, tf.nn.l2_normalize(features, axis=-1)]
            else:
                input_list = [self.normalized_img]

            self.nets[m] = UNnet(filters=self.config.filters,
                                 dropout_rate=self.config.dropout_rate,
                                 batch_norm=self.config.batch_norm,
                                 name='net_'+m)
            features = self.nets[m](K.concatenate(input_list, axis=-1))

            if m == 'semantic':
                self.outlayers[m] = keras.layers.Conv2D(filters=self.config.classes+1, 
                                                       kernel_size=1, padding='same', activation='softmax', 
                                                       kernel_initializer='he_normal', 
                                                       name='out_semantic')
                output_list.append(self.outlayers[m](features))
            if m == 'dist':
                activation = 'sigmoid' if self.config.loss_dist == 'binary_crossentropy' else 'linear'
                self.outlayers[m] = keras.layers.Conv2D(filters=1, 
                                                       kernel_size=1, padding='same', activation=activation, 
                                                       kernel_initializer='he_normal', 
                                                       name='out_dist')
                output_list.append(self.outlayers[m](features))
            if m == 'embedding':
                self.outlayers[m] = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                                       kernel_size=1, padding='same', activation='linear', 
                                                       kernel_initializer='he_normal', 
                                                       name='out_embedding')
                output_list.append(self.outlayers[m](features))         

        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
            tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=False, show_layer_names=True)
    
    def _validate(self, val_ds, metric='loss'):
        '''
        Return:
            improved: bool
        '''
        print('Running validation: ')
        val_loss = 0
        val_losses = {k: [] for k in self.module_config}
        for ds_item in val_ds:
            disp = 'validation: '
            outs = self.model(ds_item['image'])
            if len(self.module_config) == 1:
                outs = [outs]

            for k, v in zip(self.module_config, outs):
                if k == 'semantic':
                    val_losses['semantic'].append(float(self.loss_fn_semantic(ds_item['semantic'], v)))
                    disp += 'semantic loss: {:.5f} '.format(val_losses['semantic'][-1])
                elif k == 'dist':
                    val_losses['dist'].append(float(self.loss_fn_dist(ds_item['dist'], v)))
                    disp += 'dist loss: {:.5f} '.format(val_losses['dist'][-1])
                elif k == 'embedding':
                    val_losses['embedding'].append(float(self.loss_fn_embedding(ds_item['object'], v, ds_item['adj_matrix'])))
                    disp += 'embedding loss: {:.5f} '.format(val_losses['embedding'][-1])
            print(disp)

        val_losses = {k: np.mean(v) for k, v in val_losses.items()}
        for k, v in val_losses.items():
            if k == 'semantic':
                val_loss += v * self.config.weight_semantic
            elif k == 'dist':
                val_loss += v * self.config.weight_dist
            elif k == 'embedding':
                val_loss += v * self.config.weight_embedding
        # summary training loss
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=self.training_step)
            for k, v in val_losses.items():
                tf.summary.scalar('loss_'+k, v, step=self.training_step)
        # best score
        disp = "validation loss: {:.5f}".format(float(val_loss))
        for k, v in val_losses.items():
            disp += ', ' + k + ' loss: {:.5f}'.format(float(v))
        if val_loss < self.val_best:
            self.val_best = val_loss
            print("Improved: " + disp)
            return True
        else:
            print("Not improved: " + disp)
            return False

    def _stagewise_validate(self, val_ds, metric='loss'):
        '''
        Return:
            improved: bool
        '''
        print('Running Stage {:d} '.format(self.training_stage+1) + self.module_config[self.training_stage] + ' validation: ')
        val_losses = []
        m = self.module_config[self.training_stage]
        for ds_item in val_ds:
            outs = self.model(ds_item['image'])
            if len(self.module_config) == 1:
                outs = [outs]

            disp = 'validation: '
            if m == 'semantic':
                val_losses.append(float(self.loss_fn_semantic(ds_item['semantic'], outs[self.training_stage])))
                disp += 'semantic loss: {:.5f} '.format(val_losses[-1])
            elif m == 'dist':
                val_losses.append(float(self.loss_fn_dist(ds_item['dist'], outs[self.training_stage])))
                disp += 'dist loss: {:.5f} '.format(val_losses[-1])
            elif m == 'embedding':
                val_losses.append(float(self.loss_fn_embedding(ds_item['object'], outs[self.training_stage], ds_item['adj_matrix'])))
                disp += 'embedding loss: {:.5f} '.format(val_losses[-1])
            print(disp)

        val_loss = np.mean(val_losses)
        # summary training loss
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss_stage_'+self.module_config[self.training_stage], val_loss, step=self.training_step)
        # best score
        if val_loss < self.val_best:
            self.val_best = val_loss
            print("Improved validation loss: {:.5f}".format(float(val_loss)))
            return True
        else:
            print("Not improved validation loss: {:.5f}".format(float(val_loss)))
            return False
        

    def stagewise_train(self, train_data, validation_data=None, epochs=None, batch_size=None,
                        augmentation=True, image_summary=True):
        
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'object': ..., 'semantic': ...} 
                image (required): numpy array of size N x H x W x C 
                object (requeired): numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        TODO: tfrecords support
        '''
        # prepare network
        if not self.training_prepared:
            self._prepare_training()
        if epochs is None:
            epochs = self.config.train_epochs
        if batch_size is None:
            batch_size = self.config.train_batch_size

        # prepare data
        train_ds = self._ds_from_np(train_data)
        if augmentation:
            train_ds = self._ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
        val_ds = None if validation_data is None else self._ds_from_np(validation_data).batch(1)
            
        # load model
        self.load_weights()
        # self.training_stage = 1
        
        # train
        while self.training_stage < len(self.module_config):

            # if self.training_stage == 1:
            #     break

            print('training: ', self.module_config[self.training_stage], self.training_stage, self.training_epoch, self.training_step)
            
            # disable all module
            for m in self.module_config:
                if m in self.feature_suppression.keys():
                    self.feature_suppression[m].trainable = False
                self.nets[m].trainable = False
                self.outlayers[m].trainable = False
            # make the currently training module trainable
            m = self.module_config[self.training_stage]
            if m in self.feature_suppression.keys():
                self.feature_suppression[m].trainable = True
            self.nets[m].trainable = True
            self.outlayers[m].trainable = True

            if self.training_epoch == 0:
                print('load model!!!!!!')
                print(self.training_stage, self.training_epoch, self.training_step)
                self.load_weights(load_best=True, weights_only=True)
                self.val_best = float('inf')
                print(self.training_stage, self.training_epoch, self.training_step)

            for _ in range(max(epochs-self.training_epoch, 0)):
                for ds_item in train_ds:
                    with tf.GradientTape() as tape:
                        outs = self.model(ds_item['image'])
                        if len(self.module_config) == 1:
                            outs = [outs]

                        if self.module_config[self.training_stage] == 'semantic':
                            loss = self.loss_fn_semantic(ds_item['semantic'], outs[self.training_stage])
                        elif self.module_config[self.training_stage] == 'dist':
                            loss = self.loss_fn_dist(ds_item['dist'], outs[self.training_stage])
                        elif self.module_config[self.training_stage] == 'embedding':
                            loss = self.loss_fn_embedding(ds_item['object'], outs[self.training_stage], ds_item['adj_matrix'])
                        
                        grads = tape.gradient(loss, self.model.trainable_weights)
                        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                        # display trainig loss
                        self.training_step += 1
                        disp = "Stage {0:d}, Epoch {1:d}, Step {2:d}".format(self.training_stage+1, self.training_epoch+1, self.training_step)
                        disp += " with " + self.module_config[self.training_stage] + " loss: {:.5f}".format(float(loss))                        
                        print(disp)
                        # summary training loss
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar('loss_stage_'+self.module_config[self.training_stage], loss, step=self.training_step)
                        # summary output
                        if self.training_step % 200 == 0 and image_summary:
                            with self.train_summary_writer.as_default():
                                tf.summary.image('input_'+self.module_config[self.training_stage], tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                                
                                if 'semantic' == self.module_config[self.training_stage]:
                                    vis_semantic = tf.expand_dims(tf.argmax(outs[self.training_stage], axis=-1), axis=-1)
                                    tf.summary.image('stage_semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                                    tf.summary.image('stage_semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
                                if 'dist' == self.module_config[self.training_stage]:
                                    vis_dist = tf.cast(outs[self.training_stage]*255/tf.reduce_max(outs[self.training_stage]), tf.uint8)
                                    tf.summary.image('stage_dist', vis_dist, step=self.training_step, max_outputs=1)
                                    vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
                                    tf.summary.image('stage_dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
                                if 'embedding' == self.module_config[self.training_stage]:
                                    for i in range(self.config.embedding_dim//3):
                                        vis_embedding = outs[self.training_stage][:,:,:,3*i:3*(i+1)]
                                        tf.summary.image('stage_embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                        if not self.config.embedding_include_bg:
                                            vis_embedding = vis_embedding * tf.cast(ds_item['object'] > 0, vis_embedding.dtype)
                                            tf.summary.image('stage_embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                self.training_epoch += 1

                self.save_weights(stage_wise=True)
                if validation_data:
                    improved = self._stagewise_validate(val_ds)
                    if improved:
                        self.save_weights(stage_wise=True, save_best=True)
            
            self.training_epoch = 0
            self.training_step = 0
            self.val_best = float('inf')
            self.training_stage += 1

    def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
              augmentation=True, image_summary=True):
        
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'object': ..., 'semantic': ...} 
                image (required): numpy array of size N x H x W x C 
                object (requeired): numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        TODO: tfrecords support
        '''
        # prepare network
        if not self.training_prepared:
            self._prepare_training()
        if epochs is None:
            epochs = self.config.train_epochs
        if batch_size is None:
            batch_size = self.config.train_batch_size

        # prepare data
        train_ds = self._ds_from_np(train_data)
        if augmentation:
            train_ds = self._ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
        val_ds = None if validation_data is None else self._ds_from_np(validation_data).batch(1)
            
        # load model
        self.load_weights()

        # set all modules trainable
        for m in self.module_config:
            if m in self.feature_suppression.keys():
                self.feature_suppression[m].trainable = True
            self.nets[m].trainable = True
            self.outlayers[m].trainable = True
        
        # train
        for _ in range(epochs-self.training_epoch):
            for ds_item in train_ds:
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'])
                    if len(self.module_config) == 1:
                        outs = [outs]

                    losses = {}
                    loss = 0
                    for k, v in zip(self.module_config, outs):
                        if k == 'semantic':
                            losses['semantic'] = self.loss_fn_semantic(ds_item['semantic'], v)
                            loss += losses['semantic'] * self.config.weight_semantic
                        elif k == 'dist':
                            losses['dist'] = self.loss_fn_dist(ds_item['dist'], v)
                            loss += losses['dist'] * self.config.weight_dist
                        elif k == 'embedding':
                            losses['embedding'] = self.loss_fn_embedding(ds_item['object'], v, ds_item['adj_matrix'])
                            loss += losses['embedding'] * self.config.weight_embedding
                    
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # display trainig loss
                    self.training_step += 1
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}".format(self.training_epoch+1, self.training_step, float(loss))
                    for k, v in losses.items():
                        disp += ', ' + k + ' loss: {:.5f}'.format(float(v))
                    print(disp)
                    # summary training loss
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.training_step)
                        for k, v in losses.items():
                            tf.summary.scalar('loss_'+k, v, step=self.training_step)
                    # summary output
                    if self.training_step % 200 == 0 and image_summary:
                        with self.train_summary_writer.as_default():
                            tf.summary.image('input_img', tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                            outs_dict = {k: v for k, v in zip(self.module_config, outs)}

                            if 'semantic' in outs_dict.keys():
                                vis_semantic = tf.expand_dims(tf.argmax(outs_dict['semantic'], axis=-1), axis=-1)
                                tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                                tf.summary.image('semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
                            if 'dist' in outs_dict.keys():
                                vis_dist = tf.cast(outs_dict['dist']*255/tf.reduce_max(outs_dict['dist']), tf.uint8)
                                tf.summary.image('dist', vis_dist, step=self.training_step, max_outputs=1)
                                vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
                                tf.summary.image('dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
                            if 'embedding' in outs_dict.keys():
                                for i in range(self.config.embedding_dim//3):
                                    vis_embedding = outs_dict['embedding'][:,:,:,3*i:3*(i+1)]
                                    tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                    if not self.config.embedding_include_bg:
                                        vis_embedding = vis_embedding * tf.cast(ds_item['object'] > 0, vis_embedding.dtype)
                                        tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
 
            self.training_epoch += 1

            self.save_weights(stage_wise=False)
            if validation_data:
                improved = self._validate(val_ds)
                if improved:
                    self.save_weights(stage_wise=False, save_best=True)

    def predict_raw(self, image):

        image = np.squeeze(image)
        sz = self.config.image_size
        image = cv2.resize(image, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR)
        image = K.cast_to_floatx((image-image.mean())/(image.std()+1e-8)) 

        image = np.expand_dims(image, 0)
        if len(image.shape) == 3:
            image = np.expand_dims(image, -1)
        
        pred_raw = {m: p for m, p in zip(self.module_config, self.model.predict(image))}
        # semantic
        if 'semantic' in pred_raw.keys():
            pred_raw['semantic'] = np.squeeze(np.argmax(pred_raw['semantic'], axis=-1))
        # dist
        if 'dist' in pred_raw.keys():
            pred_raw['dist'] = np.squeeze(pred_raw['dist'])
        # embedding
        if 'embedding' in pred_raw.keys(): 
            embedding = np.squeeze(pred_raw['embedding'])
            pred_raw['embedding'] = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
        return pred_raw

    def predict(self, image, thres=0.7, similarity_thres=0.7, semantic_mask=None):
        
        pred = self.predict_raw(image)
        
        if semantic_mask is not None:
            pred['semantic'] = cv2.resize(semantic_mask.astype(np.uint16), (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)
        
        if len(pred) == 2 and 'semantic' not in pred.keys():
            seg = maskViaRegion(pred, thres=thres, similarity_thres=similarity_thres)
        if len(pred) == 2 and 'dist' not in pred.keys():
            pass
        if len(pred) == 2 and 'embedding' not in pred.keys():
            pass

        if len(pred) == 3:
            seg = maskViaRegion(pred, thres=thres, similarity_thres=similarity_thres)
            # seg = maskViaSeed(pred, thres=0.7, min_distance=5, similarity_thres=0.7)
            # seg = mutex(pred)
        seg = remove_noise(seg, pred['dist'], min_size=10, min_intensity=0.1)

        return seg, pred
