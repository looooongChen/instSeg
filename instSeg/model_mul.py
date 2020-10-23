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


class InstSegMul(InstSegBase):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)

    def _build(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        output_list = []

        assert self.config.semantic and self.config.dist and self.config.embedding, "all modules should be activated in 'multitask' model, otherwise use 'cascade' model " 

        self.net_stage1 = UNnet(filters=self.config.filters,
                                dropout_rate=self.config.dropout_rate,
                                batch_norm=self.config.batch_norm,
                                name='net_semantic_dist')
        self.outlayer_semantic = keras.layers.Conv2D(filters=self.config.classes+1, 
                                                     kernel_size=1, padding='same', activation='softmax', 
                                                     kernel_initializer='he_normal', 
                                                     name='out_semantic')
        activation = 'sigmoid' if self.config.loss_dist == 'binary_crossentropy' else 'linear'
        self.outlayer_dist = keras.layers.Conv2D(filters=1, 
                                                 kernel_size=1, padding='same', activation=activation, 
                                                 kernel_initializer='he_normal', 
                                                 name='out_dist')
        self.feature_suppression = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', 
                                                          activation='linear', kernel_initializer='he_normal', 
                                                          name='embedding_feature_suppression')
        self.net_stage2 = UNnet(filters=self.config.filters,
                                dropout_rate=self.config.dropout_rate,
                                batch_norm=self.config.batch_norm,
                                name='net_embedding')
        self.outlayer_embedding = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                                      kernel_size=1, padding='same', activation='linear', 
                                                      kernel_initializer='he_normal', 
                                                      name='out_embedding') 

        features = self.net_stage1(self.normalized_img)
        out_semantic = self.outlayer_semantic(features)
        out_dist = self.outlayer_dist(features)
        if self.config.stop_gradient:
            features = tf.stop_gradient(tf.identity(features))
        features = self.feature_suppression(features)
        features = tf.nn.l2_normalize(features, axis=-1)
        features_embedding = self.net_stage2(K.concatenate([self.normalized_img, features], axis=-1))
        out_embedding = self.outlayer_embedding(features_embedding)

        self.model = keras.Model(inputs=self.input_img, outputs=[out_semantic, out_dist, out_embedding])
        
        if self.config.verbose:
            self.model.summary()
            tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=False, show_layer_names=True)
    
    def _validate(self, val_ds, metric='loss'):
        '''
        Return:
            improved: bool
        '''
        print('Running validation: ')
        val_loss_semantic = []
        val_loss_dist = []
        val_loss_embedding = []
        val_loss = []
        for ds_item in val_ds:
            outs = self.model(ds_item['image'])

            val_loss_semantic.append(float(self.loss_fn_semantic(ds_item['semantic'], outs[0])))
            val_loss_dist.append(float(self.loss_fn_dist(ds_item['dist'], outs[1])))
            val_loss_embedding.append(float(self.loss_fn_embedding(ds_item['object'], outs[2], ds_item['adj_matrix'])))

            val_loss.append(val_loss_semantic[-1] * self.config.weight_semantic + val_loss_dist[-1] * self.config.weight_dist + val_loss_embedding[-1] * self.config.weight_embedding)
            disp = 'validation loss {:.5f}, semantic loss {:.5f}, dist loss {:.5f}, embedding loss {:.5f}'.format(val_loss[-1], val_loss_semantic[-1], val_loss_dist[-1], val_loss_embedding[-1])
            print(disp)

        # summary training loss
        val_loss, val_loss_semantic, val_loss_dist, val_loss_embedding = np.mean(val_loss), np.mean(val_loss_semantic), np.mean(val_loss_dist), np.mean(val_loss_embedding)
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=self.training_step)
            tf.summary.scalar('loss_semantic', val_loss_semantic, step=self.training_step)
            tf.summary.scalar('loss_dist', val_loss_dist, step=self.training_step)
            tf.summary.scalar('loss_embedding', val_loss_embedding, step=self.training_step)


        # best score
        disp = 'validation loss {:.5f}, semantic loss {:.5f}, dist loss {:.5f}, embedding loss {:.5f}'.format(val_loss, val_loss_semantic, val_loss_semantic, val_loss_embedding)
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
        print('Running Stage {:d} validation'.format(self.training_stage+1))
        val_loss_semantic = []
        val_loss_dist = []
        val_loss = []
        
        for ds_item in val_ds:
            outs = self.model(ds_item['image'])

            disp = 'validation: '

            if self.training_stage == 0:
                val_loss_semantic.append(self.loss_fn_semantic(ds_item['semantic'], outs[0]))
                val_loss_dist.append(self.loss_fn_dist(ds_item['dist'], outs[1]))
                val_loss.append(val_loss_semantic[-1] * self.config.weight_semantic + val_loss_dist[-1] * self.config.weight_dist) 
                disp += 'loss: {:.5f} '.format(val_loss[-1])
                disp += 'semantic loss: {:.5f} '.format(val_loss_semantic[-1])
                disp += 'dist loss: {:.5f} '.format(val_loss_dist[-1])
            else:
                val_loss.append(self.loss_fn_embedding(ds_item['object'], outs[2], ds_item['adj_matrix']))
                disp += 'embedding loss: {:.5f} '.format(val_loss[-1])
            print(disp)

        with self.val_summary_writer.as_default():
            val_loss = np.mean(val_loss)
            if self.training_stage == 0:
                val_loss_semantic = np.mean(val_loss_semantic)
                tf.summary.scalar('loss_stage1_semantic', val_loss_semantic, step=self.training_step)
                val_loss_dist = np.mean(val_loss_dist)
                tf.summary.scalar('loss_stage1_dist', val_loss_dist, step=self.training_step)
                tf.summary.scalar('loss_stage1', val_loss, step=self.training_step)
            else:
                tf.summary.scalar('loss_stage2_embedding', val_loss, step=self.training_step)

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
        
        # train
        while self.training_stage < 2:
            
            if self.training_stage == 0:
                self.net_stage1.trainable = True
                self.outlayer_semantic.trainable = True
                self.outlayer_dist.trainable = True
                self.feature_suppression.trainable = False
                self.net_stage2.trainable = False
                self.outlayer_embedding.trainable = False
            else:
                self.net_stage1.trainable = False
                self.outlayer_semantic.trainable = False
                self.outlayer_dist.trainable = False
                self.feature_suppression.trainable = True
                self.net_stage2.trainable = True
                self.outlayer_embedding.trainable = True

            if self.training_epoch == 0:
                self.load_weights(load_best=True, weights_only=True)
                self.val_best = float('inf')

            for _ in range(max(0, epochs-self.training_epoch)):
                for ds_item in train_ds:
                    with tf.GradientTape() as tape:
                        outs = self.model(ds_item['image'])

                        if self.training_stage == 0:
                            loss_semantic = self.loss_fn_semantic(ds_item['semantic'], outs[0])
                            loss_dist = self.loss_fn_dist(ds_item['dist'], outs[1])
                            loss = loss_semantic * self.config.weight_semantic + loss_dist * self.config.weight_dist
                        else:
                            loss = self.loss_fn_embedding(ds_item['object'], outs[2], ds_item['adj_matrix'])
                        
                        grads = tape.gradient(loss, self.model.trainable_weights)
                        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                        # display trainig loss
                        self.training_step += 1
                        disp = "Stage {0:d} , Epoch {1:d}, Step {2:d}".format(self.training_stage+1, self.training_epoch+1, self.training_step)
                        if self.training_stage == 0:
                            disp += " with loss {:.5f}, semantic loss {:.5f}, dist loss {:.5f}".format(float(loss), float(loss_semantic), float(loss_dist))
                        else:                        
                            disp += " with embedding loss {:.5f}".format(float(loss))
                        print(disp)
                        # summary training loss
                        with self.train_summary_writer.as_default():
                            if self.training_stage == 0:
                                tf.summary.scalar('loss_stage1', loss, step=self.training_step)
                                tf.summary.scalar('loss_stage1_semantic', loss_semantic, step=self.training_step)
                                tf.summary.scalar('loss_stage1_dist', loss, step=self.training_step)
                            else:
                                tf.summary.scalar('loss_stage2_embedding', loss, step=self.training_step)
                        # summary output
                        if self.training_step % 200 == 0 and image_summary:
                            with self.train_summary_writer.as_default():
                                tf.summary.image('input_stage'+str(self.training_stage+1), tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                                
                                if self.training_stage == 0:
                                    # semantic
                                    vis_semantic = tf.expand_dims(tf.argmax(outs[0], axis=-1), axis=-1)
                                    tf.summary.image('stage1_semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                                    tf.summary.image('stage1_semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
                                    # dist
                                    vis_dist = tf.cast(outs[1]*255/tf.reduce_max(outs[1]), tf.uint8)
                                    tf.summary.image('stage1_dist', vis_dist, step=self.training_step, max_outputs=1)
                                    vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
                                    tf.summary.image('stage1_dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
                                else:
                                    for i in range(self.config.embedding_dim//3):
                                        vis_embedding = outs[2][:,:,:,3*i:3*(i+1)]
                                        tf.summary.image('stage2_embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
                                        if not self.config.embedding_include_bg:
                                            vis_embedding = vis_embedding * tf.cast(ds_item['object'] > 0, vis_embedding.dtype)
                                            tf.summary.image('stage2_embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
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
        self.net_stage1.trainable = True
        self.outlayer_semantic.trainable = True
        self.outlayer_dist.trainable = True
        self.feature_suppression.trainable = True
        self.net_stage2.trainable = True
        self.outlayer_embedding.trainable = True
        
        # train
        for _ in range(epochs-self.training_epoch):
            for ds_item in train_ds:
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'])
                   
                    loss_semantic = self.loss_fn_semantic(ds_item['semantic'], outs[0])
                    loss_dist = self.loss_fn_dist(ds_item['dist'], outs[1])
                    loss_embedding = self.loss_fn_embedding(ds_item['object'], outs[2], ds_item['adj_matrix'])
                    loss = loss_semantic * self.config.weight_semantic + loss_dist * self.config.weight_dist + loss_embedding * self.config.weight_embedding
                    
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # display trainig loss
                    self.training_step += 1
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}, semantic loss: {3:.5f}, dist loss: {4:.5f}, embedding loss: {5:.5f}".format(self.training_epoch+1, self.training_step, float(loss), float(loss_semantic), float(loss_dist), float(loss_embedding))
                    print(disp)
                    # summary training loss
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.training_step)
                        tf.summary.scalar('loss_semantic', loss_semantic, step=self.training_step)
                        tf.summary.scalar('loss_dist', loss_dist, step=self.training_step)
                        tf.summary.scalar('loss_embedding', loss_embedding, step=self.training_step)
                    # summary output
                    if self.training_step % 200 == 0 and image_summary:
                        with self.train_summary_writer.as_default():
                            tf.summary.image('input_img', tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                            # semantic
                            vis_semantic = tf.expand_dims(tf.argmax(outs[0], axis=-1), axis=-1)
                            tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                            tf.summary.image('semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
                            # dist
                            vis_dist = tf.cast(outs[1]*255/tf.reduce_max(outs[1]), tf.uint8)
                            tf.summary.image('dist', vis_dist, step=self.training_step, max_outputs=1)
                            vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
                            tf.summary.image('dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
                            # embedding
                            for i in range(self.config.embedding_dim//3):
                                vis_embedding = outs[2][:,:,:,3*i:3*(i+1)]
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
        
        pred_raw = {m: p for m, p in zip(['semantic', 'dist', 'embedding'], self.model.predict(image))}
        # semantic
        pred_raw['semantic'] = np.squeeze(np.argmax(pred_raw['semantic'], axis=-1))
        # dist
        pred_raw['dist'] = np.squeeze(pred_raw['dist'])
        # embedding
        embedding = np.squeeze(pred_raw['embedding'])
        pred_raw['embedding'] = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
        return pred_raw
