# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.model_base import InstSegMul 
from instSeg.uNet import *
from instSeg.uNet2H import *
from instSeg.utils import *
import os

class InstSegCascade(InstSegMul):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)

    def build_model(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        if self.config.backbone == 'uNet2H':
            backbone_arch = UNet2H 
        elif self.config.backbone == 'uNetSA':
            backbone_arch = UNetSA
        elif self.config.backbone == 'uNetD':
            backbone_arch = UNetD
        elif self.config.backbone == 'uNet': 
            backbone_arch = UNet
        else:
            assert False, 'Architecture "' + self.config.backbone + '" not valid!'

        output_list = []
        for i, m in enumerate(self.config.modules):
            if i != 0:   
                feature_suppression = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', 
                                                             activation='linear', kernel_initializer='he_normal', 
                                                             name='feature_'+m)
                if self.config.stop_gradient:
                    features = tf.stop_gradient(tf.identity(features))
                features = feature_suppression(features)
                input_list = [self.normalized_img, tf.nn.l2_normalize(features, axis=-1)]
            else:
                input_list = [self.normalized_img]

            backbone = backbone_arch(filters=self.config.filters,
                                     dropout_rate=self.config.dropout_rate,
                                     batch_norm=self.config.batch_norm,
                                     name='net_'+m)
            features = backbone(K.concatenate(input_list, axis=-1))

            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, 
                                               kernel_size=3, padding='same', activation='softmax', 
                                               kernel_initializer='he_normal',
                                               name='out_semantic')
                output_list.append(outlayer(features))
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation='sigmoid', 
                                               kernel_initializer='he_normal', 
                                               name='out_contour')
                output_list.append(outlayer(features))
            if m == 'dist':
                activation = 'sigmoid' if self.config.loss_dist == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation=activation, 
                                               kernel_initializer='he_normal',
                                               name='out_dist')
                output_list.append(outlayer(features))
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                               kernel_size=3, padding='same', activation='linear', 
                                               kernel_initializer='he_normal', 
                                               name='out_embedding')
                output_list.append(outlayer(features))      

        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
    

    # def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
    #           augmentation=True, image_summary=True):
        
    #     '''
    #     Inputs: 
    #         train_data/validation_data: a dict of numpy array {'image': ..., 'instance': ..., 'semantic': ...} 
    #             image (required): numpy array of size N x H x W x C 
    #             instance (requeired): numpy array of size N x H x W x 1, 0 indicated background
    #             semantic: numpy array of size N x H x W x 1
    #     '''
    #     modules_dict = {m: True for m in self.config.modules}
    #     # prepare network
    #     if not self.training_prepared:
    #         self.prepare_training(**modules_dict)
    #     epochs = self.config.train_epochs if epochs is None else epochs
    #     batch_size = self.config.train_batch_size if batch_size is None else batch_size

    #     # prepare data
    #     train_ds = self.ds_from_np(train_data, **modules_dict)
    #     if augmentation:
    #         train_ds = self.ds_augment(train_ds)
    #     train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
    #     if validation_data is None or len(validation_data['image']) == 0:
    #         val_ds = None
    #     else:
    #         modules_dict['instance'] = True
    #         val_ds = self.ds_from_np(validation_data, **modules_dict).batch(1)
            
    #     # load model
    #     self.load_weights()

    #     # train
    #     for _ in range(epochs-self.training_epoch):
    #         for ds_item in train_ds:
    #             with tf.GradientTape() as tape:
    #                 outs = self.model(ds_item['image'])
    #                 if len(self.module_config) == 1:
    #                     outs = [outs]

    #                 losses, loss = {}, 0
    #                 for m, out in zip(self.config.modules, outs):
    #                     if k == 'semantic':
    #                         losses['semantic'] = self.loss_fns['semantic'](ds_item['semantic'], out)
    #                         loss += losses['semantic'] * self.config.weight_semantic
    #                     elif k == 'contour':
    #                         losses['contour'] = self.loss_fns['contour'](ds_item['contour'], out)
    #                         loss += losses['contour'] * self.config.weight_dist
    #                     elif k == 'dist':
    #                         losses['dist'] = self.loss_fns['dist'](ds_item['dist'], out)
    #                         loss += losses['dist'] * self.config.weight_dist
    #                     elif k == 'embedding':
    #                         losses['embedding'] = self.loss_fn['embedding'](ds_item['instance'], v, ds_item['adj_matrix'])
    #                         loss += losses['embedding'] * self.config.weight_embedding
                    
    #                 grads = tape.gradient(loss, self.model.trainable_weights)
    #                 self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    #                 # display trainig loss
    #                 self.training_step += 1
    #                 disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}".format(self.training_epoch+1, self.training_step, float(loss))
    #                 for m, l in losses.items():
    #                     disp += ', ' + m + ' loss: {:.5f}'.format(float(l))
    #                 print(disp)
    #                 # summary training loss
    #                 with self.train_summary_writer.as_default():
    #                     tf.summary.scalar('loss', loss, step=self.training_step)
    #                     for m, l in losses.items():
    #                         tf.summary.scalar('loss_'+m, l, step=self.training_step)
    #                 # summary output
    #                 if self.training_step % 200 == 0 and image_summary:
    #                     with self.train_summary_writer.as_default():
    #                         tf.summary.image('input_img', tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
    #                         outs_dict = {k: v for k, v in zip(self.module_config, outs)}
    #                         # semantic
    #                         if 'semantic' in outs_dict.keys():
    #                             vis_semantic = tf.expand_dims(tf.argmax(outs_dict['semantic'], axis=-1), axis=-1)
    #                             tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
    #                             tf.summary.image('semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
    #                         # contour
    #                         if 'contour' in outs_dict.keys():
    #                             vis_contour = tf.cast(outs_dict['contour']*255, tf.uint8)
    #                             tf.summary.image('contour', vis_contour, step=self.training_step, max_outputs=1)
    #                             vis_contour_gt = tf.cast(ds_item['contour'], tf.uint8) * 255
    #                             tf.summary.image('contour_gt', vis_contour_gt, step=self.training_step, max_outputs=1)
    #                         # dist regression
    #                         if 'dist' in outs_dict.keys():
    #                             vis_dist = tf.cast(outs_dict['dist']*255/tf.reduce_max(outs_dict['dist']), tf.uint8)
    #                             tf.summary.image('dist', vis_dist, step=self.training_step, max_outputs=1)
    #                             vis_dist_gt = tf.cast(ds_item['dist']*255/tf.reduce_max(ds_item['dist']), tf.uint8)
    #                             tf.summary.image('dist_gt', vis_dist_gt, step=self.training_step, max_outputs=1)
    #                         # embedding
    #                         if 'embedding' in outs_dict.keys():
    #                             for i in range(self.config.embedding_dim//3):
    #                                 vis_embedding = outs_dict['embedding'][:,:,:,3*i:3*(i+1)]
    #                                 tf.summary.image('embedding_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
    #                                 if not self.config.embedding_include_bg:
    #                                     vis_embedding = vis_embedding * tf.cast(ds_item['object'] > 0, vis_embedding.dtype)
    #                                     tf.summary.image('embedding_masked_{}-{}'.format(3*i+1, 3*i+3), vis_embedding, step=self.training_step, max_outputs=1)
 
    #         self.training_epoch += 1

    #         self.save_weights(stage_wise=False)
    #         self.validate(val_ds, save_best=True)

