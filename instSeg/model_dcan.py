# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import InstSegBase 
from instSeg.uNet import *
from instSeg.uNetDoubleHead import *
from instSeg.utils import *
import instSeg.loss as L
from instSeg.post_process import *
import os
from skimage.measure import label as relabel
from instSeg.evaluation import Evaluator

import cv2

try:
    import tfAugmentor as tfaug 
    augemntor_available = True
except:
    augemntor_available = False


class InstSegDCAN(InstSegBase):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)
        self.best_score = float('inf') if self.config.save_best_metric == 'loss' else 0

    def _build(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        output_list = []

        backbone = UNnetDoubleHead if self.config.backbone == 'uNetDoubleHead' else UNnet
        self.net = backbone(filters=self.config.filters,
                            dropout_rate=self.config.dropout_rate,
                            batch_norm=self.config.batch_norm,
                            name='net')
                    
        self.outlayer_semantic = keras.layers.Conv2D(filters=self.config.classes, 
                                                     kernel_size=3, padding='same', activation='softmax', 
                                                     kernel_initializer='he_normal', 
                                                     name='out_semantic')
        self.outlayer_contour = keras.layers.Conv2D(filters=1, 
                                                    kernel_size=3, padding='same', activation='sigmoid', 
                                                    kernel_initializer='he_normal', 
                                                    name='out_contour')

        if self.config.backbone == 'uNetDoubleHead':
            features1, features2 = self.net(self.normalized_img)
            out_semantic = self.outlayer_semantic(features1)
            out_contour = self.outlayer_contour(features2)
        else:
            features = self.net(self.normalized_img)
            out_semantic = self.outlayer_semantic(features)
            out_contour = self.outlayer_contour(features)
            
        

        self.model = keras.Model(inputs=self.input_img, outputs=[out_semantic, out_contour])
        
        if self.config.verbose:
            self.model.summary()
            tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=False, show_layer_names=True)
    
    def evaluate(self, ds):
        '''
        Args:
            ds: validation dataset
        Return:
            improved: bool
        '''
        e = Evaluator(dimension=2, mode='area')
        loss = []
        for ds_item in ds:
            raw = self.predict_raw(ds_item['image'])
            instances, _, _ = self.postprocess(raw)
            e.add_example(instances, np.squeeze(ds_item['instance']))
            loss_semantic = self.loss_fn_semantic(ds_item['semantic'], raw[0])
            loss_contour = self.loss_fn_contour(ds_item['contour'], raw[1])
            loss.append(loss_semantic * self.config.weight_semantic + loss_contour * self.config.weight_contour)
        mAP = e.mAP()
        mAJ = e.mAJ()
        loss = np.mean(loss)
        return mAP, mAJ, loss


    def _validate(self, val_ds):
        '''
        Args:
            val_ds: validation dataset
        Return:
            improved: bool
        '''
        print('Running validation: ')
        mAP, mAJ, loss = self.evaluate(val_ds)
        # summary training loss
        with self.val_summary_writer.as_default():
            tf.summary.scalar('validation mAP', mAP, step=self.training_step)
            tf.summary.scalar('validation mAJ', mAJ, step=self.training_step)
            tf.summary.scalar('validation loss', loss, step=self.training_step)
        # best score
        disp = 'validation loss: {:.5f} mAP: {:.5f}, mAJ: {:.5f}'.format(loss, mAP, mAJ)
        if self.config.save_best_metric == 'loss':
            score = loss
        elif self.config.save_best_metric == 'mAP':
            score = mAP
        else: # use mAJ
            score= mAJ

        if score > self.best_score:
            self.best_score = score
            print("Validation Score Improved: " + disp)
            return True
        else:
            print("Validation Score Not Improved: " + disp)
            return False

    def train(self, train_data, validation_data=None, epochs=None, batch_size=None,
              augmentation=True, image_summary=True):
        
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'instance': ..., 'semantic': ...} 
                image (required): numpy array of size N x H x W x C 
                instance (requeired): numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        TODO: tfrecords support
        '''
        # prepare network
        if not self.training_prepared:
            self.prepare_training(semantic=True, contour=True)
        if epochs is None:
            epochs = self.config.train_epochs
        if batch_size is None:
            batch_size = self.config.train_batch_size

        # prepare data
        train_ds = self.ds_from_np(train_data, semantic=True, contour=True)
        if augmentation:
            train_ds = self.ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
        if validation_data is None or len(validation_data['image']) == 0:
            val_ds = None
        else:
            val_ds = self.ds_from_np(validation_data, instance=True, semantic=True, contour=True).batch(1)
            
        # load model
        self.load_weights()

        # train
        for _ in range(epochs-self.training_epoch):
            for ds_item in train_ds:
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'])
                   
                    loss_semantic = self.loss_fn_semantic(ds_item['semantic'], outs[0])
                    loss_contour = self.loss_fn_contour(ds_item['contour'], outs[1])
                    loss = loss_semantic * self.config.weight_semantic + loss_contour * self.config.weight_contour
                    
                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    # display trainig loss
                    self.training_step += 1
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}, semantic loss: {3:.5f}, contour loss: {4:.5f}".format(self.training_epoch+1, self.training_step, float(loss), float(loss_semantic), float(loss_contour))
                    print(disp)
                    # summary training loss
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.training_step)
                        tf.summary.scalar('loss_semantic', loss_semantic, step=self.training_step)
                        tf.summary.scalar('loss_contour', loss_contour, step=self.training_step)
                    # summary output
                    if self.training_step % 200 == 0 and image_summary:
                        with self.train_summary_writer.as_default():
                            tf.summary.image('input_img', tf.cast(ds_item['image'], tf.uint8), step=self.training_step, max_outputs=1)
                            # semantic
                            vis_semantic = tf.expand_dims(tf.argmax(outs[0], axis=-1), axis=-1)
                            tf.summary.image('semantic', vis_semantic*255/tf.reduce_max(vis_semantic), step=self.training_step, max_outputs=1)
                            tf.summary.image('semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
                            # contour
                            vis_contour = tf.cast(outs[1]*255, tf.uint8)
                            tf.summary.image('contour', vis_contour, step=self.training_step, max_outputs=1)
                            vis_contour_gt = tf.cast(ds_item['contour'], tf.uint8) * 255
                            tf.summary.image('contour_gt', vis_contour_gt, step=self.training_step, max_outputs=1)

                    
            self.training_epoch += 1

            self.save_weights()
            if validation_data:
                improved = self._validate(val_ds)
                if improved:
                    self.save_weights(save_best=True)

    def postprocess(self, raw):

        contour_thres=0.5

        semantic = np.argmax(raw[0], axis=-1).astype(np.uint16)
        contour = cv2.dilate((raw[1] > contour_thres).astype(np.uint8), disk_np(1, np.uint8), iterations = 1)

        instances = relabel(semantic * (contour == 0)).astype(np.uint16)
        fg = (semantic > 0).astype(np.uint16)
        while True:
            pixel_add = cv2.dilate(instances, disk_np(1, np.uint8), iterations = 1) * (instances == 0) * fg
            if np.sum(pixel_add) != 0:
                instances += pixel_add
            else:
                break

        return instances, semantic, contour
    
    def predict(self, image, keep_size=True):

        sz = image.shape
        raw = self.predict_raw(image)

        instance, semantic, contour = self.postprocess(raw)
        
        if keep_size:
            instances = cv2.resize(instances, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)
            semantic = cv2.resize(semantic, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)
            contour = cv2.resize(contour, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        return instances, semantic, contour
    