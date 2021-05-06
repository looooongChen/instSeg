from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import ModelBase 
from instSeg.uNet import *
from instSeg.utils import *
import instSeg.loss as L
from instSeg.post_process import *
import os
from skimage.measure import label as relabel
from skimage.measure import regionprops
from instSeg.evaluation import Evaluator
import cv2


class Seg(ModelBase):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)
        assert len(config.modules) == 2

    def build_model(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        self.net = UNnet(filters=self.config.filters,
                         dropout_rate=self.config.dropout_rate,
                         batch_norm=self.config.batch_norm,
                         upsample=self.config.net_upsample,
                         merge=self.config.net_merge,
                         name='net')
        features = self.net(self.normalized_img)

        outlayer = keras.layers.Conv2D(filters=self.config.classes, 
                                        kernel_size=3, padding='same', activation='softmax', 
                                        kernel_initializer='he_normal',
                                        name='out_semantic')
        outputs = outlayer(features[idx])

        self.model = keras.Model(inputs=self.input_img, outputs=outputs)
        
        if self.config.verbose:
            self.model.summary()

    def validate(self, val_ds, save_best=True):
        '''
        Args:
            val_ds: validation dataset
        '''
        if val_ds is not None:
            print('Running validation: ')
            e = Evaluator(dimension=2, mode='area')
            for ds_item in val_ds:
                outs = self.model(ds_item['image'])
                instances = self.postprocess(outs)
                if isinstance(instances, tuple):
                    instances = instances[0]
                e.add_example(instances, np.squeeze(ds_item['instance']))
            mAP, mAJ = e.mAP(), e.mAJ()
            # summary training loss
            with self.val_summary_writer.as_default():
                tf.summary.scalar('validation mAP', mAP, step=self.training_step)
                tf.summary.scalar('validation mAJ', mAJ, step=self.training_step)
            # best score
            disp = 'validation mAP: {:.5f}, mAJ: {:.5f}'.format(mAP, mAJ)
            if self.config.save_best_metric == 'mAP':
                score = mAP
            else: # use mAJ
                score= mAJ

            if score > self.best_score:
                self.best_score = score
                print("Validation Score Improved: " + disp)
                if save_best:
                    self.save_weights(save_best=True)
            else:
                print("Validation Score Not Improved: " + disp)

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
        train_ds = self.ds_from_np(train_data, **modules_dict)
        if augmentation:
            train_ds = self.ds_augment(train_ds)
        train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size)
        if validation_data is None or len(validation_data['image']) == 0:
            val_ds = None
        else:
            modules_dict['instance'] = True
            val_ds = self.ds_from_np(validation_data, **modules_dict).batch(1)
            
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
                        if m == 'semantic':
                            losses['semantic'] = self.loss_fns['semantic'](ds_item['semantic'], out)
                            loss += losses['semantic'] * self.config.weight_semantic
                        elif m == 'contour':
                            losses['contour'] = self.loss_fns['contour'](ds_item['contour'], out)
                            loss += losses['contour'] * self.config.weight_contour
                        elif m == 'dist':
                            losses['dist'] = self.loss_fns['dist'](ds_item['dist'], out)
                            loss += losses['dist'] * self.config.weight_dist
                        elif m == 'embedding':
                            losses['embedding'] = self.loss_fn['embedding'](ds_item['instance'], v, ds_item['adj_matrix'])
                            loss += losses['embedding'] * self.config.weight_embedding
                    
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
                                tf.summary.image('semantic_gt', tf.cast(ds_item['semantic']*255/tf.reduce_max(ds_item['semantic']), tf.uint8), step=self.training_step, max_outputs=1)
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
            self.validate(val_ds, save_best=True)

    def postprocess(self, raw, min_size=20):

        for m, r in zip(self.config.modules, [np.array(p) for p in raw]):
            if m == 'semantic':
                semantic = r
            if m == 'contour':
                contour = r
            if m == 'dist':
                dist = r
            if m == 'embedding':
                embedding =r

        if 'semantic' in self.config.modules and 'contour' in self.config.modules:
            contour_thres=0.5

            semantic = np.squeeze(np.argmax(semantic, axis=-1)).astype(np.uint16)
            contour = cv2.dilate((np.squeeze(contour) > contour_thres).astype(np.uint8), disk_np(1, np.uint8), iterations = 1)

            instances = relabel(semantic * (contour == 0)).astype(np.uint16)
            fg = (semantic > 0).astype(np.uint16)
            while True:
                pixel_add = cv2.dilate(instances, disk_np(1, np.uint8), iterations = 1) * (instances == 0) * fg
                if np.sum(pixel_add) != 0:
                    instances += pixel_add
                else:
                    break
            
            for r in regionprops(instances):
                if r.area < min_size:
                    instances[r.coords[:,0], r.coords[:,1]] = 0

            return instances
        if 'semantic' in self.config.modules and 'dist' in self.config.modules:
            pass
        if 'embedding' in self.config.modules and 'dist' in self.config.modules:
            pass
    
    def predict(self, image, keep_size=True):
        
        sz = image.shape
        # model inference
        img = np.squeeze(image)
        img = image_resize_np([img], (self.config.H, self.config.W))
        img = K.cast_to_floatx(img)
        raw = self.model(img)
        # post processing
        instances = self.postprocess(raw)
        # resize to original resolution
        if keep_size:
            instances = cv2.resize(instances, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)
            # semantic = cv2.resize(semantic, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)
            # contour = cv2.resize(contour, (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        return instances
    