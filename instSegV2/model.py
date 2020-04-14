# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSegV2.nets.uNet import *
from instSegV2.utils import *
import instSegV2.loss as L
import os

# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
MAX_OBJ = 300

class Config(object):

    def __init__(self, image_channel=3, net='uNet', 
                 semantic_module=True, classes=1,
                 dist_module=True,
                 embedding_module=True, embedding_dim=8):

        assert net in ['uNet', 'uNetD7']
        self.verbose = True

        # input size
        self.image_size = 512, 512
        self.image_channel = image_channel
        
        # backbone config
        self.net = net
        self.dropout_rate=0
        self.batch_normalization=False

        # module config, here you can change the order of cascaded modules
        # this list only determines the order, whether a certain module will be really included depends on 
        # self.semantic_module, self.dist_module, self.embedding
        self.module_order = ['semantic', 'dist', 'embedding']
        # config of semantic module
        self.semantic_module = semantic_module
        self.classes = classes
        self.filters_semantic = 16
        # config of seed module
        self.dist_module = dist_module
        self.filters_dist = 16
        # config of embedding module
        self.embedding_module = embedding_module
        self.embedding_dim = embedding_dim
        self.filters_embedding = 16
        self.max_obj = MAX_OBJ
        
        # config of the semantic loss
        self.loss_semantic = 'focal_loss' 
        self.weight_semantic = 1
        # config of the seed loss
        self.loss_dist = 'binary_crossentropy'
        self.weight_dist = 1
        # config of the embedding loss
        self.loss_embedding = 'cos'
        self.neighbor_distance = 15
        self.weight_embedding = 1


        # training config:
        self.train_epochs = 100
        self.train_batch_size = 4
        self.train_learning_rate = 0.0001
    
    def check_assert(self):
        assert self.net in ['uNet', 'uNetD7']

        assert len(self.module_order) == 3
        for m in self.module_order:
            assert m in ['semantic', 'dist', 'embedding']

        assert self.loss_semantic in ['crossentropy', 'focal_loss']
        assert self.loss_seed in ['binary_crossentropy']
        assert self.loss_embedding in ['cos']


class InstSeg(object):

    def __init__(self, config, base_dir='./', run_name=''):
        self.config = config
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, 'model_'+run_name)
        self.log_dir = os.path.join(base_dir, 'log_'+run_name)
        self._build()
        self.training_prepared = False

    def _build(self):
        self.input_img = keras.layers.Input((*self.config.image_size, self.config.image_channel), name='input_img')
        # self.adjcent_matrix = keras.layers.Input((self.config.max_obj, self.config.max_obj), name='adjacent_matrix', dtype=tf.bool)

        if self.config.net == 'uNetD7':
            backbone = UNnet_d7
        else:
            backbone = UNnet

        self.nets = {}
        self.out_layer = {}
        # create semantic module
        if self.config.semantic_module:
            self.nets['semantic'] = backbone(filters=self.config.filters_semantic,
                                             dropout_rate=self.config.dropout_rate,
                                             batch_normalization=self.config.batch_normalization,
                                             name='net_semantic')
            self.out_layer['semantic'] = keras.layers.Conv2D(filters=self.config.classes+1, 
                                                             kernel_size=1, activation='softmax', name='semantic')
        # create distance regression module
        if self.config.dist_module:
            self.nets['dist'] = backbone(filters=self.config.filters_dist,
                                         dropout_rate=self.config.dropout_rate,
                                         batch_normalization=self.config.batch_normalization,
                                         name='net_dist')
            self.out_layer['dist'] = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='dist')
        # create embedding module
        if self.config.embedding_module:
            self.nets['embedding'] = backbone(filters=self.config.filters_embedding,
                                              dropout_rate=self.config.dropout_rate,
                                              batch_normalization=self.config.batch_normalization,
                                              name='net_embedding')
            self.out_layer['embedding'] = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                                              kernel_size=1, activation='linear', name='embedding')
        
        self.module_config = []
        for m in self.config.module_order:
            if m in self.nets.keys():
                self.module_config.append(m)
        
        input_list = [self.input_img]
        output_list = []
        for m in self.module_config:
            m_out = self.out_layer[m](self.nets[m](K.concatenate(input_list, axis=-1)))
            input_list.append(m_out)
            output_list.append(m_out)
        
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
    
    def _prepare_training(self):
        self.optimizer = keras.optimizers.Adam(lr=self.config.train_learning_rate)
        # self.optimizer = tf.keras.optimizers.RMSprop(lr=self.config.train_learning_rate)

        self.loss_fn = {}
        # metrics = []
        for m in self.module_config:
            if m == 'semantic':
                loss_semantic = {'crossentropy': L.cross_entropy, 
                                 'focal_loss': lambda y_true, y_pred: L.focal_loss(y_true, y_pred, gamma=2.0)}
                self.loss_fn['semantic'] = loss_semantic[self.config.loss_semantic] 
                # metrics.append(['accuracy'])
            elif m == 'dist':
                loss_dist = {'binary_crossentropy': tf.keras.losses.BinaryCrossentropy()} 
                self.loss_fn['dist'] = loss_dist[self.config.loss_dist]
            elif m == 'embedding':
                 loss_embedding = {'cos': lambda y_true, y_pred, adj_indicator: L.cosine_embedding_loss(y_true, y_pred, adj_indicator, self.config.max_obj, include_background=not self.config.semantic_module)}
                 self.loss_fn['embedding'] = loss_embedding[self.config.loss_embedding]

        self.training_prepared = True
    
    def _training_ds_from_np(self, data):
        for k in data.keys():
            if k == 'image':
                data[k] = image_resize_np(data[k], self.config.image_size)
                data[k] = K.cast_to_floatx(image_normalization_np(data[k])) 
            else:
                data[k] = image_resize_np(data[k], self.config.image_size, method='nearest')
        
        required = ['image']
        for m in self.module_config:
            if m == 'semantic':
                required.append('semantic')
                if 'semantic' in data.keys():
                    data['semantic'] = data['semantic']
                else:
                    data['semantic'] = data['object']>0
            elif m == 'dist':
                required.append('dist')
                data['dist']  = edt_np(data['object'], normalize=True)
            elif m == 'embedding':
                required.append('object')
                required.append('adj_matrix')
                data['adj_matrix'] = adj_matrix_np(data['object'], self.config.neighbor_distance, self.config.max_obj)

        for k in list(data.keys()):
            if k not in required:
                del data[k]

        # return data
        return tf.data.Dataset.from_tensor_slices(data)

    def train(self, train_data, validation_data=None, 
              augmentation=None, epochs=None, batch_size=None):
        
        '''
        Inputs: 
            train_data/validation_data: a dict of numpy array {'image': ..., 'semantic': ..., 'object': ...} 
                                        or a tfrecords folder (use our script to save your data in tfrecords)
                image (required): numpy array of size N x H x W x C 
                object: numpy array of size N x H x W x 1, 0 indicated background
                semantic: numpy array of size N x H x W x 1
        '''
        # prepare network
        if not self.training_prepared:
            self._prepare_training()
        if epochs is None:
            epochs = self.config.train_epochs
        if batch_size is None:
            batch_size = self.config.train_batch_size    
        # prepare data
        if isinstance(train_data, str):
            pass
        else:
            train_ds = self._training_ds_from_np(train_data).shuffle(buffer_size=64).batch(batch_size)
            val_ds = None if validation_data is None else _training_ds_from_np(validation_data).batch(batch_size)
        
        # load model
        cp_file = tf.train.latest_checkpoint(self.model_dir)
        if cp_file is not None:
            self.model.load_weights(cp_file)
            parsed = os.path.basename(cp_file).split('_')
            finishedEpoch = int(parsed[1][5:])
            finishedStep = int(parsed[2][4:])
            print('Model restored from Step {:d}, Epoch {:d}'.format(finishedStep, finishedEpoch))
        else:
            finishedEpoch = 0
            finishedStep = 0

        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # train
        for _ in range(epochs-finishedEpoch):
            for ds_item in train_ds:
                with tf.GradientTape() as tape:
                    outs = self.model(ds_item['image'])
                    if len(self.module_config) == 1:
                        outs = [outs]

                    losses = {}
                    loss = 0
                    for k, v in zip(self.module_config, outs):
                        fn = self.loss_fn[k]
                        if k == 'semantic':
                            losses['semantic'] = fn(ds_item['semantic'], v)
                            loss += losses['semantic'] * self.config.weight_semantic
                        elif k == 'dist':
                            losses['dist'] = fn(ds_item['dist'], v)
                            # print(ds_item['dist'].shape, v.shape)
                            loss += losses['dist'] * self.config.weight_dist
                        elif k == 'embedding':
                            losses['embedding'] = fn(ds_item['object'], v, ds_item['adj_matrix'])
                            loss += losses['embedding'] * self.config.weight_embedding

                    grads = tape.gradient(loss, self.model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                    
                    disp = "Epoch {0:d}, Step {1:d} with loss: {2:.5f}".format(finishedEpoch+1, finishedStep, float(loss))
                    for k, v in losses.items():
                        disp += ', ' + k + ' loss: {:.5f}'.format(float(v))
                    print(disp)
                    finishedStep += 1
                    
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=finishedStep)
                        for k, v in losses.items():
                            tf.summary.scalar(k, v, step=finishedStep)

            finishedEpoch += 1
            
            self.model.save_weights(os.path.join(self.model_dir, 'weights_epoch'+str(finishedEpoch)+'_step'+str(finishedStep)))
            print('Model saved at Step {:d}, Epoch {:d}'.format(finishedStep, finishedEpoch))


    def predict(self, images):
        cp_file = tf.train.latest_checkpoint(self.model_dir)
        if cp_file is not None:
            self.model.load_weights(cp_file)
        images = image_resize_np(images, self.config.image_size)
        images = K.cast_to_floatx(image_normalization_np(images)) 

        ds = tf.data.Dataset.from_tensor_slices(images).batch(1)
        if len(self.module_config) == 1:
            pred = {self.module_config[0]: self.model.predict(ds)}
        else:
            pred = {m: p for m, p in zip(self.module_config, self.model.predict(ds))}
        
        if 'embedding' in pred.keys():
            print("==================nomr")
            pred['embedding'] = tf.math.l2_normalize(pred['embedding'], axis=-1, name='emb_normalization')
        
        return pred
