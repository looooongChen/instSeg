import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose, Add, Cropping2D, SpatialDropout2D
from tensorflow.keras.regularizers import l2
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

class UNet(tf.keras.Model):

    def __init__(self,
                 nfilters=64,
                 nstage=5,
                 stage_conv=2,
                 padding='same', # 'same', 'reflect'
                 residual=False,
                 dropout_rate=0.2,
                 dropout_type='default',  # 'spatial', 'default'
                 batch_norm=True,
                 up_type='upConv', # 'upConv', 'deConv'
                 merge_type='cat', # 'add', 'cat'
                 kernel_initializer='he_normal',
                 use_bias=False,
                 weight_decay=1e-4,
                 name='UNet',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        assert up_type in ['upConv', 'deConv']
        assert merge_type in ['add', 'cat']
        assert dropout_type in ['spatial', 'default']

        self.nstage = nstage - 1
        self.stage_conv = stage_conv
        self.padding = padding
        padding_ = 'same' if padding == 'same' else 'valid'
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.batch_norm = True if self.residual else batch_norm
        self.up_type = up_type
        self.merge_type = merge_type

        self.filters = [nfilters*2**i for i in range(self.nstage)] + [nfilters*2**(self.nstage-i) for i in range(self.nstage+1)]
        self.layers_c = {}

        self.init_conv = Conv2D(self.filters[0], 3, padding=padding_, kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))

        # convolutional, dropout (optional), batch normatlization (optional) and residual add (optional) layers
        for block_idx in range(2*self.nstage+1):
            if residual:
                self.layers_c['residualAdd{:d}'.format(block_idx)] = Add()
            for conv_idx in range(self.stage_conv):
                # batch normalization
                if self.batch_norm:
                    self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)] = BatchNormalization()
                # relu activation
                self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)] = ReLU()
                # dropout layer
                if self.dropout_rate:
                    if dropout_type == 'spatial':
                        self.layers_c['dropout{:d}_{:d}'.format(block_idx, conv_idx)] = SpatialDropout2D(self.dropout_rate)
                    else:
                        self.layers_c['dropout{:d}_{:d}'.format(block_idx, conv_idx)] = Dropout(self.dropout_rate)
                # conv layers
                self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)] = Conv2D(self.filters[block_idx], 3, padding=padding_, kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
        
        # pooling layers of encoder path
        for block_idx in range(self.nstage):
            if residual:
                self.layers_c['residualConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 1, padding='valid', kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
            self.layers_c['pool{:d}'.format(block_idx)] = MaxPooling2D(pool_size=(2, 2))
        
        # upsampling layers of decoder path
        for block_idx in range(self.nstage+1, 2*self.nstage+1):
            ## upsampling - batch normalization
            if self.batch_norm:
                self.layers_c['upBatchnorm{:d}'.format(block_idx)] = BatchNormalization()
            ## upsampling - relu activation
            self.layers_c['upRelu{:d}'.format(block_idx)] = ReLU()
            ## upsampling - upsampling
            if self.up_type == 'upConv':
                self.layers_c['up{:d}'.format(block_idx)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
                self.layers_c['upConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 3, padding=padding_, kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
            else:
                self.layers_c['upConv{:d}'.format(block_idx)] = Conv2DTranspose(self.filters[block_idx], 4, 2, padding='same', kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
            # merge
            if self.merge_type == 'cat':
                self.layers_c['merge{:d}'.format(block_idx)] = Concatenate(axis=-1)
                if residual:
                    self.layers_c['residualConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 1, padding='valid', kernel_initializer=kernel_initializer, use_bias=use_bias, kernel_regularizer=l2(weight_decay))
            else:
                self.layers_c['merge{:d}'.format(block_idx)] = Add()


    def call(self, inputs, training=False):

        self.tensors = {}

        if self.padding == 'reflect':
            inputs = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
        outputs = self.init_conv(inputs)
        ##### encouding path ####
        for block_idx in range(self.nstage):
            if self.residual:
                inputs = self.layers_c['residualConv{:d}'.format(block_idx)](outputs)
            for conv_idx in range(self.stage_conv):
                # batch normalization
                if self.batch_norm:
                    outputs = self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)](outputs, training)
                # relu activation
                outputs = self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                # dropout
                if self.dropout_rate:
                    outputs = self.layers_c['dropout{:d}_{:d}'.format(block_idx, conv_idx)](outputs, training)
                # conv layers
                if self.padding == 'reflect':
                    outputs = tf.pad(outputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
                outputs = self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
            # residual addition
            if self.residual:
                outputs = self.layers_c['residualAdd{:d}'.format(block_idx)]([outputs, inputs])
            # pooling
            self.tensors['block{:d}'.format(block_idx)] = outputs
            outputs = self.layers_c['pool{:d}'.format(block_idx)](outputs)

        #### bottom block ####
        for conv_idx in range(self.stage_conv):
            # batch normalization
            if self.batch_norm:
                outputs = self.layers_c['batchnorm{:d}_{:d}'.format(self.nstage, conv_idx)](outputs, training)
            # relu activation
            outputs = self.layers_c['relu{:d}_{:d}'.format(self.nstage, conv_idx)](outputs)
            # dropout
            if self.dropout_rate :
                outputs = self.layers_c['dropout{:d}_{:d}'.format(self.nstage, conv_idx)](outputs, training)
            # conv layers
            if self.padding == 'reflect':
                outputs = tf.pad(outputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
            outputs = self.layers_c['conv{:d}_{:d}'.format(self.nstage, conv_idx)](outputs)

        # decoding path
        for block_idx in range(self.nstage+1, 2*self.nstage+1):
            # upsampling
            if self.up_type == 'upConv':
                outputs = self.layers_c['up{:d}'.format(block_idx)](outputs)
                if self.padding == 'reflect':
                    outputs = tf.pad(outputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
            if self.batch_norm:
                outputs = self.layers_c['upBatchnorm{:d}'.format(block_idx)](outputs, training)
            outputs = self.layers_c['upRelu{:d}'.format(block_idx)](outputs)
            outputs = self.layers_c['upConv{:d}'.format(block_idx)](outputs)
            # merge
            tensor_merge = self.tensors['block{:d}'.format(2*self.nstage-block_idx)]
            outputs = self.layers_c['merge{:d}'.format(block_idx)]([outputs, tensor_merge])
            # conv
            if self.residual:
                if self.merge_type == 'cat':
                    inputs = self.layers_c['residualConv{:d}'.format(block_idx)](outputs)
                else:
                    inputs = outputs
            for conv_idx in range(self.stage_conv):
                # batch normalization
                if self.batch_norm:
                    outputs = self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)](outputs, training)
                # relu activation
                outputs = self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                # dropout
                if self.dropout_rate:
                    outputs = self.layers_c['dropout{:d}_{:d}'.format(block_idx, conv_idx)](outputs, training)
                # conv layers
                if self.padding == 'reflect':
                    outputs = tf.pad(outputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
                outputs = self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
            if self.residual:
                outputs = self.layers_c['residualAdd{:d}'.format(block_idx)]([outputs, inputs])
        
        return outputs


if __name__ == "__main__":
    import numpy as np
    from tensorflow import keras
    import os

    ############################

    # is_valid, input_sz, output_sz = valid_shape(input_sz=(532,532), D=4, padding='valid')
    # print(is_valid, input_sz, output_sz)

    ############################

    model = UNet(nfilters=64,
                 nstage=4,
                 stage_conv=2,
                 residual=True,
                 dropout_rate=0.2,
                 dropout_type='default',  # 'spatial', 'default'
                 batch_norm=True,
                 up_type='deConv', # 'upConv', 'deConv'
                 merge_type='cat', # 'add', 'cat'
                 kernel_initializer='he_normal',
                 use_bias=False,
                 weight_decay=1e-4,
                 name='U-Net')

    # keras.utils.plot_model(model, "model.png", show_shapes=True)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    model.build(input_shape=(1,512,512,1))
    # model.summary()

    logdir="./logs_check"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # train_images = np.zeros((4,572,572,1)).astype(np.float32)
    # train_labels = np.zeros((4,388,388,1)).astype(np.int32)
    train_images = np.zeros((4,512,512,1)).astype(np.float32)
    train_labels = np.zeros((4,512,512,1)).astype(np.int32)

    # Train the model.
    model.fit(train_images, train_labels, batch_size=1, epochs=1, 
              callbacks=[tensorboard_callback])

