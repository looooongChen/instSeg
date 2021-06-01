import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose, Add, Cropping2D
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

# def valid_shape(input_sz, D, padding='valid'):
#     if padding == 'same':
#         output_sz = [s/(2**D) for s in input_sz]
#         is_valid = (output_sz[0] == round(output_sz[0])) and (output_sz[1] == round(output_sz[1]))
#         input_sz = [int(round(s)*(2**D)) for s in output_sz]
#         output_sz = input_sz
#     if padding == 'valid':
#         output_sz = input_sz
#         is_valid = True
#         for _ in range(D):
#             output_sz = [(s-4)/2 for s in output_sz]
#             is_valid = is_valid and (output_sz[0] == round(output_sz[0])) and (output_sz[1] == round(output_sz[1]))
#         output_sz = [round(s) for s in output_sz]
#         input_sz = output_sz[:]
#         for _ in range(D):
#             input_sz = [s*2+4 for s in input_sz]
#         output_sz = [s-4 for s in output_sz]
#         for _ in range(D):
#             output_sz = [s*2-4 for s in output_sz]
#         input_sz = [int(s) for s in input_sz]
#         output_sz = [int(s) for s in output_sz]

#     return is_valid, tuple(input_sz), tuple(output_sz)

class UNet(tf.keras.Model):

    def __init__(self,
                 pooling_stage=4,
                 block_conv=2,
                 padding='same',
                 residual=False,
                 filters=64,
                 dropout_rate=0.5,
                 batch_norm=False,
                 upsample='deconv', # 'interp', 'deconv'
                 merge='add', # 'add', 'cat'
                 kernel_initializer='glorot_uniform',
                 name='UNet',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        assert upsample in ['interp', 'deconv']
        assert merge in ['add', 'cat']
        self.pooling_stage = pooling_stage
        self.block_conv = block_conv
        self.padding = padding
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.batch_norm = True if self.residual else batch_norm
        self.upsample = upsample
        self.merge = merge

        self.filters = [filters*2**i for i in range(self.pooling_stage)] + [filters*2**(self.pooling_stage-i) for i in range(self.pooling_stage+1)]
        self.layers_c = {}

        # convolutional, dropout (optional), batch normatlization (optional) and residual add (optional) layers
        for block_idx in range(2*self.pooling_stage+1):
            # dropout layer
            if self.dropout_rate < 1 and block_idx != 0:
                self.layers_c['dropout{:d}'.format(block_idx)] = Dropout(self.dropout_rate)
            for conv_idx in range(self.block_conv):
                if block_idx != 0 or conv_idx != 0:
                    # batch normalization
                    if self.batch_norm:
                        self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)] = BatchNormalization()
                    # relu activation
                    self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)] = ReLU()
                # conv layers
                self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)] = Conv2D(self.filters[block_idx], 3, padding=self.padding, kernel_initializer=kernel_initializer)
            if self.residual and block_idx != self.pooling_stage:
                self.layers_c['residualAdd{:d}'.format(block_idx)] = Add()
                self.layers_c['residualCrop{:d}'.format(block_idx)] = Cropping2D(cropping=self.block_conv)
        
        # pooling layers of encoder path
        for block_idx in range(self.pooling_stage):
            self.layers_c['residualConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 1, padding='same', kernel_initializer=kernel_initializer)
            self.layers_c['pool{:d}'.format(block_idx)] = MaxPooling2D(pool_size=(2, 2))
        
        margin = 2 * self.block_conv
        # upsampling layers of decoder path
        for block_idx in range(self.pooling_stage+1, 2*self.pooling_stage+1):
            if self.padding != 'same':
                self.layers_c['crop{:d}'.format(block_idx)] = Cropping2D(cropping=margin)
                margin = 2 * (margin + 2 * self.block_conv)
            # upsampling
            if self.upsample == 'interp':
                self.layers_c['upConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 1, padding='same', kernel_initializer=kernel_initializer)
                self.layers_c['up{:d}'.format(block_idx)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
            else:
                self.layers_c['up{:d}'.format(block_idx)] = Conv2DTranspose(self.filters[block_idx], 2, 2, kernel_initializer=kernel_initializer)
            # merge
            if self.merge == 'cat':
                self.layers_c['merge{:d}'.format(block_idx)] = Concatenate(axis=-1)
                self.layers_c['catConv{:d}'.format(block_idx)] = Conv2D(self.filters[block_idx], 1, padding='same', kernel_initializer=kernel_initializer)
            else:
                self.layers_c['merge{:d}'.format(block_idx)] = Add()

    # def receptive_field(self):
    #     RF = 0
    #     for i in range(self.D):
    #         RF = 3 + RF-1 if i == 0 else 5 + RF -1
    #         RF = 3 + RF-1


    def call(self, inputs):

        self.tensors = {}

        outputs = inputs
        ##### encouding path ####
        for block_idx in range(self.pooling_stage):
            # dropout
            if self.dropout_rate < 1 and block_idx != 0:
                outputs = self.layers_c['dropout{:d}'.format(block_idx)](outputs)
            if self.residual:
                inputs = self.layers_c['residualConv{:d}'.format(block_idx)](outputs)
            for conv_idx in range(self.block_conv):
                if block_idx != 0 or conv_idx != 0:
                    # batch normalization
                    if self.batch_norm:
                        outputs = self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                    # relu activation
                    outputs = self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                # conv layers
                outputs = self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
            # residual addition
            if self.residual:
                if self.padding != 'same':
                    inputs = self.layers_c['residualCrop{:d}'.format(block_idx)](inputs)
                outputs = self.layers_c['residualAdd{:d}'.format(block_idx)]([outputs, inputs])
            # pooling
            self.tensors['block{:d}'.format(block_idx)] = outputs
            outputs = self.layers_c['pool{:d}'.format(block_idx)](outputs)

        #### bottom block ####
        # dropout
        if self.dropout_rate < 1:
            outputs = self.layers_c['dropout{:d}'.format(self.pooling_stage)](outputs)
        for conv_idx in range(self.block_conv):
            # batch normalization
            if self.batch_norm:
                outputs = self.layers_c['batchnorm{:d}_{:d}'.format(self.pooling_stage, conv_idx)](outputs)
            # relu activation
            outputs = self.layers_c['relu{:d}_{:d}'.format(self.pooling_stage, conv_idx)](outputs)
            # conv layers
            outputs = self.layers_c['conv{:d}_{:d}'.format(self.pooling_stage, conv_idx)](outputs)

        # decoding path
        for block_idx in range(self.pooling_stage+1, 2*self.pooling_stage+1):
            # upsampling
            outputs = self.layers_c['up{:d}'.format(block_idx)](outputs)
            if self.upsample == 'interp':
                outputs = self.layers_c['upConv{:d}'.format(block_idx)](outputs)
            # merge
            if self.padding != 'same':
                tensor_merge = self.layers_c['crop{:d}'.format(block_idx)](self.tensors['block{:d}'.format(2*self.pooling_stage-block_idx)])
            else:
                tensor_merge = self.tensors['block{:d}'.format(2*self.pooling_stage-block_idx)]
            outputs = self.layers_c['merge{:d}'.format(block_idx)]([outputs, tensor_merge])
            if self.merge == 'cat':
                outputs = self.layers_c['catConv{:d}'.format(block_idx)](outputs)
            # dropout
            if self.dropout_rate < 1:
                outputs = self.layers_c['dropout{:d}'.format(block_idx)](outputs)
            if self.residual:
                inputs = outputs
            for conv_idx in range(self.block_conv):
                # batch normalization
                if self.batch_norm:
                    outputs = self.layers_c['batchnorm{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                # relu activation
                outputs = self.layers_c['relu{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
                # conv layers
                outputs = self.layers_c['conv{:d}_{:d}'.format(block_idx, conv_idx)](outputs)
            if self.residual:
                if self.padding != 'same':
                    inputs = self.layers_c['residualCrop{:d}'.format(block_idx)](inputs)
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

    # model = UNet(D=3, filters=32, dropout_rate=0.5, batch_norm=True, upsample='interp', merge='cat')
    model = UNet(pooling_stage=4,
                 block_conv=2,
                 padding='valid',
                 residual=True,
                 filters=64,
                 dropout_rate=0.5,
                 batch_norm=False,
                 upsample='interp', # 'interp', 'deconv'
                 merge='cat', # 'add', 'cat'
                 name='U-Net')
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    # model.build(input_shape=(1,512,512,1))
    # model.summary()

    logdir="./logs_check"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    train_images = np.zeros((4,572,572,1)).astype(np.float32)
    train_labels = np.zeros((4,388,388,1)).astype(np.int32)

    # Train the model.
    model.fit(train_images, train_labels, batch_size=1, epochs=1, 
              callbacks=[tensorboard_callback])

