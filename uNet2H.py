import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose, Add
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

INIT = 'he_normal' # 'glorot_uniform'
PAD = 'same' # 'valid'

class UNnet2H(tf.keras.Model):

  def __init__(self,
               D=4,
               filters=32,
               dropout_rate=0.2,
               batch_norm=False,
               upsample='conv', # 'interp', 'conv'
               merge='add', # 'add', 'cat'
               name='UNet2H',
               **kwargs):

    super().__init__(name=name, **kwargs)
    assert upsample in ['interp', 'conv']
    assert merge in ['add', 'cat']
    self.D = D
    self.dropout_rate = dropout_rate
    self.batch_norm = batch_norm
    self.upsample = upsample
    self.merge = merge

    self.filters = [filters*2**i for i in range(D)] + [filters*2**(D-i) for i in range(D+1)]
    self.L = {}

    for i in range(1, self.D+2):
        # dropout layer
        if self.dropout_rate < 1 and i != 1:
            self.L['dropout{:d}'.format(i)] = Dropout(self.dropout_rate)
        # conv layers
        self.L['conv{:d}_1'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
        self.L['conv{:d}_2'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
        # relu activation
        self.L['relu{:d}_1'.format(i)] = ReLU()
        self.L['relu{:d}_2'.format(i)] = ReLU()
        # batch normalization
        if self.batch_norm:
            self.L['batchnorm{:d}_1'.format(i)] = BatchNormalization()
            self.L['batchnorm{:d}_2'.format(i)] = BatchNormalization()
        # pooling
        if i != self.D + 1:
            self.L['pool{:d}'.format(i)] = MaxPooling2D(pool_size=(2, 2))
    
    for i in range(self.D+2, 2*self.D+2):
        for dec in range(1,3):
            # up sampling
            if self.upsample == 'interp':
                self.L['dec{:d}_up{:d}'.format(dec,i)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
                self.L['dec{:d}_conv{:d}_up'.format(dec,i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            else:
                self.L['dec{:d}_up{:d}'.format(dec,i)] = Conv2DTranspose(self.filters[i-1], 2, 2, kernel_initializer=INIT)
            self.L['dec{:d}_relu{:d}_up'.format(dec,i)] = ReLU()
            self.L['dec{:d}_batchnorm{:d}_up'.format(dec,i)] = BatchNormalization()
            # merge
            if self.merge == 'cat':
                self.L['dec{:d}_merge{:d}'.format(dec,i)] = Concatenate(axis=-1)
            else:
                self.L['dec{:d}_merge{:d}'.format(dec,i)] = Add()
            # dropout layer
            if self.dropout_rate < 1 and i != 1:
                self.L['dec{:d}_dropout{:d}'.format(dec,i)] = Dropout(self.dropout_rate)
            # conv layers
            self.L['dec{:d}_conv{:d}_1'.format(dec,i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            self.L['dec{:d}_conv{:d}_2'.format(dec,i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            # relu activation
            self.L['dec{:d}_relu{:d}_1'.format(dec,i)] = ReLU()
            self.L['dec{:d}_relu{:d}_2'.format(dec,i)] = ReLU()
            # batch normalization
            if self.batch_norm:
                self.L['dec{:d}_batchnorm{:d}_1'.format(dec,i)] = BatchNormalization()
                self.L['dec{:d}_batchnorm{:d}_2'.format(dec,i)] = BatchNormalization()

  def call(self, inputs):

    self.T = {}

    outputs = inputs
    for i in range(1, self.D+2):
        # dropout
        if self.dropout_rate < 1 and i != 1:
            outputs = self.L['dropout{:d}'.format(i)](outputs)
        # conv1
        outputs = self.L['conv{:d}_1'.format(i)](outputs)
        if self.batch_norm:
            outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
        outputs = self.L['relu{:d}_1'.format(i)](outputs)
        # conv2
        outputs = self.L['conv{:d}_2'.format(i)](outputs)
        if self.batch_norm:
            outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
        outputs = self.L['relu{:d}_2'.format(i)](outputs)
        # pooling
        if i != self.D+1:
            self.T['conv{:d}'.format(i)] = outputs
            outputs = self.L['pool{:d}'.format(i)](outputs)

    outputs_enc = outputs

    for dec in range(1,3):
        outputs = outputs_enc
        for i in range(self.D+2, 2*self.D+2):
            # upsampling
            outputs = self.L['dec{:d}_up{:d}'.format(dec,i)](outputs)
            if self.upsample == 'interp':
                outputs = self.L['dec{:d}_conv{:d}_up'.format(dec,i)](outputs)
            if self.batch_norm:
                outputs = self.L['dec{:d}_batchnorm{:d}_up'.format(dec,i)](outputs)
            outputs = self.L['dec{:d}_relu{:d}_up'.format(dec,i)](outputs)
            # merge
            outputs = self.L['dec{:d}_merge{:d}'.format(dec,i)]([outputs, self.T['conv{:d}'.format(2*self.D+2-i)]])
            # dropout
            if self.dropout_rate < 1:
                outputs = self.L['dec{:d}_dropout{:d}'.format(dec,i)](outputs)
            # conv1
            outputs = self.L['dec{:d}_conv{:d}_1'.format(dec,i)](outputs)
            if self.batch_norm:
                outputs = self.L['dec{:d}_batchnorm{:d}_1'.format(dec,i)](outputs)
            outputs = self.L['dec{:d}_relu{:d}_1'.format(dec,i)](outputs)
            # conv2
            outputs = self.L['dec{:d}_conv{:d}_2'.format(dec,i)](outputs)
            if self.batch_norm:
                outputs = self.L['dec{:d}_batchnorm{:d}_2'.format(dec,i)](outputs)
            outputs = self.L['dec{:d}_relu{:d}_2'.format(dec,i)](outputs)
        self.T['dec{:d}'.format(dec)] = outputs
    
    return self.T['dec1'], self.T['dec2'] 


if __name__ == "__main__":
    import numpy as np
    import os

    model = UNnet2H(D=3, filters=32, dropout_rate=0.5, batch_norm=True, merge='add', upsample='interp')
    model.build(input_shape=(1,512,512,1))
    model.summary()

    @tf.function
    def trace_func():
        inputs = np.zeros((4,512,512,1)).astype(np.float32)
        return model(inputs)

    # Set up logging.
    logdir = '.\\logs_check'
    writer = tf.summary.create_file_writer(logdir)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = trace_func()
    with writer.as_default():
        tf.summary.trace_export(name="network_check", step=0, profiler_outdir=logdir)
