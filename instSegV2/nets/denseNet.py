import tensorflow as tf
from denseBlock import *
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: denseNet
Densely Connected Convolutional Networks
'''

class DenseNet(tf.keras.Model):

  def __init__(self,
               filters=24,
               dropout_rate=0.2,
               version='DenseNet-121',
               name='DenseNet',
               **kwargs):

    super(DenseNet, self).__init__(name=name, **kwargs)

    assert version in ['DenseNet-121', 'DenseNet-169', 'DenseNet-201', 'DenseNet-264']
    if version == 'DenseNet-121':
        depths = [6, 12, 24, 15]
    elif version == 'DenseNet-169':
        depths = [6, 12, 32, 32]
    elif version == 'DenseNet-201':
        depths = [6, 12, 48, 32]
    else:
        depths = [6, 12, 64, 48]

    self.Conv = tf.keras.layers.Conv2D(2*filters, 7, activation="linear", padding='same')
    self.Pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)

    self.Dense1 = DenseBlock(filters=filters, depth=depths[0], dropout_rate=dropout_rate)
    self.Transition1 = TransitionLayer(filters=filters, dropout_rate=dropout_rate)
    self.Dense2 = DenseBlock(filters=filters, depth=depths[1], dropout_rate=dropout_rate)
    self.Transition2 = TransitionLayer(filters=filters, dropout_rate=dropout_rate)
    self.Dense3 = DenseBlock(filters=filters, depth=depths[2], dropout_rate=dropout_rate)
    self.Transition3 = TransitionLayer(filters=filters, dropout_rate=dropout_rate)
    self.Dense4 = DenseBlock(filters=filters, depth=depths[3], dropout_rate=dropout_rate)
    self.Transition4 = TransitionLayer(filters=filters, dropout_rate=dropout_rate)

  def call(self, inputs):
    conv = self.Conv(inputs)
    pool = self.Pool(conv)

    dense1 = self.Dense1(pool)
    transition1 = self.Transition1(dense1)

    dense2 = self.Dense2(transition1)
    transition2 = self.Transition2(dense2)

    dense3 = self.Dense3(transition2)
    transition3 = self.Transition3(dense3)

    dense4 = self.Dense4(transition3)
    transition4 = self.Transition4(dense4)

    return transition4

if __name__ == "__main__":
    import numpy as np
    import os

    model = DenseNet()

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
