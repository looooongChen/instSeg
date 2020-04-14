import tensorflow as tf
from denseBlock import *
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: denseVNet
Automatic Multi-Organ Segmentation on Abdominal CT With Dense V-Networks
'''

class DenseVNet(tf.keras.Model):

  def __init__(self,
               filters=4,
               dropout_rate=0.2,
               name='DenseVNet',
               **kwargs):

    super(DenseVNet, self).__init__(name=name, **kwargs)

    self.Conv = tf.keras.layers.Conv3D(filters, 3, strides=2, activation="linear", padding='same')

    self.Dense1 = DenseBlock3D(filters=filters, depth=4, dropout_rate=dropout_rate)
    self.Transition1 = TransitionLayer3D(filters=6*filters, dropout_rate=dropout_rate)
    self.BatchNorm1 = tf.keras.layers.BatchNormalization()
    self.Conv1 = tf.keras.layers.Conv3D(filters*3, 3, activation="linear", padding='same')
    self.Drop1 = tf.keras.layers.Dropout(dropout_rate)

    self.Dense2 = DenseBlock3D(filters=filters*2, depth=9, dropout_rate=dropout_rate)
    self.Transition2 = TransitionLayer3D(filters=6*filters, dropout_rate=dropout_rate)
    self.BatchNorm2 = tf.keras.layers.BatchNormalization()
    self.Conv2 = tf.keras.layers.Conv3D(filters*6, 3, activation="linear", padding='same')
    self.Drop2 = tf.keras.layers.Dropout(dropout_rate)
    self.Up2 = tf.keras.layers.UpSampling3D(size=2)

    self.Dense3 = DenseBlock3D(filters=filters*4, depth=9, dropout_rate=dropout_rate)
    self.BatchNorm3 = tf.keras.layers.BatchNormalization()
    self.Conv3 = tf.keras.layers.Conv3D(filters*6, 3, activation="linear", padding='same')
    self.Drop3 = tf.keras.layers.Dropout(dropout_rate)
    self.Up3 = tf.keras.layers.UpSampling3D(size=4)

    self.Merge = tf.keras.layers.Concatenate(axis=-1)


  def call(self, inputs):
    conv = self.Conv(inputs)

    dense1 = self.Dense1(conv)
    transition1 = self.Transition1(dense1)
    batchNorm1 = self.BatchNorm1(dense1)
    batchNorm1_relu = tf.keras.activations.relu(batchNorm1)
    conv1 = self.Conv1(batchNorm1_relu)
    drop1 = self.Drop1(conv1)

    dense2 = self.Dense2(transition1)
    transition2 = self.Transition2(dense2)
    batchNorm2 = self.BatchNorm2(dense2)
    batchNorm2_relu = tf.keras.activations.relu(batchNorm2)
    conv2 = self.Conv2(batchNorm2_relu)
    drop2 = self.Drop2(conv2)
    up2 = self.Up2(drop2)

    dense3 = self.Dense3(transition2)
    batchNorm3 = self.BatchNorm3(dense3)
    batchNorm3_relu = tf.keras.activations.relu(batchNorm3)
    conv3 = self.Conv3(batchNorm3_relu)
    drop3 = self.Drop3(conv3)
    up3 = self.Up3(drop3)

    merge = self.Merge([drop1, up2, up3])

    return merge


if __name__ == "__main__":
    import numpy as np
    import os

    model = DenseVNet()

    @tf.function
    def trace_func():
        inputs = np.zeros((4,512,512,64,1)).astype(np.float32)
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
