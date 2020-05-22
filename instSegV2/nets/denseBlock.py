import tensorflow as tf
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: denseNet, denseVNet
Densely Connected Convolutional Networks
Automatic Multi-Organ Segmentation on Abdominal CT With Dense V-Networks
'''

class BottleneckLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super(BottleneckLayer, self).__init__(**kwargs)
        self.BatchNorm1 = tf.keras.layers.BatchNormalization()
        self.Conv1 = tf.keras.layers.Conv2D(4*filters, 1, activation="linear", padding='same')
        self.Drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.BatchNorm2 = tf.keras.layers.BatchNormalization()
        self.Conv2 = tf.keras.layers.Conv2D(filters, 3, activation="linear", padding='same')
        self.Drop2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        batchNorm1 = self.BatchNorm1(inputs)
        batchNorm1_relu = tf.keras.activations.relu(batchNorm1)
        conv1 = self.Conv1(batchNorm1_relu)
        drop1 = self.Drop1(conv1)

        batchNorm2 = self.BatchNorm2(drop1)
        batchNorm2_relu = tf.keras.activations.relu(batchNorm2)
        conv2 = self.Conv2(batchNorm2_relu)
        drop2 = self.Drop2(conv2)

        return drop2

class BottleneckLayer3D(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super(BottleneckLayer3D, self).__init__(**kwargs)
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        self.Conv = tf.keras.layers.Conv3D(filters, 1, activation="linear", padding='same')
        self.Drop = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        batchNorm = self.BatchNorm(inputs)
        batchNorm_relu = tf.keras.activations.relu(batchNorm)
        conv = self.Conv(batchNorm_relu)
        drop = self.Drop(conv)

        return drop
    
class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super(TransitionLayer, self).__init__(**kwargs)
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        self.Conv = tf.keras.layers.Conv2D(filters, 1, activation="linear", padding='same')
        self.Drop = tf.keras.layers.Dropout(dropout_rate)
        self.AveragePool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')

    def call(self, inputs):
        batchNorm = self.BatchNorm(inputs)
        batchNorm_relu = tf.keras.activations.relu(batchNorm)
        conv = self.Conv(batchNorm_relu)
        drop = self.Drop(conv)
        averagePool = self.AveragePool(drop)
        return averagePool

class TransitionLayer3D(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2, **kwargs):
        super(TransitionLayer3D, self).__init__(**kwargs)
        self.BatchNorm = tf.keras.layers.BatchNormalization()
        self.Conv = tf.keras.layers.Conv3D(filters, 3, strides=2, activation="linear", padding='same')
        self.Drop = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        batchNorm = self.BatchNorm(inputs)
        batchNorm_relu = tf.keras.activations.relu(batchNorm)
        conv = self.Conv(batchNorm_relu)
        drop = self.Drop(conv)
        return drop

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, filters, depth, dropout_rate=0.2, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.layer_stack = []

        self.Bottleneck = BottleneckLayer(filters, dropout_rate=dropout_rate)
        for d in range(depth-1):
            self.layer_stack.append(tf.keras.layers.Concatenate(axis=-1))
            self.layer_stack.append(BottleneckLayer(filters, dropout_rate=dropout_rate))
        self.layer_stack.append(tf.keras.layers.Concatenate(axis=-1))
    
    def call(self, inputs):
        layers_concat = [inputs, self.Bottleneck(inputs)]
        for i in range(len(self.layer_stack)):
            if i % 2 == 0:
                outputs = self.layer_stack[i](layers_concat)
            else:
                layers_concat.append(self.layer_stack[i](outputs))
        return outputs


class DenseBlock3D(tf.keras.layers.Layer):
    def __init__(self, filters, depth, dropout_rate=0.2, **kwargs):
        super(DenseBlock3D, self).__init__(**kwargs)
        self.layer_stack = []

        self.Bottleneck = BottleneckLayer3D(filters, dropout_rate=dropout_rate)
        for d in range(depth-1):
            self.layer_stack.append(tf.keras.layers.Concatenate(axis=-1))
            self.layer_stack.append(BottleneckLayer3D(filters, dropout_rate=dropout_rate))
        self.layer_stack.append(tf.keras.layers.Concatenate(axis=-1))
    
    def call(self, inputs):
        layers_concat = [inputs, self.Bottleneck(inputs)]
        for i in range(len(self.layer_stack)):
            if i % 2 == 0:
                outputs = self.layer_stack[i](layers_concat)
            else:
                layers_concat.append(self.layer_stack[i](outputs))
        return outputs
