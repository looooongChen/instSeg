import tensorflow as tf
from tensorflow import keras

class PosConv(tf.keras.Model):

    def __init__(self, filters, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=1, activation='linear')
        # self.scale =tf.Variable(initial_value=tf.random.normal([1,], mean=1, stddev=2.0, dtype=tf.dtypes.float64), trainable=True)
    
    def call(self, inputs):
        sz = tf.shape(inputs)
        coordX, coordY = tf.meshgrid(tf.range(0,sz[2]), tf.range(0,sz[1]))
        coordX, coordY = coordX / (sz[2]-1) - 0.5, coordY / (sz[1]-1) - 0.5
        coords = tf.repeat(tf.expand_dims(tf.stack([coordX, coordY], axis=-1), axis=0), sz[0], axis=0) 

        # coords = coords * 2 * 3.1415926 * self.scale
        # coords = tf.math.sin(coords)

        coords = tf.cast(coords, inputs.dtype)
        coords = tf.stop_gradient(coords)

        outputs = tf.concat([inputs, coords], axis=-1)
        outputs = self.conv(outputs)
        return outputs

# class ConvPos(tf.keras.Model):

#     def __init__(self, filters, channels=1, kernel_size=1, activation='linear', padding='same', name=None, **kwargs):
#         super().__init__(name=name, **kwargs)
#         # self.scale =tf.Variable(initial_value=tf.random.normal((1,1,1,channels*2), mean=1, stddev=2.0, dtype=tf.dtypes.float32), trainable=True)
#         self.scale =tf.Variable(initial_value=tf.random.normal([1,], mean=1, stddev=2.0, dtype=tf.dtypes.float64), trainable=True)
#         self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)


#     def call(self, inputs):

#         sz = tf.shape(inputs)
#         sz_f = tf.cast(sz, tf.dtypes.float64)
#         coordX, coordY = tf.meshgrid(tf.range(0,sz[2]), tf.range(0,sz[1]))
#         coordX, coordY = tf.cast(coordX, dtype=tf.dtypes.float64), tf.cast(coordY, dtype=tf.dtypes.float64)
#         coordX, coordY = self.scale * 2.0 * 3.1415926 * (coordX / sz_f[2] - 0.5), self.scale * 2.0 * 3.1415926 * (coordY / sz_f[1] - 0.5)
#         coords = tf.expand_dims(tf.stack([coordX, coordY], axis=-1), axis=0)
#         coords = tf.repeat(coords, sz[0], axis=0) 
#         coords = tf.math.sin(coords)
#         coords = tf.cast(coords, inputs.dtype)
        
#         outputs = tf.concat([inputs, coords], axis=-1)
#         outputs = self.conv(outputs)
#         return outputs