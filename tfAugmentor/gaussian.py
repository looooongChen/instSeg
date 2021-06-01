import tensorflow as tf

def gaussian_kernel(sigma):
    size = round(sigma*2)
    g = tf.range(-size, size, dtype=tf.float32)
    g = tf.math.exp(-(tf.pow(g, 2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
    g_kernel = tf.tensordot(g, g, axes=0)
    return tf.stop_gradient(g_kernel / tf.reduce_sum(g_kernel))
     
def gaussian_blur(image, sigma):

    shape = image.get_shape()
    blurred = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image
    blurred = tf.cast(blurred, tf.float32)

    g_kernel = gaussian_kernel(sigma)
    g_kernel = tf.expand_dims(tf.expand_dims(g_kernel, -1), -1)
    g_kernel = tf.repeat(g_kernel, shape[-1], axis=-2)
    blurred = tf.nn.depthwise_conv2d(blurred, g_kernel, strides=[1, 1, 1, 1], padding="SAME")
    # blurred = tf.nn.conv2d(blurred, g_kernel, strides=[1, 1, 1, 1], padding="SAME")
    
    blurred = tf.squeeze(blurred, axis=0) if shape.ndims == 3 else blurred
    blurred = tf.cast(blurred, image.dtype)
    return blurred