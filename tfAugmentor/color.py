import tensorflow as tf

def adjust_contrast(image, contrast_factor):

    '''
    Args:
        image: 3D or 4D
        contrast_factor: scalar or 1D
    '''

    shape = image.get_shape()
    image_c = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image

    sz = image_c.get_shape()
    batch_size, height, width, channels = sz[0], sz[1], sz[2], sz[3]

    contrast_factor = tf.stop_gradient(tf.convert_to_tensor(contrast_factor))
    contrast_factor = tf.expand_dims(contrast_factor, axis=0) if contrast_factor.get_shape().ndims == 0 else contrast_factor
    contrast_factor = tf.repeat(contrast_factor, batch_size, axis=0) if contrast_factor.get_shape()[0] != batch_size else contrast_factor

    image_c = tf.unstack(image_c, axis=0)
    contrast_factor = tf.unstack(contrast_factor, axis=0)
    image_c = [tf.image.adjust_contrast(img, f) for img, f in zip(image_c, contrast_factor)]
    image_c = tf.concat(image_c, axis=0)

    return image_c


def adjust_gamma(image, gamma):

    '''
    Args:
        image: 3D or 4D
        gamma: scalar or 1D
    '''

    shape = image.get_shape()
    image_c = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image

    sz = image_c.get_shape()
    batch_size, height, width, channels = sz[0], sz[1], sz[2], sz[3]

    gamma = tf.stop_gradient(tf.convert_to_tensor(gamma))
    gamma = tf.expand_dims(gamma, axis=0) if gamma.get_shape().ndims == 0 else gamma
    gamma = tf.repeat(gamma, batch_size, axis=0) if gamma.get_shape()[0] != batch_size else gamma
    
    image_c = tf.unstack(image_c, axis=0)
    gamma = tf.unstack(gamma, axis=0)
    image_c = [tf.image.adjust_gamma(img, f) for img, f in zip(image_c, gamma)]
    image_c = tf.concat(image_c, axis=0)

    return image_c