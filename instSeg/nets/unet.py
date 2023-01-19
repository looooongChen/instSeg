import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from .blocks2d import *

normalization_types = ['batch', 'layer', None]
up_scaling_types_unet = ['upConv', 'deConv']
up_scaling_types_cunet = ['upConv', 'deConv', 'up']

def unet(inputs,
         filters=64,
         stages=4,
         convs=2,
         drop_rate=0,
         normalization='batch',
         up_scaling='upConv',
         concatenate=True):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types_unet
    # encoding path
    p = inputs
    x_enc = []
    for idx in range(stages):
        drop_rate_ = 0 if idx == 0 else drop_rate
        x, p = block_encoder(p, filters*(2**idx), convs=convs, normalization=normalization, drop_rate=drop_rate_, name="encoder"+str(idx))
        x_enc.append(x)
    # bottleneck
    x = block_bottleneck(p, filters*(2**stages), convs=convs, normalization=normalization, drop_rate=drop_rate, name='bottleneck')
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder(x, x_enc.pop(), filters*(2**idx), convs=convs, normalization=normalization, drop_rate=drop_rate, up_scaling=up_scaling, concatenate=concatenate, name="decoder"+str(idx))
    
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model, x
    

def cunet(inputs,
          filters=64,
          stages=4,
          convs=2,
          residual=False,
          drop_rate=0,
          normalization='batch',
          up_scaling='up'):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types_cunet
    # encoding path
    p = inputs
    x_enc = []
    for idx in range(stages):
        drop_rate_ = 0 if idx == 0 else drop_rate
        skip = True if idx == 0 else False
        x, p = block_encoder(p, filters, convs=convs, residual=residual, skip_conv=skip, normalization=normalization, drop_rate=drop_rate_, name="encoder"+str(idx))
        x_enc.append(x)
    # bottleneck
    x = block_bottleneck(p, filters, convs=convs, residual=residual, skip_conv=False, normalization=normalization, drop_rate=drop_rate, name='bottleneck1')
    x = block_bottleneck(x, filters, convs=convs, residual=residual, skip_conv=False, normalization=normalization, drop_rate=drop_rate, name='bottleneck2')
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder(x, x_enc.pop(), filters, convs=convs, residual=residual, skip_conv=False, normalization=normalization, drop_rate=drop_rate, up_scaling=up_scaling, concatenate=False, name="decoder"+str(idx))
    
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model, x


if __name__ == "__main__":
    input_img = tf.keras.layers.Input((256,256,3), name='input_img')
    model, fts = unet(input_img, filters=64)
    
    # model = tf.keras.Model(inputs=input_img,
    #                      outputs=fts)

    model.summary()