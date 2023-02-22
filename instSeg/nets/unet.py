import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from .blocks2d import *

normalization_types = ['batch', 'layer', None]
up_scaling_types = ['deConv', 'bilinear', 'nearest']

def unet(inputs,
         filters=64,
         stages=4,
         convs=2,
         padding='same',
         drop_rate=0,
         normalization='batch',
         up_scaling='deConv',
         concatenate=True,
         encoder_only=False):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types
    # encoding path
    p = inputs
    crop, rp = 0, 1
    x_enc, crop_enc = [], []
    for idx in range(stages):
        drop_rate_ = 0 if idx == 0 else drop_rate
        x, p = block_encoder(p, filters*(2**idx), convs=convs, conv_padding=padding,
                             normalization=normalization, drop_rate=drop_rate_, name="encoder"+str(idx))
        x_enc.append(x)

        crop = crop + convs * (2 ** idx) if padding == 'valid' else crop
        crop_enc.append(crop)
        rp = rp + (2 ** idx) * 2 * convs + (2 ** idx)
    # bottleneck
    x = block_bottleneck(p, filters*(2**stages), convs=convs, conv_padding=padding,
                         normalization=normalization, drop_rate=drop_rate, name='bottleneck')
    crop = crop + convs * (2 ** stages) if padding == 'valid' else crop
    rp = rp + (2 ** stages) * 2 * convs 
    if encoder_only:
        model = tf.keras.Model(inputs=inputs, outputs=x)
        props = {'crop': crop, 'receptive field': rp}

        return model, x, props
    # decoding
    for idx in reversed(range(stages)):
        cat_crop = 0 if padding == 'same' else (crop - crop_enc.pop())/(2 ** idx)
        cat_conv = True if up_scaling != 'deConv' else False
        x = block_decoder(x, x_enc.pop(), filters*(2**idx), convs=convs, conv_padding=padding,
                          normalization=normalization, drop_rate=drop_rate, 
                          up_scaling=up_scaling, concatenate=concatenate, 
                          cat_crop=cat_crop, cat_conv=cat_conv, name="decoder"+str(idx))
        crop = crop + convs * (2 ** idx) if padding == 'valid' else crop
        rp = rp + (2 ** idx) * 2 * convs
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    props = {'crop': crop, 'receptive field': rp}

    return model, x, props
    

def cunet(inputs,
          filters=64,
          stages=4,
          convs=2,
          padding='same',
          residual=False,
          drop_rate=0,
          normalization='batch',
          up_scaling='deConv',
          concatenate=True):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types
    # encoding path
    p = inputs
    crop, rp = 0, 1
    x_enc, crop_enc = [], []
    for idx in range(stages):
        drop_rate_ = 0 if idx == 0 else drop_rate
        residual_ = False if idx == 0 else residual
        x, p = block_encoder(p, filters, convs=convs, conv_padding=padding, residual=residual_, skip_conv=False, 
                             normalization=normalization, drop_rate=drop_rate_, name="encoder"+str(idx))
        x_enc.append(x)

        crop = crop + convs * (2 ** idx) if padding == 'valid' else crop
        crop_enc.append(crop)
        rp = rp + (2 ** idx) * 2 * convs + (2 ** idx)
    # bottleneck
    x = block_bottleneck(p, filters, convs=convs, conv_padding=padding, residual=residual, skip_conv=False, 
                         normalization=normalization, drop_rate=drop_rate, name='bottleneck1')
    x = block_bottleneck(x, filters, convs=convs, conv_padding=padding, residual=residual, skip_conv=False, 
                         normalization=normalization, drop_rate=drop_rate, name='bottleneck2')
    crop = crop + 2 * convs * (2 ** stages) if padding == 'valid' else crop
    rp = rp + 2 * (2 ** stages) * 2 * convs 
    # decoding
    for idx in reversed(range(stages)):
        cat_crop = 0 if padding == 'same' else (crop - crop_enc.pop())/(2 ** idx)
        skip_conv = True if concatenate is True else False
        x = block_decoder(x, x_enc.pop(), filters, convs=convs, conv_padding=padding,
                          residual=residual, skip_conv=skip_conv,
                          normalization=normalization, drop_rate=drop_rate, 
                          up_scaling=up_scaling, concatenate=concatenate, 
                          cat_crop=cat_crop, cat_conv=False, name="decoder"+str(idx))
        crop = crop + convs * (2 ** idx) if padding == 'valid' else crop
        rp = rp + (2 ** idx) * 2 * convs
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    props = {'crop': crop, 'receptive field': rp}

    return model, x, props


if __name__ == "__main__":
    input_img = tf.keras.layers.Input((256,256,3), name='input_img')
    model, fts, _ = unet(input_img, filters=64)
    
    # model = tf.keras.Model(inputs=input_img,
    #                      outputs=fts)

    model.summary()