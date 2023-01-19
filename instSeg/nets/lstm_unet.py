import tensorflow as tf
from .blocks2d import *

normalization_types = ['batch', 'layer', None]
up_scaling_types = ['upConv', 'deConv']

def unet_lstm(inputs,
         filters=64,
         stages=4,
         convs=2,
         drop_rate=0,
         normalization='batch',
         stateful = False,
         up_scaling='upConv'):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types
    # encoding path
    p = inputs
    x_enc = []
    for idx in range(stages):
        drop_rate_ = 0 if idx == 0 else drop_rate
        x, p = block_encoder_convLSTM(p, filters*(2**idx), convs=convs, normalization=normalization, drop_rate=drop_rate_, name="encoder"+str(idx))
        x_enc.append(x)
    # bottleneck
    x = block_bottleneck_convLSTM(p, filters*(2**stages), convs=convs, normalization=normalization, drop_rate=drop_rate, name='bottleneck')
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder_convLSTM(x, x_enc.pop(), filters*(2**idx), convs=convs, normalization=normalization, drop_rate=drop_rate, up_scaling=up_scaling, name="decoder"+str(idx))
    
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model, x
    


if __name__ == "__main__":
    input_img = tf.keras.layers.Input((200, 64,64,3), name='input_img')
    model, fts = unet_lstm(input_img, stages=3, filters=32)
    
    # model = tf.keras.Model(inputs=input_img,
    #                      outputs=fts)

    model.summary()