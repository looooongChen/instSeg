import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D
from .blocks2d import *
# import tempfile
import os


def resnet(inputs,
           filters=64,
           convs=2,
           drop_rate=0,
           normalization='batch',
           up_scaling='upConv',
           version='ResNet101',
           transfer_training=True):

    x = Conv2D(3, 1, activation='relu')(inputs)

    if version.lower() == 'resnet50':
        net = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    if version.lower() == 'resnet101':
        net = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet')
    if version.lower() == 'resnet152':
        net = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')

    # if weight_decay > 0:
    #     for layer in net.layers:
    #         if isinstance(layer, Conv2D):
    #             layer.kernel_regularizer = l2(weight_decay)
    #             layer.bias_regularizer=l2(weight_decay)

    #     tmp_weights_path = os.path.join(tempfile.gettempdir(), 'resnet_tmp_weights.h5')
    #     net.save_weights(tmp_weights_path)
    #     net = tf.keras.models.model_from_json(net.to_json())        
    #     net.load_weights(tmp_weights_path, by_name=True)
    
    net = tf.keras.Model(inputs=net.inputs,
                         outputs=[net.get_layer('conv1_conv').output, # stride 1/2
                                  net.get_layer('conv2_block2_out').output, # stride 1/4
                                  net.get_layer('conv3_block3_out').output, # stride 1/8
                                  net.get_layer('conv4_block5_out').output, # stride 1/16
                                  net.get_layer('conv5_block3_out').output]) # stride 1/32
    
    if transfer_training:
        ft2, ft4, ft8, ft16, ft32 = net(x, training=False)
    else:
        ft2, ft4, ft8, ft16, ft32 = net(x, training=True)

    # filters = [2048, 1024, 512, 256, 64]
    nfilters = [filters*(2**i) for i in reversed(range(5))]
    ft32 = Conv2D(nfilters[0], 1, padding='same', activation='relu', name='ft32_compress')(ft32)
    ft32 = normlize(ft32, normalization=normalization, name='ft32_compress_norm')
    ft16 = Conv2D(nfilters[1], 1, padding='same', activation='relu', name='ft16_compress')(ft16)
    ft16 = normlize(ft16, normalization=normalization, name='ft16_compress_norm')
    ft8 = Conv2D(nfilters[2], 1, padding='same', activation='relu', name='ft8_compress')(ft8)
    ft8 = normlize(ft8, normalization=normalization, name='ft8_compress_norm')
    ft4 = Conv2D(nfilters[3], 1, padding='same', activation='relu', name='ft4_compress')(ft4)
    ft4 = normlize(ft4, normalization=normalization, name='ft4_compress_norm')
    ft2 = Conv2D(nfilters[4], 1, padding='same', activation='relu', name='ft2_compress')(ft2)
    ft2 = normlize(ft2, normalization=normalization, name='ft2_compress_norm')

    fts_enc = [ft32, ft16, ft8, ft4, ft2]
    fts = ft32
    for idx in range(1, 5):
        fts = block_decoder(fts, fts_enc[idx], nfilters[idx], convs=convs, normalization=normalization, drop_rate=drop_rate, up_scaling=up_scaling, name="decoder"+str(idx))

    fts = UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_up')(fts)
    fts = Conv2D(filters, 1, padding='same', activation='relu', name='final_conv')(fts)

    return net, fts





