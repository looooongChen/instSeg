import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose, LayerNormalization, ConvLSTM2D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Add, Cropping2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
'''
-- Long Chen, LfB, RWTH Aachen University --
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''


def normlize(x, normalization='batch', name=None):
    # batch normalization
    if normalization == 'batch':
        return BatchNormalization(name=name)(x)
    elif normalization == 'layer':
        return LayerNormalization(name=name)(x)
    else:
        return x


def dropout(x, drop_rate, name):
    if drop_rate > 0:
        return Dropout(drop_rate, name=name)(x)
    else:
        return x


def block_conv2D(x, filters, convs=2, conv_padding='same',
                 residual=False, skip_conv=False, 
                 normalization='batch', name=None):
    '''
    a seqence of convolutional layers
    Args:
        x: input tensor
        filters: number of filters
        convs: number of convolutional layers
        residual: skip connection between input and output, the block learns the residual
        skip_conv: use it, if the input and output are of different size
        normalization: 'batch' or None
    '''
    y = x
    for idx in range(convs):
        # conv layers
        name_conv = name + '_Conv' + str(idx) if name is not None else None
        y = Conv2D(filters, 3, padding=conv_padding, activation='relu', name=name_conv)(y)
        # normalization layers
        name_norm = name + '_Norm' + str(idx) if name is not None else None
        y = normlize(y, normalization=normalization, name=name_norm)
    if residual:
        if skip_conv:
            name_skip_conv = name + '_SkipConv' if name is not None else None
            x = Conv2D(filters, 1, activation='relu', name=name_skip_conv)(y)
        if conv_padding == 'valid':
            y = Cropping2D(cropping=convs)(y)
        y = x + y
    return y

# def block_convLSTM2D(x, filters, convs=2, normalization='batch', stateful=False, name=None):
#     '''
#     a seqence of convolutional layers
#     Args:
#         x: input tensor
#         filters: number of filters
#         convs: number of convolutional layers
#     '''
#     for idx in range(convs):
#         name_conv = name + '_ConvLSTM' + str(idx) if name is not None else None
#         x = ConvLSTM2D(filters=filters, kernel_size=3, padding="same", activation="relu", return_sequences=True, stateful=stateful, name=name_conv)(x)
#         # normalization layers
#         name_norm = name + '_Norm' + str(idx) if name is not None else None
#         x = normlize(x, normalization=normalization, name=name_norm)

#     return x


def block_encoder2D(x, filters, convs=2, conv_padding='same',
                    residual=False, skip_conv=False, 
                    normalization='batch', drop_rate=0, name=None):
    '''
    dropout (optional) + convs + pooling
    Args:
        x: input tensor
        filters: number of filters
        convs: number of convolutional layers
        residual: skip connection between input and output, the block learns the residual
        skip_conv: use it, if the input and output are of different size
        normalization: 'batch' or None
    '''
    # dropout layer before conv
    name_drop = name + '_Dropout' if name is not None else None
    x = dropout(x, drop_rate, name=name_drop)
    # conv layers
    x = block_conv2D(x=x, filters=filters, convs=convs, residual=residual, skip_conv=skip_conv, conv_padding=conv_padding, normalization=normalization, name=name)
    # pooling layer
    name_pool = name + '_Pool' if name is not None else None
    p = MaxPooling2D(pool_size=(2,2), name=name_pool)(x)
    # p = AveragePooling2D(pool_size=(2,2), name=name_pool)(x)
    return x, p

block_encoder = block_encoder2D

# def block_encoder_convLSTM(x, filters, convs=2, normalization='batch', drop_rate=0, stateful=False, name=None):
#     '''
#     dropout (optional) + convs + pooling
#     '''
#     # dropout layer before conv
#     name_drop = name + '_Dropout' if name is not None else None
#     x = dropout(x, drop_rate, name=name_drop)
#     # conv layers
#     x = block_convLSTM2D(x=x, filters=filters, convs=convs, normalization=normalization, stateful=stateful, name=name)
#     # pooling layer
#     name_pool = name + '_Pool' if name is not None else None
#     p = MaxPooling3D(pool_size=(1,2,2), name=name_pool)(x)
#     return x, p


def block_bottleneck2D(x, filters, convs=2, conv_padding='same',
                       residual=False, skip_conv=False,
                       normalization='batch', drop_rate=0, name=None):
    '''
    dropout (optional) + convs
    Args:
        x: input tensor
        filters: number of filters
        convs: number of convolutional layers
        residual: skip connection between input and output, the block learns the residual
        skip_conv: use it, if the input and output are of different size
        normalization: 'batch' or None
    '''
    # dropout layer before conv
    name_drop = name + '_Dropout' if name is not None else None
    x = dropout(x, drop_rate, name=name_drop)
    # conv layers
    x = block_conv2D(x=x, filters=filters, convs=convs, residual=residual, skip_conv=skip_conv, conv_padding=conv_padding, normalization=normalization, name=name)
    return x


block_bottleneck = block_bottleneck2D

# def block_bottleneck_convLSTM(x, filters, convs=2, normalization='batch', drop_rate=0, stateful=False, name=None):
#     '''
#     dropout (optional) + convs + pooling
#     '''
#     # dropout layer before conv
#     name_drop = name + '_Dropout' if name is not None else None
#     x = dropout(x, drop_rate, name=name_drop)
#     # conv layers
#     x = block_convLSTM2D(x=x, filters=filters, convs=convs, normalization=normalization, stateful=stateful, name=name)
#     return x


def block_decoder2D(x, x_enc, filters, convs=2, conv_padding='same',
                    residual=False, skip_conv=False, 
                    up_scaling='deConv', concatenate=True,
                    cat_crop=0, cat_conv=False, 
                    normalization='batch', 
                    drop_rate=0, 
                    name=None):
    '''
    up + concatenate + dropout (optional) + convs 
    Args:
        x: input tensor
        filters: number of filters
        convs: number of convolutional layers
        residual: skip connection between input and output, the block learns the residual
        skip_conv: use it, if the input and output are of different size
        up_scaling: 'deConv', 'bilinear', 'nearest'
        concatenate: True for concatenate, False for add
        normalization: 'batch' or None        
    '''
    # up layer
    name_up = name + '_Up' if name is not None else None
    if up_scaling == 'bilinear':
        x = UpSampling2D(size=(2, 2), interpolation='bilinear', name=name_up)(x)
    elif up_scaling == 'nearest':
        x = UpSampling2D(size=(2, 2), interpolation='nearest', name=name_up)(x)
    else:
        x = Conv2DTranspose(filters, 2, 2, padding='same', activation='relu', name=name_up)(x)
    if up_scaling != 'deConv' and cat_conv is True:
        x = Conv2D(filters, 1, activation='relu', name=name_up+'Conv')(x)
    # normalization layer
    name_norm = name + '_Norm' if name is not None else None
    x = normlize(x, normalization=normalization, name=name_norm)
    # concatenate
    cat_crop = int(cat_crop)
    if cat_crop > 0:
        x_enc = Cropping2D(cropping=cat_crop)(x_enc)
    if concatenate:
        name_concat = name + '_Concat' if name is not None else None
        x = Concatenate(axis=-1, name=name_concat)([x, x_enc])
    else:
        name_add = name + '_Add' if name is not None else None
        x = Add(name=name_add)([x, x_enc])

    # dropout layer before conv
    name_drop = name + '_Dropout' if name is not None else None
    x = dropout(x, drop_rate, name=name_drop)
    # convs
    x = block_conv2D(x=x, filters=filters, convs=convs, residual=residual, skip_conv=skip_conv, conv_padding=conv_padding, normalization=normalization, name=name)

    return x

block_decoder = block_decoder2D

# def block_decoder_convLSTM(x, x_enc, filters, convs=2, normalization='batch', drop_rate=0, up_scaling='deConv', stateful=False, name=None):
#     '''
#     up + concatenate + dropout (optional) + convs 
#     '''
#     # up layer
#     name_up = name + '_Up' if name is not None else None
#     if up_scaling == 'upConv':
#         x = UpSampling3D(size=(1, 2, 2), name=name_up)(x)
#         x = Conv2D(filters, 1, activation='relu', name=name_up+'Conv')(x)
#     else:
#         x = Conv3DTranspose (filters, (1,2,2), (1,2,2), padding='same', activation='relu', name=name_up)(x)
#     # normalization layer
#     name_norm = name + '_Norm' if name is not None else None
#     x = normlize(x, normalization=normalization, name=name_norm)
#     # concatenate
#     name_concat = name + '_Concat' if name is not None else None
#     x = Concatenate(axis=-1, name=name_concat)([x, x_enc])
#     # dropout layer before conv
#     name_drop = name + '_Dropout' if name is not None else None
#     x = dropout(x, drop_rate, name=name_drop)
#     # convs
#     x = block_convLSTM2D(x=x, filters=filters, convs=convs, normalization=normalization, stateful=stateful, name=name)

#     return x


