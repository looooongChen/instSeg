import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet with two heads
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

class UNnetDoubleHead(tf.keras.Model):

  def __init__(self,
               filters=32,
               dropout_rate=0.5,
               batch_norm=True,
               name='UNetDoubleHead',
               **kwargs):

    super(UNnetDoubleHead, self).__init__(name=name, **kwargs)
    self.batch_norm=batch_norm
    # encode_conv1
    self.Conv1_1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu1_1 = ReLU()
    self.Conv1_2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu1_2 = ReLU()
    self.Pool1 = MaxPooling2D(pool_size=(2, 2))
    if batch_norm:
        self.Norm1_1 = BatchNormalization()
        self.Norm1_2 = BatchNormalization()    
    # encode_conv2
    self.Drop2 = Dropout(dropout_rate)
    self.Conv2_1 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu2_1 = ReLU()
    self.Conv2_2 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu2_2 = ReLU()
    self.Pool2 = MaxPooling2D(pool_size=(2, 2))
    if batch_norm:
        self.Norm2_1 = BatchNormalization()
        self.Norm2_2 = BatchNormalization()
    # encode_conv3
    self.Drop3 = Dropout(dropout_rate)
    self.Conv3_1 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu3_1 = ReLU()
    self.Conv3_2 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu3_2 = ReLU()
    self.Pool3 = MaxPooling2D(pool_size=(2, 2))
    if batch_norm:
        self.Norm3_1 = BatchNormalization()
        self.Norm3_2 = BatchNormalization()
    # encode_conv4
    self.Drop4 = Dropout(dropout_rate)
    self.Conv4_1 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu4_1 = ReLU()
    self.Conv4_2 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu4_2 = ReLU()
    self.Pool4 = MaxPooling2D(pool_size=(2, 2))
    if batch_norm:
        self.Norm4_1 = BatchNormalization()
        self.Norm4_2 = BatchNormalization()
    # encode_conv5
    self.Drop5 = Dropout(dropout_rate)
    self.Conv5_1 = Conv2D(16 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu5_1 = ReLU()
    self.Conv5_2 = Conv2D(16 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu5_2 = ReLU()
    self.UpSample5 = UpSampling2D(size=(2, 2))
    self.UpConv5 = Conv2D(8 * filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp5 = ReLU()
    if batch_norm:
        self.Norm5_1 = BatchNormalization()
        self.Norm5_2 = BatchNormalization()
        self.NormUp5 = BatchNormalization()
    # decode_conv6_1
    self.Drop6_1 = Dropout(dropout_rate)
    self.Merge6_1 = Concatenate(axis=-1)
    self.Conv6_1_1 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu6_1_1 = ReLU()
    self.Conv6_1_2 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu6_1_2 = ReLU()
    self.UpSample6_1 = UpSampling2D(size=(2, 2))
    self.UpConv6_1 = Conv2D(4 * filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp6_1 = ReLU()
    if batch_norm:
        self.Norm6_1_1 = BatchNormalization()
        self.Norm6_1_2 = BatchNormalization()
        self.NormUp6_1 = BatchNormalization()
    # decode_conv6_2
    self.Drop6_2 = Dropout(dropout_rate)
    self.Merge6_2 = Concatenate(axis=-1)
    self.Conv6_2_1 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu6_2_1 = ReLU()
    self.Conv6_2_2 = Conv2D(8 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu6_2_2 = ReLU()
    self.UpSample6_2 = UpSampling2D(size=(2, 2))
    self.UpConv6_2 = Conv2D(4 * filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp6_2 = ReLU()
    if batch_norm:
        self.Norm6_2_1 = BatchNormalization()
        self.Norm6_2_2 = BatchNormalization()
        self.NormUp6_2 = BatchNormalization()
    # decode_conv7_1
    self.Drop7_1 = Dropout(dropout_rate)
    self.Merge7_1 = Concatenate(axis=-1)
    self.Conv7_1_1 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu7_1_1 = ReLU()
    self.Conv7_1_2 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu7_1_2 = ReLU()
    self.UpSample7_1 = UpSampling2D(size=(2, 2))
    self.UpConv7_1 = Conv2D(2 * filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp7_1 = ReLU()
    if batch_norm:
        self.Norm7_1_1 = BatchNormalization()
        self.Norm7_1_2 = BatchNormalization()
        self.NormUp7_1 = BatchNormalization()
    # decode_conv7_2
    self.Drop7_2 = Dropout(dropout_rate)
    self.Merge7_2 = Concatenate(axis=-1)
    self.Conv7_2_1 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu7_2_1 = ReLU()
    self.Conv7_2_2 = Conv2D(4 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu7_2_2 = ReLU()
    self.UpSample7_2 = UpSampling2D(size=(2, 2))
    self.UpConv7_2 = Conv2D(2 * filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp7_2 = ReLU()
    if batch_norm:
        self.Norm7_2_1 = BatchNormalization()
        self.Norm7_2_2 = BatchNormalization()
        self.NormUp7_2 = BatchNormalization()
    # decode_conv8_1
    self.Drop8_1 = Dropout(dropout_rate)
    self.Merge8_1 = Concatenate(axis=-1)
    self.Conv8_1_1 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu8_1_1 = ReLU()
    self.Conv8_1_2 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu8_1_2 = ReLU()
    self.UpSample8_1 = UpSampling2D(size=(2, 2))
    self.UpConv8_1 = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp8_1 = ReLU()
    if batch_norm:
        self.Norm8_1_1 = BatchNormalization()
        self.Norm8_1_2 = BatchNormalization()
        self.NormUp8_1 = BatchNormalization()
    # decode_conv8_2
    self.Drop8_2 = Dropout(dropout_rate)
    self.Merge8_2 = Concatenate(axis=-1)
    self.Conv8_2_1 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu8_2_1 = ReLU()
    self.Conv8_2_2 = Conv2D(2 * filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu8_2_2 = ReLU()
    self.UpSample8_2 = UpSampling2D(size=(2, 2))
    self.UpConv8_2 = Conv2D(filters, 1, padding='same', kernel_initializer='he_normal')
    self.ReluUp8_2 = ReLU()
    if batch_norm:
        self.Norm8_2_1 = BatchNormalization()
        self.Norm8_2_2 = BatchNormalization()
        self.NormUp8_2 = BatchNormalization()
    # decode_conv9_1
    self.Drop9_1 = Dropout(dropout_rate)
    self.Merge9_1 = Concatenate(axis=-1)
    self.Conv9_1_1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu9_1_1 = ReLU()
    self.Conv9_1_2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu9_1_2 = ReLU()
    if batch_norm:
        self.Norm9_1_1 = BatchNormalization()
        self.Norm9_1_2 = BatchNormalization()
    # decode_conv9_2
    self.Drop9_2 = Dropout(dropout_rate)
    self.Merge9_2 = Concatenate(axis=-1)
    self.Conv9_2_1 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu9_2_1 = ReLU()
    self.Conv9_2_2 = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
    self.Relu9_2_2 = ReLU()
    if batch_norm:
        self.Norm9_2_1 = BatchNormalization()
        self.Norm9_2_2 = BatchNormalization()

  def call(self, inputs):
    
    # conv1 
    conv1_1 = self.Conv1_1(inputs)
    conv1_1 = self.Relu1_1(self.Norm1_1(conv1_1)) if self.batch_norm else self.Relu1_1(conv1_1)
    conv1_2 = self.Conv1_2(conv1_1)
    conv1_2 = self.Relu1_1(self.Norm1_2(conv1_2)) if self.batch_norm else self.Relu1_2(conv1_2)
    pool1 = self.Pool1(conv1_2)
    # conv2
    drop2 = self.Drop2(pool1)
    conv2_1 = self.Conv2_1(drop2)
    conv2_1 = self.Relu2_1(self.Norm2_1(conv2_1)) if self.batch_norm else self.Relu2_1(conv2_1)
    conv2_2 = self.Conv2_2(conv2_1)
    conv2_2 = self.Relu2_2(self.Norm2_2(conv2_2)) if self.batch_norm else self.Relu2_2(conv2_2)
    pool2 = self.Pool2(conv2_2)
    # conv3
    drop3 = self.Drop3(pool2)
    conv3_1 = self.Conv3_1(drop3)
    conv3_1 = self.Relu3_1(self.Norm3_1(conv3_1)) if self.batch_norm else self.Relu3_1(conv3_1)
    conv3_2 = self.Conv3_2(conv3_1)
    conv3_2 = self.Relu3_2(self.Norm3_2(conv3_2)) if self.batch_norm else self.Relu3_2(conv3_2)
    pool3 = self.Pool3(conv3_2)
    # conv4
    drop4 = self.Drop4(pool3)
    conv4_1 = self.Conv4_1(drop4)
    conv4_1 = self.Relu4_1(self.Norm4_1(conv4_1)) if self.batch_norm else self.Relu4_1(conv4_1)
    conv4_2 = self.Conv4_2(conv4_1)
    conv4_2 = self.Relu4_2(self.Norm4_2(conv4_2)) if self.batch_norm else self.Relu4_2(conv4_2)
    pool4 = self.Pool4(conv4_2)
    # conv5
    drop5 = self.Drop5(pool4)
    conv5_1 = self.Conv5_1(drop5)
    conv5_1 = self.Relu5_1(self.Norm5_1(conv5_1)) if self.batch_norm else self.Relu5_1(conv5_1)
    conv5_2 = self.Conv5_2(conv5_1)
    conv5_2 = self.Relu5_2(self.Norm5_2(conv5_2)) if self.batch_norm else self.Relu5_2(conv5_2)
    up5 = self.UpSample5(conv5_2)
    up5 = self.UpConv5(up5)
    up5 = self.ReluUp5(self.NormUp5(up5)) if self.batch_norm else self.ReluUp5(up5)
    # conv6_1
    merge6_1 = self.Merge6_1([conv4_2, up5])
    drop6_1 = self.Drop6_1(merge6_1)
    conv6_1_1 = self.Conv6_1_1(drop6_1)
    conv6_1_1 = self.Relu6_1_1(self.Norm6_1_1(conv6_1_1)) if self.batch_norm else self.Relu6_1_1(conv6_1_1)
    conv6_1_2 = self.Conv6_1_2(conv6_1_1)
    conv6_1_2 = self.Relu6_1_2(self.Norm6_1_2(conv6_1_2)) if self.batch_norm else self.Relu6_1_2(conv6_1_2)
    up6_1 = self.UpSample6_1(conv6_1_2)
    up6_1 = self.UpConv6_1(up6_1)
    up6_1 = self.ReluUp6_1(self.NormUp6_1(up6_1)) if self.batch_norm else self.ReluUp6_1(up6_1)
    # conv6_2
    merge6_2 = self.Merge6_2([conv4_2, up5])
    drop6_2 = self.Drop6_2(merge6_2)
    conv6_2_1 = self.Conv6_2_1(drop6_2)
    conv6_2_1 = self.Relu6_2_1(self.Norm6_2_1(conv6_2_1)) if self.batch_norm else self.Relu6_2_1(conv6_2_1)
    conv6_2_2 = self.Conv6_2_2(conv6_2_1)
    conv6_2_2 = self.Relu6_2_2(self.Norm6_2_2(conv6_2_2)) if self.batch_norm else self.Relu6_2_2(conv6_2_2)
    up6_2 = self.UpSample6_2(conv6_2_2)
    up6_2 = self.UpConv6_2(up6_2)
    up6_2 = self.ReluUp6_2(self.NormUp6_2(up6_2)) if self.batch_norm else self.ReluUp6_2(up6_2)
    # conv7_1
    merge7_1 = self.Merge7_1([conv3_2, up6_1])
    drop7_1 = self.Drop7_1(merge7_1)
    conv7_1_1 = self.Conv7_1_1(drop7_1)
    conv7_1_1 = self.Relu7_1_1(self.Norm7_1_1(conv7_1_1)) if self.batch_norm else self.Relu7_1_1(conv7_1_1)
    conv7_1_2 = self.Conv7_1_2(conv7_1_1)
    conv7_1_2 = self.Relu7_1_2(self.Norm7_1_2(conv7_1_2)) if self.batch_norm else self.Relu7_1_2(conv7_1_2)
    up7_1 = self.UpSample7_1(conv7_1_2)
    up7_1 = self.UpConv7_1(up7_1)
    up7_1 = self.ReluUp7_1(self.NormUp7_1(up7_1)) if self.batch_norm else self.ReluUp7_1(up7_1)
    # conv7_2
    merge7_2 = self.Merge7_2([conv3_2, up6_2])
    drop7_2 = self.Drop7_2(merge7_2)
    conv7_2_1 = self.Conv7_2_1(drop7_2)
    conv7_2_1 = self.Relu7_2_1(self.Norm7_2_1(conv7_2_1)) if self.batch_norm else self.Relu7_2_1(conv7_2_1)
    conv7_2_2 = self.Conv7_2_2(conv7_2_1)
    conv7_2_2 = self.Relu7_2_2(self.Norm7_2_2(conv7_2_2)) if self.batch_norm else self.Relu7_2_2(conv7_2_2)
    up7_2 = self.UpSample7_2(conv7_2_2)
    up7_2 = self.UpConv7_2(up7_2)
    up7_2 = self.ReluUp7_2(self.NormUp7_2(up7_2)) if self.batch_norm else self.ReluUp7_2(up7_2)
    # conv8_1
    merge8_1 = self.Merge8_1([conv2_2, up7_1])
    drop8_1 = self.Drop8_1(merge8_1)
    conv8_1_1 = self.Conv8_1_1(drop8_1)
    conv8_1_1 = self.Relu8_1_1(self.Norm8_1_1(conv8_1_1)) if self.batch_norm else self.Relu8_1_1(conv8_1_1)
    conv8_1_2 = self.Conv8_1_2(conv8_1_1)
    conv8_1_2 = self.Relu8_1_2(self.Norm8_1_2(conv8_1_2)) if self.batch_norm else self.Relu8_1_2(conv8_1_2)
    up8_1 = self.UpSample8_1(conv8_1_2)
    up8_1 = self.UpConv8_1(up8_1)
    up8_1 = self.ReluUp8_1(self.NormUp8_1(up8_1)) if self.batch_norm else self.ReluUp8_1(up8_1)
    # conv8_2
    merge8_2 = self.Merge8_2([conv2_2, up7_2])
    drop8_2 = self.Drop8_2(merge8_2)
    conv8_2_1 = self.Conv8_2_1(drop8_2)
    conv8_2_1 = self.Relu8_2_1(self.Norm8_2_1(conv8_2_1)) if self.batch_norm else self.Relu8_2_1(conv8_2_1)
    conv8_2_2 = self.Conv8_2_2(conv8_2_1)
    conv8_2_2 = self.Relu8_2_2(self.Norm8_2_2(conv8_2_2)) if self.batch_norm else self.Relu8_2_2(conv8_2_2)
    up8_2 = self.UpSample8_2(conv8_2_2)
    up8_2 = self.UpConv8_2(up8_2)
    up8_2 = self.ReluUp8_2(self.NormUp8_2(up8_2)) if self.batch_norm else self.ReluUp8_2(up8_2)
    # conv9_1
    merge9_1 = self.Merge9_1([conv1_2, up8_1])
    drop9_1 = self.Drop9_1(merge9_1)
    conv9_1_1 = self.Conv9_1_1(drop9_1)
    conv9_1_1 = self.Relu9_1_1(self.Norm9_1_1(conv9_1_1)) if self.batch_norm else self.Relu9_1_1(conv9_1_1)
    conv9_1_2 = self.Conv9_1_2(conv9_1_1)
    conv9_1_2 = self.Relu9_1_2(self.Norm9_1_2(conv9_1_2)) if self.batch_norm else self.Relu9_1_2(conv9_1_2)
    # conv9_2
    merge9_2 = self.Merge9_2([conv1_2, up8_2])
    drop9_2 = self.Drop9_2(merge9_2)
    conv9_2_1 = self.Conv9_2_1(drop9_2)
    conv9_2_1 = self.Relu9_2_1(self.Norm9_2_1(conv9_2_1)) if self.batch_norm else self.Relu9_2_1(conv9_2_1)
    conv9_2_2 = self.Conv9_2_2(conv9_2_1)
    conv9_2_2 = self.Relu9_2_2(self.Norm9_2_2(conv9_2_2)) if self.batch_norm else self.Relu9_2_2(conv9_2_2)

    return conv9_1_2, conv9_2_2