import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D

class ResNetSeg(tf.keras.Model):

    def __init__(self,
                 input_shape=(512,512,3),
                 filters=32,
                 layers=50,
                 name='ResNetSeg',
                 **kwargs):

        super().__init__(name=name, **kwargs)

        if layers == 101:
            resnet = tf.keras.applications.ResNet101V2(include_top=False, 
                                                       weights='imagenet',
                                                       input_shape=input_shape)
        else:
            resnet = tf.keras.applications.ResNet50V2(include_top=False, 
                                                      weights='imagenet',
                                                      input_shape=input_shape)

        ft2 = resnet.get_layer('conv1_conv').output
        ft4 = resnet.get_layer('conv2_block2_out').output # stride 1/4
        ft8 = resnet.get_layer('conv3_block3_out').output # stride 1/8
        ft16 = resnet.get_layer('conv4_block5_out').output
        ft32 = resnet.get_layer('post_relu').output
        
        self.backbone = tf.keras.Model(inputs=resnet.inputs,
                                       outputs=[ft2, ft4, ft8, ft16, ft32])

        self.ft2_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.ft4_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.ft8_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.ft16_up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.ft32_up = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.ft2_conv = Conv2D(32, 3, padding='same')
        self.ft4_conv = Conv2D(64, 3, padding='same')
        self.ft8_conv = Conv2D(256, 3, padding='same')
        self.ft16_conv = Conv2D(512, 3, padding='same')
        self.ft32_conv = Conv2D(1024, 3, padding='same')

        self.ft_conv = Conv2D(filters, 3, padding='same')

        
    def call(self, inputs, training=False):
        
        ft2, ft4, ft8, ft16, ft32 = self.backbone(inputs, training)

        ft = self.ft32_up(self.ft32_conv(ft32)) + ft16 
        ft = self.ft16_up(self.ft16_conv(ft)) + ft8 
        ft = self.ft8_up(self.ft8_conv(ft)) + ft4 
        ft = self.ft4_up(self.ft4_conv(ft)) + ft2
        ft = self.ft2_up(self.ft2_conv(ft))
        ft = self.ft_conv(ft)

        return ft

if __name__ == "__main__":
    import numpy as np
    a = np.ones((2, 512, 512, 3))
    model = ResNetSeg(input_shape=(512, 512, 3))
    # model = tf.keras.applications.ResNet50V2(include_top=True, 
                                                    # weights='imagenet')
    # model = efficientNet = tf.keras.applications.EfficientNetB7(include_top=False, 
    #                                                         weights='imagenet', 
    #                                                         input_tensor=None,
    #                                                         input_shape=(512,512,3))
    # model.summary()
    # for l in model.layers:
    #     print(l.name)
    ft = model(a)
    print(ft.shape)
    # print(fts[0].shape, fts[1].shape, fts[2].shape, fts[3].shape, fts[4].shape)
    # print(f.shape)