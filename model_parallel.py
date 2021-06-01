# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import InstSegMul 
from instSeg.enumDef import *
from instSeg.uNet import *
from instSeg.ResNetSeg import *
from instSeg.utils import *
import os


class InstSegParallel(InstSegMul):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_PARALLEL
        # assert len(config.modules) == 2

    def build_model(self):

        if self.config.backbone.startswith('resnet'):
            assert self.config.image_channel == 3

        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        output_list = []

        if self.config.backbone.startswith('uNet'):
            self.net = UNet(pooling_stage=self.config.pooling_stage,
                            block_conv=self.config.block_conv,
                            padding='same',
                            residual=self.config.residual,
                            filters=self.config.filters,
                            dropout_rate=self.config.dropout_rate,
                            batch_norm=self.config.batch_norm,
                            upsample=self.config.net_upsample,
                            merge=self.config.net_merge,
                            name='UNet')
        elif self.config.backbone.startswith('resnet'):
            if self.config.backbone == 'resnet50':
                self.net = ResNetSeg(input_shape=(self.config.H, self.config.W, 3), filters=self.config.filters, layers=50, name='resnet50')
            elif self.config.backbone == 'resnet101':
                self.net = ResNetSeg(input_shape=(self.config.H, self.config.W, 3), filters=self.config.filters, layers=101, name='resnet101')
        else:
            assert False, 'Architecture "' + self.config.backbone + '" not valid!'

        if self.config.backbone == 'uNet2H':
            features1, features2 = self.net(self.normalized_img)
            features = [features1, features2]
        else:
            features = self.net(self.normalized_img)
            features = [features, features]

        output_list = []
        for idx, m in enumerate(self.config.modules):
            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, kernel_size=1, activation='softmax')
                out = outlayer(features[idx])
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
                out = outlayer(features[idx])
            if m == 'edt':
                activation = 'sigmoid' if self.config.edt_loss == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation=activation)
                out = outlayer(features[idx])
            if m == 'edt_flow':
                outlayer = keras.layers.Conv2D(filters=2, kernel_size=1, activation='linear')
                out = outlayer(features[idx])
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, kernel_size=1, activation='linear')
                out = outlayer(features[idx])
            output_list.append(out)     
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()