# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import InstSegMul 
from instSeg.uNet import *
from instSeg.uNet2H import *
from instSeg.utils import *
import os


class InstSegParallel(InstSegMul):

    def __init__(self, config, base_dir='./', run_name=''):
        super().__init__(config, base_dir, run_name)
        assert len(config.modules) == 2

    def build_model(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        output_list = []

        if self.config.backbone == 'uNet2H':
            backbone = UNet2H 
        elif self.config.backbone == 'uNetSA':
            backbone = UNetSA
        elif self.config.backbone == 'uNetD':
            backbone = UNetD
        elif self.config.backbone == 'uNet': 
            backbone = UNet
        else:
            assert False, 'Architecture "' + self.config.backbone + '" not valid!'

        self.net = backbone(filters=self.config.filters,
                            dropout_rate=self.config.dropout_rate,
                            batch_norm=self.config.batch_norm,
                            upsample=self.config.net_upsample,
                            merge=self.config.net_merge,
                            name='net')
        if self.config.backbone == 'uNet2H':
            features1, features2 = self.net(self.normalized_img)
            features = [features1, features2]
        else:
            features = self.net(self.normalized_img)
            features = [features, features]

        output_list = []
        for idx, m in enumerate(self.config.modules):
            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, 
                                               kernel_size=3, padding='same', activation='softmax', 
                                               kernel_initializer='he_normal',
                                               name='out_semantic')
                output_list.append(outlayer(features[idx]))
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation='sigmoid', 
                                               kernel_initializer='he_normal', 
                                               name='out_contour')
                output_list.append(outlayer(features[idx]))
            if m == 'dist':
                activation = 'sigmoid' if self.config.loss_dist == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation=activation, 
                                               kernel_initializer='he_normal',
                                               name='out_dist')
                output_list.append(outlayer(features[idx]))
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                               kernel_size=3, padding='same', activation='linear', 
                                               kernel_initializer='he_normal', 
                                               name='out_embedding')
                output_list.append(outlayer(features[idx]))     
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()