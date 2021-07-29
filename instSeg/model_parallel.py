# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K
from instSeg.model_base import InstSegMul 
from instSeg.enumDef import *
from instSeg.uNet import *
from instSeg.ResNetSeg import *
from instSeg.utils import *
from instSeg.layers import *
import tensorflow as tf
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

        if self.config.positional_input is not None:
            coords = tf.repeat(self.coords, tf.shape(self.normalized_img)[0], axis=0)
            coords = tf.stop_gradient(K.cast_to_floatx(coords))
            self.normalized_img = tf.concat([self.normalized_img, coords], axis=-1)

        output_list = []

        if self.config.backbone.startswith('uNet'):
            self.net = UNet(nfilters=self.config.filters,
                            nstage=self.config.nstage,
                            stage_conv=self.config.stage_conv,
                            residual=self.config.residual,
                            dropout_rate=self.config.dropout_rate,
                            batch_norm=self.config.batch_norm,
                            up_type=self.config.net_upsample, 
                            merge_type=self.config.net_merge, 
                            weight_decay=self.config.weight_decay,
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
                if self.config.embedding_positional is None:
                    outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, kernel_size=1, activation='linear')
                elif self.config.embedding_positional == 'trainable':
                    outlayer = ConvPos(self.config.embedding_dim, channels=1, kernel_size=1, activation='linear')
                else:
                    outlayer = EmbeddingPos(self.config.embedding_dim, self.config.embedding_positional)
                out = outlayer(features[idx])
            output_list.append(out)     
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()