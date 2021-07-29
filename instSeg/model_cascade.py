# import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.model_base import InstSegMul 
from instSeg.enumDef import *
from instSeg.uNet import *
from instSeg.utils import *
import os

class InstSegCascade(InstSegMul):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_CASCADE

    def build_model(self):
        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        self.normalized_img = tf.image.per_image_standardization(self.input_img)

        if self.config.backbone.startswith('uNet'):
            backbone_arch = lambda name: UNet(nfilters=self.config.filters,
                                            nstage=self.config.nstage,
                                            stage_conv=self.config.stage_conv,
                                            residual=self.config.residual,
                                            dropout_rate=self.config.dropout_rate,
                                            batch_norm=self.config.batch_norm,
                                            up_type=self.config.net_upsample, 
                                            merge_type=self.config.net_merge, 
                                            weight_decay=self.config.weight_decay,
                                            name=name)
        elif self.config.backbone.startswith('resnet'):
            if self.config.backbone == 'resnet50':
                backbone_arch = lambda name: ResNetSeg(input_shape=(self.config.H, self.config.W, 3), filters=self.config.filters, layers=50, name=name)
            elif self.config.backbone == 'resnet101':
                backbone_arch = lambda name: ResNetSeg(input_shape=(self.config.H, self.config.W, 3), filters=self.config.filters, layers=101, name=name)
        else:
            assert False, 'Architecture "' + self.config.backbone + '" not valid!'
        
        output_list = []
        for i, m in enumerate(self.config.modules):
            if i != 0:   
                feature_suppression = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', 
                                                             activation='linear', kernel_initializer='he_normal', 
                                                             name='feature_'+m)
                if self.config.stop_gradient:
                    features = tf.stop_gradient(tf.identity(features))
                features = feature_suppression(features)
                input_list = [self.normalized_img, tf.nn.l2_normalize(features, axis=-1)]
            else:
                input_list = [self.normalized_img]

            backbone = backbone_arch(name='net_'+m)
            features = backbone(K.concatenate(input_list, axis=-1))

            if m == 'semantic':
                outlayer = keras.layers.Conv2D(filters=self.config.classes, kernel_size=1, activation='softmax')
                output_list.append(outlayer(features))
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')
                output_list.append(outlayer(features))
            if m == 'edt':
                activation = 'sigmoid' if self.config.edt_loss == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, kernel_size=1, activation=activation)
                output_list.append(outlayer(features))
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, kernel_size=1, activation='linear')
                output_list.append(outlayer(features))      

        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
    


