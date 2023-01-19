import tensorflow as tf 
from tensorflow import keras
import tensorflow.keras.backend as K 
from instSeg.model_base import * 
from instSeg.enumDef import *
from instSeg.layers import  *
from instSeg.net_factory import net_factory
from instSeg.utils import *
import os

class InstSegCascade(ModelBase):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_CASCADE

    def build_model(self):

        self.input_img = keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        x = input_process(self.input_img, self.config)
        
        if len(self.config.modules) > 1 and self.config.feature_forward_dimension > 0:
            if self.config.feature_forward_normalization == 'z-score' and self.config.input_normalization == 'per-image':
                x_forward = tf.image.per_image_standardization(x)
            if self.config.feature_forward_normalization == 'l2':
                x_forward = tf.nn.l2_normalize(x, axis=-1)
            else:
                x_forward = x

        self.modules = []
        output_list = []
        for i, m in enumerate(self.config.modules):
            if i != 0 and self.config.feature_forward_dimension > 0:   
                if self.config.stop_gradient:
                    features = tf.stop_gradient(tf.identity(features))
                features = tf.keras.layers.Conv2D(self.config.feature_forward_dimension, 1, padding='same', activation='linear')(features)
                if self.config.feature_forward_normalization == 'z-score':
                    features = tf.image.per_image_standardization(features)
                elif self.config.feature_forward_normalization == 'l2':
                    features = tf.nn.l2_normalize(features, axis=-1)
                else:
                    features = features
                input_list = [x_forward, features]
            else:
                input_list = [x]

            features = net_factory(K.concatenate(input_list, axis=-1), self.config)
            if isinstance(m, list):
                for mm in m:
                    y = module_output(features, mm, self.config)
                    output_list.append(y) 
                    self.modules.append(mm)
            else:  
                y = module_output(features, m, self.config)
                output_list.append(y)   
                self.modules.append(m)
                    
        self.model = keras.Model(inputs=self.input_img, outputs=output_list)
        
        if self.config.verbose:
            self.model.summary()
    


