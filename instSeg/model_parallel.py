import tensorflow as tf 
from instSeg.model_base import ModelBase, input_process, module_output
from instSeg.enumDef import *
from instSeg.net_factory import net_factory
from instSeg.nets.blocks2d import block_conv2D
import os


class InstSegParallel(ModelBase):

    def __init__(self, config, model_dir='./'):
        super().__init__(config, model_dir)
        self.config.model_type = MODEL_PARALLEL

    def build_model(self):
        self.input_img = tf.keras.layers.Input((self.config.H, self.config.W, self.config.image_channel), name='input_img')
        x = input_process(self.input_img, self.config)

        self.backbone, self.features = net_factory(x, self.config)

        output_list = []
        for m in self.config.modules:
            features = block_conv2D(self.features, filters=self.config.filters, convs=2, normalization=self.config.net_normalization, name=m)
            y = module_output(features, m, self.config)
            output_list.append(y)     

        self.model = tf.keras.Model(inputs=self.input_img, outputs=output_list)

        if self.config.verbose:
            self.model.summary()