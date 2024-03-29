import pickle
import os
from instSeg.enumDef import *
from instSeg.model_base import *
from instSeg.model import *
from instSeg.config import *
import tensorflow as tf

def load_model(model_dir, load_best=True, aug_dropout=0):
    tf.keras.backend.clear_session()
    config_file = os.path.join(model_dir, 'config.pkl')
    if os.path.exists(config_file):
        with open(config_file, 'rb') as input:
            config = pickle.load(input)
            if config.dropout_rate == 0 and aug_dropout > 0:
                config.dropout_rate = aug_dropout
            for k, v in Config().__dict__.items():
                if k not in config.__dict__.keys():
                    config.__dict__[k] = v
            model = Model(config=config, model_dir=model_dir)
            if load_best is not None:
                model.load_weights(load_best=load_best)
            else:
                return None
        return model
    else:
        print('Config file not found: '+config_file)
        return None
