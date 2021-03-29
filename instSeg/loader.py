import pickle
import os
from instSeg.enumDef import *
from instSeg.model_base import *
from instSeg.model_cascade import *
from instSeg.model_parallel import *

def load_model(model_dir, load_best=True):
    config_file = os.path.join(model_dir, 'config.pkl')
    if os.path.exists(config_file):
        with open(config_file, 'rb') as input:
            config = pickle.load(input)
            if config.model_type == MODEL_BASE:
                model = ModelBase(config=config, model_dir=model_dir)
            elif config.model_type == MODEL_INST:
                model = InstSegMul(config=config, model_dir=model_dir)
            if config.model_type == MODEL_CASCADE:
                model = InstSegCascade(config=config, model_dir=model_dir)
                model.load_weights(load_best=load_best)
            elif config.model_type == MODEL_PARALLEL:
                model = InstSegParallel(config=config, model_dir=model_dir)
                model.load_weights(load_best=load_best)
            else:
                return None
        return model
    else:
        print('Config file not found: '+config_file)
        return None
