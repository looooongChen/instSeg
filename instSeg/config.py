# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
import pickle
from instSeg.enumDef import *

MAX_OBJ = 300

class Config(object):

    def __init__(self, image_channel=3):

        self.verbose = False

        self.model_type = MODEL_BASE
        self.save_best_metric = 'mAJ' # 'mAJ'/'mAP'/'loss'
        # input size
        self.H = 512
        self.W = 512
        self.image_channel = image_channel
        
        # backbone config
        self.backbone = 'uNet' # 'uNet2H', 'uNetSA', 'uNnetD'
        self.dropout_rate = 0.5
        self.batch_norm = True
        self.filters = 32
        self.net_upsample = 'interp' # 'interp', 'conv'
        self.net_merge = 'cat' # 'add', 'cat'

        # losses
        self.focal_loss_gamma = 2.0
        self.sensitivity_specificity_loss_beta = 1.0
        ## semantic loss
        self.classes = 2
        self.loss_semantic = 'dice'
        self.weight_semantic = 1
        ## contour loss
        self.loss_contour = 'focal_loss'
        self.weight_contour = 1
        self.contour_radius = 1 
        ## dist regression 
        self.loss_dist = 'mse'
        self.weight_dist = 1
        ## embedding loss
        self.embedding_dim = 8
        self.loss_embedding = 'cos'
        self.weight_embedding = 1
        self.embedding_include_bg = True
        self.neighbor_distance = 0.03
        self.max_obj = MAX_OBJ

        # data augmentation
        self.flip = True
        self.elastic_strength = 2
        self.elastic_scale = 10
        self.rotation = True
        self.random_crop = True
        self.random_crop_range = (0.6, 0.8)

        # training config:
        self.train_epochs = 100
        self.train_batch_size = 3
        self.train_learning_rate = 1e-4
        self.lr_decay_period = 10000
        self.lr_decay_rate = 0.9

        # validation 
        self.validation_start_epoch = 1

        # post-process
        # dcan
        self.dcan_thres_contour=0.5
        # embedding
        self.embedding_cluster = 'argmax' # 'argmax', 'meanshift', 'mws' 
        self.emb_thres=0.7
        self.emb_dist_thres=5
        self.emb_dist_intensity = 0
        self.emb_max_step=float('inf')
        # for all
        self.min_size=0
        self.max_size=float('inf')
    
    def save(self, path):
        if path.endswith('.pkl'):
            with open(path, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# class ConfigContour(Config):

#     def __init__(self, image_channel=3):
#         super().__init__(image_channel=image_channel)

class ConfigCascade(Config):

    def __init__(self, image_channel=3):
        super().__init__(image_channel=image_channel)
        self.model_type = MODEL_CASCADE
        self.modules = ['semantic', 'dist', 'embedding']
        # config feature forward
        self.feature_forward_dimension = 32
        self.stop_gradient = True

class ConfigParallel(Config):

    def __init__(self, image_channel=3):
        super().__init__(image_channel=image_channel)
        self.model_type = MODEL_PARALLEL
        self.modules = ['semantic', 'contour']

