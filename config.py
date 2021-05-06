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
        self.D = 4
        self.residual = False
        self.dropout_rate = 0.5
        self.batch_norm = True
        self.filters = 32
        self.net_upsample = 'interp' # 'interp', 'conv'
        self.net_merge = 'add' # 'add', 'cat'

        # losses
        self.focal_loss_gamma = 2.0
        self.sensitivity_specificity_loss_beta = 1.0
        ## semantic module
        self.classes = 2
        self.semantic_loss = 'dice'
        self.semantic_weight = 1
        self.semantic_in_ram = False
        ## contour loss
        self.contour_loss = 'focal_loss'
        self.contour_weight = 1
        self.contour_radius = 1 
        self.contour_in_ram = False
        ## euclidean dist transform regression
        self.edt_loss = 'masked_mse'
        self.edt_weight = 1
        self.edt_normalize = False
        self.edt_in_ram = False 
        ## euclidean dist transform flow regression
        self.edt_flow_loss = 'masked_mse'
        self.edt_flow_weight = 1
        self.edt_flow_normalize = True
        self.edt_flow_in_ram = False 
        ## embedding loss
        self.embedding_dim = 8
        self.embedding_loss = 'cos'
        self.embedding_weight = 1
        self.embedding_include_bg = True
        self.neighbor_distance = 0.03
        self.max_obj = MAX_OBJ

        # data augmentation
        self.flip = True
        self.elastic_deform = True
        self.elastic_strength = 200
        self.elastic_scale = 10
        self.random_rotation = True
        self.random_crop = True
        self.random_crop_range = (0.6, 0.8)
        self.random_gamma = True
        self.random_gamma_range = (0.5, 2)

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
        self.emb_max_step=float('inf')
        # distance regression map
        self.edt_mode = 'thresholding' # 'thresholding', 'tracking'
        self.tracking_iters = 10
        self.edt_instance_thres = 0.3
        self.edt_fg_thres = 0.05
        # semantic
        self.semantic_bg = 0
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

