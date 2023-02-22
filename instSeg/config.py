# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
import pickle
from instSeg.enumDef import *

class Config(object):

    def __init__(self, image_channel=3):

        self.verbose = False
        self.transfer_training = True

        self.model_type = MODEL_BASE
        self.save_best_metric = 'AJI' # 'AJI'/'AP'/
        # input size
        self.H = 512
        self.W = 512
        self.image_channel = image_channel
        # image normalization
        self.positional_padding = None
        self.input_normalization = 'per-image' # 'per-image', 'constant', None
        self.input_normalization_bias = 0
        self.input_normalization_scale = 1
        
        # backbone config
        self.backbone = 'uNet' # 'uNet', 'ResNet50', 'ResNet101', 'ResNet152', 'EfficientNetB0' - 'EfficientNetB7'
        self.filters = 64
        self.padding = 'same'
        self.weight_decay = 1e-5
        self.net_normalization = 'batch' # 'batch', None
        self.dropout_rate = 0
        self.up_scaling = 'deConv' # 'deConv', 'bilinear', 'nearest'
        self.concatenate = True
        ## config for specific to unet
        self.nstage = 4
        self.stage_conv = 2

        # losses
        self.loss = {'foreground': 'crossentropy', 
                     'contour': 'binary_crossentropy', 
                     'edt': 'mse', 
                     'flow': 'masked_mse', 
                     'embedding': 'cos',
                     'embedding_sigmoid': 'cos', 
                     'layered_embedding': 'sparse_cos'}
        self.loss_weights = {'foreground': 1, 'contour': 1, 'edt': 1, 'flow': 1, 'embedding': 1, 'layered_embedding': 1}
        self.focal_loss_gamma = 2.0
        self.ce_neg_ratio = 2
        self.sensitivity_specificity_loss_beta = 1.0
        
        self.post_processing = 'layered_embedding'

        ## semantic segmentation
        self.classes = 2

        ## foreground module
        self.thres_foreground = 0.5
        self.thres_layering = 0.5
        
        ## contour loss
        self.contour_radius = 1 
        self.contour_mode = 'overlap_midline' # 'overlap_ignore' or 'overlap_midline'
        self.thres_contour=0.5
        
        ## euclidean distance transform (EDT) regression
        self.edt_normalize = True
        self.edt_scaling = 10 if self.edt_normalize else 1
        self.edt_instance_thres = 5
        self.edt_fg_thres = 1
        
        ## embedding loss
        self.embedding_dim = 8
        self.embedding_include_bg = False
        self.dynamic_weighting = True
        self.neighbor_distance = 10
        # self.positional_embedding = None # 'global', 'harmonic'
        # self.octave = 4
        self.emb_clustering = 'meanshift'
        
        ## layered instances
        self.instance_layers = 8
        self.layering = 'embedding'
        
        # data augmentation
        self.flip = False
        self.random_rotation = False
        self.random_rotation_p = 0.5
        self.random_shift = False
        self.random_shift_p = 0.5
        self.max_shift = 64 # maximal shift
        self.random_gamma = False
        self.random_gamma_p = 0.2
        self.random_blur = False
        self.random_blur_p = 0.1
        self.random_saturation = False
        self.random_saturation_p = 0.2
        self.random_hue = False
        self.random_hue_p = 0.2
        self.elastic_deform = False
        self.elastic_deform_p = 0.8
        self.deform_scale = [16,256]
        self.deform_strength = 0.25

        # training config:
        self.ds_repeat = 1
        self.train_epochs = 100
        self.train_batch_size = 3
        self.train_learning_rate = 1e-4
        self.lr_decay_period = 10000
        self.lr_decay_rate = 0.9
        self.summary_step = 100

        # validation 
        self.snapshots = []
        self.validation_start_epoch = 1

        # post-process
        # object filtering
        self.obj_min_edt = 2
        self.obj_min_size=0
        self.obj_max_size=float('inf')

        
    
    def save(self, path):
        if path.endswith('.pkl'):
            with open(path, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

# for back compatability
ConfigCascade = Config
ConfigParallel = Config

# class ConfigContour(Config):

#     def __init__(self, image_channel=3):
#         super().__init__(image_channel=image_channel)

# class ConfigCascade(Config):

#     def __init__(self, image_channel=3):
#         super().__init__(image_channel=image_channel)
#         self.model_type = MODEL_CASCADE
#         self.modules = ['foreground', 'edt', 'embedding']
#         # config feature forward
#         self.feature_forward_dimension = 32
#         self.feature_forward_normalization = 'z-score' # 'z-score', 'l2'
#         self.stop_gradient = True

# class ConfigParallel(Config):

#     def __init__(self, image_channel=3):
#         super().__init__(image_channel=image_channel)
#         self.model_type = MODEL_PARALLEL
#         self.modules = ['foreground', 'edt']

