# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
import pickle
from instSeg.enumDef import *

class Config(object):

    def __init__(self, image_channel=3):

        self.verbose = False
        self.transfer_training = True

        self.model_type = MODEL_BASE
        # input image size, all images will be resized for input
        self.H = 512
        self.W = 512
        self.image_channel = image_channel
        # guidance function
        self.guide_function_period = None # 'cosine'
        self.guide_function_type = None # a list of period that 
        # image normalization
        self.input_normalization = 'per-image' # 'per-image', 'constant', None
        self.input_normalization_bias = 0
        self.input_normalization_scale = 1
        
        # backbone config
        self.backbone = 'uNet' # 'uNet', 'ResNet50', 'ResNet101', 'ResNet152', 'EfficientNetB0' - 'EfficientNetB7'
        self.filters = 64
        self.padding = 'same'
        # self.weight_decay = 1e-5
        self.net_normalization = 'batch' # 'batch', None
        self.dropout_rate = 0
        self.up_scaling = 'deConv' # 'deConv', 'bilinear', 'nearest'
        self.concatenate = True
        ## config for specific to unet
        self.nstage = 4
        self.stage_conv = 2

        # outputs
        self.modules = ['foreground', '']
        
        # postprocessing
        self.post_processing = None # processing used to get final instance segmentation, if None, automatically selected
        self.allow_overlap = True

        # losses
        self.loss = {'foreground': 'CE', 
                     'contour': 'CE', 
                     'edt': 'MSE', 
                    #  'flow': 'masked_mse', 
                     'embedding': 'cosine'}
        self.loss_weights = {'foreground': 1, 'contour': 1, 'edt': 1, 'flow': 1, 'embedding': 1}
        self.focal_loss_gamma = 2.0
        self.neg_ratio = 2 # the proportion of negative pixels will be included in the training, for unblanced forground

        ## semantic segmentation
        # self.classes = 2

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
        self.embedding_activation = 'sigmoid' 
        self.embedding_include_bg = False
        self.embedding_regularization = 0
        self.dynamic_weighting = True
        self.neighbor_distance = 15 # if None, global constraint applies
        self.margin_attr = 0
        self.margin_rep = 0
        
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
        self.random_blur_p = 0.2
        self.random_saturation = False
        self.random_saturation_p = 0.2
        self.random_hue = False
        self.random_hue_p = 0.2
        self.elastic_deform = False
        self.elastic_deform_p = 0.8
        self.deform_scale = [16,256]
        self.deform_strength = 0.25

        # training config
        self.ds_repeat = 1
        self.shuffle_buffer = 64
        self.train_epochs = 100
        self.train_batch_size = 4
        self.optimizer = 'RMSprop'
        self.learning_rate = 1e-4
        self.lr_decay_period = 10000
        self.lr_decay_rate = 0.9
        self.summary_step = 100

        # validation config
        self.save_best_metric = 'loss' # 'AJI'/'AP'/'loss'
        self.snapshots = []
        self.validation_start_epoch = 10
        self.early_stoping_steps = None
        self.early_stoping_epochs = None

        # post-process
        self.obj_min_edt = 2
        self.obj_min_size = 0
        self.obj_max_size = float('inf')

    
    def save(self, path):
        if path.endswith('.pkl'):
            with open(path, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def info(self):
        text = '==== model config summary ===='
        print(text)
        # print('[INFO] model backbone: {}'.format(self.backbone))
        # print('[INFO] prediction branches: {}'.format(','.join(self.modules)))
        # print('[INFO] {} guidance function used with periods {}'.format(self.guide_function_type, ','.join(self.guide_function_period)))
        for k in dir(self):
            if k.startswith('__'):
                continue
            if k in ['info', 'save', 'exp_name']:
                continue
            line = '[INFO] {}: {}'.format(k, getattr(self, k)) 
            print(line)
            text = text + '\n' + line 
        return text



    
    def exp_name(self, preffix=None, suffix=None):

        # model
        arch = self.backbone 
        if self.backbone.lower() == 'unet':
            arch = arch + str(self.nstage) + 'f' + str(self.filters)
        arch = arch + self.up_scaling
        name = arch


        if 'embedding' in self.modules:
            # ouput
            out = 'D'+str(self.embedding_dim)+self.embedding_activation
            name = name + '_' + out
            # loss
            loss = self.loss['embedding']
            if self.neighbor_distance is None:
                loss = loss + 'Global'
            if self.embedding_include_bg:
                loss = loss + 'Bg'
            if self.margin_attr > 0:
                loss = loss + 'Ma{:.2}'.format(float(self.margin_attr))
            if self.margin_rep > 0:
                loss = loss + 'Mp{:.2}'.format(float(self.margin_rep))
            if self.embedding_regularization > 0:
                loss = loss + 'Reg'
            if self.dynamic_weighting:
                loss = loss + 'Dy'
            name = name + '_' + loss

        # # optimizer
        # optimizer = self.optimizer + str(self.learning_rate)+'decay'+str(self.lr_decay_rate)
        # name = name + '_' + optimizer
        # postprocessing
        if self.post_processing is not None:
            name = name + '_' + self.post_processing
        if isinstance(preffix, str) and len(preffix) != 0:
            name = preffix + '_' + name
        if isinstance(suffix, str) and len(suffix) != 0:
            name = name + '_' + suffix
        if self.guide_function_type is not None and self.guide_function_period is not None:
            period = [str(p) for p in self.guide_function_period]
            name = name + '_cood' + '-'.join(period)
        
        return name


# for backwards compatability
ConfigCascade = Config
ConfigParallel = Config

