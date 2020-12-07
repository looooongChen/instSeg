# construct the adjacent matrix using MAX_OBJ, 
# if the object number in your cases is large, increase it correspondingly 
MAX_OBJ = 300

class Config(object):

    def __init__(self, image_channel=3):

        self.verbose = False
        self.save_best_metric = 'mAJ' # 'loss'/'mAJ'/'mAP'
        # input size
        self.H = 512
        self.W = 512
        self.image_channel = image_channel
        
        # backbone config
        self.backbone = 'uNet'
        self.dropout_rate = 0.5
        self.batch_norm=False
        self.filters = 32

        # data augmentation
        self.flip = True
        self.elastic_strength = 2
        self.elastic_scale = 10
        self.rotation = True
        self.random_crop = True
        self.random_crop_range = (0.6, 0.8)

        # training config:
        self.stagewise_training = False
        self.train_epochs = 100
        self.train_batch_size = 3
        self.train_learning_rate = 0.0001
        self.lr_decay_period = 5000
        self.lr_decay_rate = 0.9


class ConfigContour(Config):

    def __init__(self, image_channel=3):
        super().__init__(image_channel=image_channel)
        # config of semantic
        self.classes = 2
        self.loss_semantic = 'dice_loss'
        self.weight_semantic = 1
        # config of contour
        self.loss_contour = 'focal_loss'
        self.weight_contour = 1 
        self.contour_radius = 1 

# class ConfigXXX(object):

#     def __init__(self, image_channel=3):

#         self.verbose = False
#         self.module_order = ['semantic', 'dist', 'embedding']

#         # input size
#         self.H = 512
#         self.W = 512
#         self.image_channel = image_channel
        
#         # backbone config
#         self.dropout_rate = 0.5
#         self.batch_norm=False

#         # config feature forward
#         self.feature_forward_dimension = 32
#         self.stop_gradient = True

#         # network config
#         self.filters = 32
#           
#         # config of semantic module
#         self.semantic = False
#         self.classes = 1
#         self.weight_semantic = 1 
#         self.loss_semantic = 'dice_loss'
#         # config of dist module
#         self.dist = False
#         self.weight_dist = 1
#         self.loss_dist = 'mse'
#         self.dist_neg_weight = 1
#         # config of embedding module
#         self.embedding = False
#         self.embedding_dim = 8
#         self.weight_embedding = 1
#         self.loss_embedding = 'cos'
#         self.embedding_include_bg = True
#         self.neighbor_distance = 0.03
#         self.max_obj = MAX_OBJ

#         # data augmentation
#         self.flip = True
#         self.elastic_strength = 2
#         self.elastic_scale = 10
#         self.rotation = True
#         self.random_crop = True
#         self.random_crop_range = (0.6, 0.8)

#         # training config:
#         self.stagewise_training = False
#         self.train_epochs = 100
#         self.train_batch_size = 3
#         self.train_learning_rate = 0.0001
#         self.lr_decay_period = 5000
#         self.lr_decay_rate = 0.9
    
