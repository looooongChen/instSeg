import tensorflow as tf
import tensorflow.keras.backend as K 
from instSeg.utils import disk_tf
# import numpy as np
# from utils import disk_tf

#######################
#### corss entropy ####
#######################

def ce(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x C, 'softmax' activated
    '''
    y_true_onehot = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_onehot = K.cast_to_floatx(K.one_hot(y_true_onehot, y_pred.shape[-1]))
    y_pred = K.cast_to_floatx(K.clip(y_pred, K.epsilon(), 1.0-K.epsilon()))
    ce = -1 * y_true_onehot * K.log(y_pred)
    ce = tf.reduce_sum(ce, axis=-1)
    return tf.reduce_mean(ce)

def bce(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
    '''
    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = K.clip(y_pred[:,:,:,-1], K.epsilon(), 1.0-K.epsilon())
    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    return tf.reduce_mean(ce)

def wbce(y_true, y_pred, w=None):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
        w: weight on the positive pixels
    '''
    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = K.clip(y_pred[:,:,:,-1], K.epsilon(), 1.0-K.epsilon())
    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    if w is None:
        w = tf.cast(y_true>0, y_pred.dtype)
        w = tf.math.reduce_sum(w, axis=[1,2], keepdims=True) + 1
        w = (y_true.shape[1]*y_true.shape[2] - w)/w
    w = tf.cast(y_true > 0, y_pred.dtype) * w + tf.cast(y_true == 0, y_pred.dtype)
    w = tf.stop_gradient(w)
    ce = ce * w
    return tf.reduce_mean(ce)

def bbce(y_true, y_pred, w=None):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
        w: weight on the positive pixels
    '''
    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = K.clip(y_pred[:,:,:,-1], K.epsilon(), 1.0-K.epsilon())
    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    if w is None or w > 1:
        w = tf.cast(y_true>0, y_pred.dtype)
        w = tf.math.reduce_sum(w, axis=[1,2], keepdims=True) + 1
        w = w/(y_true.shape[1]*y_true.shape[2])
    w = tf.cast(y_true > 0, y_pred.dtype) * (1-w) + tf.cast(y_true == 0, y_pred.dtype) * w
    w = tf.stop_gradient(w)
    ce = ce * w
    return tf.reduce_mean(ce)

###################
#### dice loss ####
###################

def dice_square(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
    '''

    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = y_pred[:,:,:,-1]

    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true ** 2, axis=[1, 2]) + tf.reduce_sum(y_pred ** 2, axis=[1, 2])

    dice_loss = 1 - 2 * numerator / (denominator + + K.epsilon())
    return tf.reduce_mean(dice_loss)

def dice_union(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
    '''

    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = y_pred[:,:,:,-1]

    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])

    dice_loss = 1 - (2 * numerator + 1) / (denominator + 1)
    return tf.reduce_mean(dice_loss)


def object_dice(y_true, y_pred, beta=0.9):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
    '''

    y_true = y_true[:,:,:,-1]
    y_pred = y_pred[:,:,:,-1]

    losses = []
    for i in range(y_true.shape[0]):
        obj_loss = []
        yy_true = tf.cast(y_true[i], tf.int32)
        yy_pred = y_pred[i]
        labels, _ = tf.unique(tf.reshape(yy_true, [-1]))
        for l in labels:
            # if l == 0:
            #     continue
            obj = tf.cast(yy_true==l, yy_pred.dtype)
            occupy = tf.reduce_sum(obj*yy_pred)/(tf.reduce_sum(obj)+ K.epsilon())
            obj_loss.append(1-occupy)
        loss = tf.reduce_mean(obj_loss)
        losses.append(loss)

    return tf.reduce_mean(losses)

def gdice(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x C, 'softmax' activated
    '''
    y_true_onehot = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_onehot = K.cast_to_floatx(K.one_hot(y_true_onehot, y_pred.shape[-1]))
    y_pred = K.cast_to_floatx(y_pred)

    w = tf.reduce_sum(y_true_onehot, axis=[1, 2])
    # w = 1 / (w + 1)
    w = 1 / (w ** 2 + K.epsilon())
    w = tf.stop_gradient(w)

    numerator = tf.reduce_sum(y_true_onehot * y_pred, axis=[1, 2])
    numerator = w * numerator

    denominator = tf.reduce_sum(y_true_onehot + y_pred, axis=[1, 2])
    denominator = w * denominator

    dice_loss = 1 - (2 * tf.reduce_sum(numerator, axis=1) + 1)/ (tf.reduce_sum(denominator, axis=1) + 1)
    return tf.reduce_mean(dice_loss)

def sensitivity_specificity_loss(y_true, y_pred, beta=1):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
            or B x H x W x 2, softmax activated
    '''

    y_true = tf.cast(y_true[:,:,:,-1], y_pred.dtype)
    y_pred = y_pred[:,:,:,-1]

    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    fn = tf.reduce_sum(y_true * (1-y_pred), axis=[1, 2])
    fp = tf.reduce_sum((1-y_true) * y_pred, axis=[1, 2])

    beta = beta ** 2
    loss = 1 - ((1+beta)*tp)/((1+beta)*tp + beta*fn + fp)
    return tf.reduce_mean(loss)


####################
#### focal loss ####
####################

def focal_loss(y_true, y_pred, gamma=2.0):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x C, 'softmax' activated
    '''
    y_true_onehot = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_onehot = K.cast_to_floatx(K.one_hot(y_true_onehot, y_pred.shape[-1]))
    y_pred = K.cast_to_floatx(K.clip(y_pred, K.epsilon(), 1.0-K.epsilon()))
    # cross entropy
    ce = -1 * y_true_onehot * K.log(y_pred)
    # weight
    weight = K.pow((1-y_pred), gamma) * y_true_onehot
    # compute the focal loss
    ce = tf.reduce_sum(weight * ce, axis=-1)
    return tf.reduce_mean(ce)

def binary_focal_loss(y_true, y_pred, gamma=2.0):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1, 'sigmoid' activated
    '''
    y_true = K.cast_to_floatx(y_true)
    y_pred = K.cast_to_floatx(K.clip(y_pred, K.epsilon(), 1.0-K.epsilon()))
    
    L = -1 * y_true * ((1 - y_pred) ** gamma) * K.log(y_pred) - (1-y_true) * (y_pred ** gamma) * K.log(1-y_pred)
    
    return tf.reduce_mean(L)

def mse(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x 1
    '''
    y_true = tf.cast(y_true * 10.0, y_pred.dtype)
    # y_true = tf.cast(y_true, y_pred.dtype)
    mse = tf.square(y_pred - y_true)
    # weights = tf.cast(y_true>0, y_pred.dtype) + tf.cast(y_true==0, y_pred.dtype) * neg_weight
    # mse = tf.reduce_sum(mse*weights)/tf.reduce_sum(weights)
    mse = tf.reduce_mean(mse)
    return mse

def cosine_embedding_loss(y_true, y_pred, adj_indicator, max_obj, include_background=True):

    '''
    Args:
        adj_indicator: bool matrix, representing the adjacent relationship, B x InstNum x InstNum
        y_true: label map of size B x H x W x 1
        y_pred: pixel embedding of size B x H x W x C
    '''

    y_pred = tf.math.l2_normalize(y_pred, axis=-1, name='embedding_normalization')
    y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
    adj_indicator = tf.cast(adj_indicator, tf.int32)

    def _loss(x):
        label, adj, pred = x[0], x[1], x[2]
        # flatten the tensors
        label_flat = tf.reshape(label, [-1])
        pred_flat = tf.reshape(pred, [-1, tf.shape(pred)[-1]])

        # if not include background, mask out background pixels
        if not include_background:
            ind = tf.greater(label_flat, 0)
            label_flat = tf.boolean_mask(label_flat, ind)
            pred_flat = tf.boolean_mask(pred_flat, ind)

        # grouping labels
        unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
        counts = tf.reshape(K.cast_to_floatx(counts), (-1, 1))
        instance_num = tf.size(unique_labels, out_type=tf.int32)
        label_depth = instance_num if include_background else instance_num + 1
        # compute mean embedding of each instance
        segmented_sum = tf.math.unsorted_segment_sum(pred_flat, unique_id, instance_num)
        counts = tf.stop_gradient(counts)
        mu = tf.nn.l2_normalize(segmented_sum/counts, axis=1)
        # compute adjacent matrix is too slow, pre-computer before training starts
        inter_mask = (1 - tf.eye(max_obj, dtype=tf.int32)) * tf.cast(adj, tf.int32)

        ##########################
        #### inner class loss ####
        ##########################
        
        mu_expand = tf.gather(mu, unique_id)
        loss_inner = 1 - tf.reduce_sum(mu_expand * pred_flat, axis=-1)
        loss_inner = tf.reduce_mean(loss_inner)

        ##########################
        #### inter class loss ####
        ##########################

        # get inter loss for each pair
        mu_interleave = tf.tile(mu, [instance_num, 1])
        mu_rep = tf.reshape(tf.tile(mu, [1, instance_num]), (instance_num*instance_num, -1))
        loss_inter = tf.abs(tf.reduce_sum(mu_interleave * mu_rep, axis=-1))
        # apply inter loss mask
        inter_mask = tf.gather(inter_mask, unique_labels, axis=0)
        inter_mask = tf.gather(inter_mask, unique_labels, axis=1)
        inter_mask = K.cast_to_floatx(tf.reshape(inter_mask, [-1]))
        loss_inter = tf.reduce_sum(loss_inter*inter_mask)/(tf.reduce_sum(inter_mask)+K.epsilon())

        return 0, 0, loss_inner + loss_inter

    losses = tf.map_fn(_loss, (y_true, adj_indicator, y_pred))[2]
    return tf.reduce_mean(losses)

# def l1_embedding_loss(y_true, y_pred, adj_indicator, max_obj, include_background=True):

#     '''
#     Args:
#         adj_indicator: bool matrix, representing the adjacent relationship, B x InstNum x InstNum
#         y_true: label map of size B x H x W x 1
#         y_pred: pixel embedding of size B x H x W x C
#     '''

#     y_pred = tf.nn.softmax(y_pred, axis=-1, name='embedding_normalization')
#     y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
#     adj_indicator = tf.cast(adj_indicator, tf.int32)

#     def _loss(x):
#         label, adj, pred = x[0], x[1], x[2]
#         # flatten the tensors
#         label_flat = tf.reshape(label, [-1])
#         pred_flat = tf.reshape(pred, [-1, tf.shape(pred)[-1]])

#         # if not include background, mask out background pixels
#         if not include_background:
#             ind = tf.greater(label_flat, 0)
#             label_flat = tf.boolean_mask(label_flat, ind)
#             pred_flat = tf.boolean_mask(pred_flat, ind)

#         # grouping labels
#         unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
#         counts = tf.reshape(K.cast_to_floatx(counts), (-1, 1))
#         instance_num = tf.size(unique_labels, out_type=tf.int32)
#         label_depth = instance_num if include_background else instance_num + 1
#         # compute mean embedding of each instance
#         segmented_sum = tf.math.unsorted_segment_sum(pred_flat, unique_id, instance_num)
#         counts = tf.stop_gradient(counts)
#         # mu = tf.nn.l2_normalize(segmented_sum/counts, axis=1)
#         mu = tf.nn.softmax(segmented_sum/counts, axis=-1)
#         # compute adjacent matrix is too slow, pre-computer before training starts
#         inter_mask = (1 - tf.eye(max_obj, dtype=tf.int32)) * tf.cast(adj, tf.int32)

#         ##########################
#         #### inner class loss ####
#         ##########################
        
#         mu_expand = tf.gather(mu, unique_id)
#         # loss_inner = 1 - tf.reduce_sum(mu_expand * pred_flat, axis=-1)
#         loss_inner = tf.norm(mu_expand - pred_flat, ord=1, axis=-1)
#         loss_inner = tf.reduce_mean(loss_inner)

#         ##########################
#         #### inter class loss ####
#         ##########################

#         # get inter loss for each pair
#         mu_interleave = tf.tile(mu, [instance_num, 1])
#         mu_rep = tf.reshape(tf.tile(mu, [1, instance_num]), (instance_num*instance_num, -1))
#         # loss_inter = tf.abs(tf.reduce_sum(mu_interleave * mu_rep, axis=-1))
#         loss_inter = 1 - tf.norm(mu_interleave - mu_rep, ord=1, axis=-1)
#         # apply inter loss mask
#         inter_mask = tf.gather(inter_mask, unique_labels, axis=0)
#         inter_mask = tf.gather(inter_mask, unique_labels, axis=1)
#         inter_mask = K.cast_to_floatx(tf.reshape(inter_mask, [-1]))
#         loss_inter = tf.reduce_sum(loss_inter*inter_mask)/(tf.reduce_sum(inter_mask)+K.epsilon())

#         return 0, 0, loss_inner + loss_inter

#     losses = tf.map_fn(_loss, (y_true, adj_indicator, y_pred))[2]
#     return tf.reduce_mean(losses)

# def argmax_embedding_loss(y_true, y_pred, adj_indicator, max_obj, include_background=True):

#     '''
#     Args:
#         adj_indicator: bool matrix, representing the adjacent relationship, B x InstNum x InstNum
#         y_true: label map of size B x H x W x 1
#         y_pred: pixel embedding of size B x H x W x C
#     '''

#     y_pred = tf.nn.softmax(y_pred, axis=-1, name='embedding_normalization')
#     y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
#     adj_indicator = tf.cast(adj_indicator, tf.int32)

#     def _loss(x):
#         label, adj, pred = x[0], x[1], x[2]
#         # flatten the tensors
#         label_flat = tf.reshape(label, [-1])
#         pred_flat = tf.reshape(pred, [-1, tf.shape(pred)[-1]])

#         # if not include background, mask out background pixels
#         if not include_background:
#             ind = tf.greater(label_flat, 0)
#             label_flat = tf.boolean_mask(label_flat, ind)
#             pred_flat = tf.boolean_mask(pred_flat, ind)

#         # grouping labels
#         unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
#         counts = tf.reshape(K.cast_to_floatx(counts), (-1, 1))
#         instance_num = tf.size(unique_labels, out_type=tf.int32)
#         label_depth = instance_num if include_background else instance_num + 1
#         # compute mean embedding of each instance
#         segmented_sum = tf.math.unsorted_segment_sum(pred_flat, unique_id, instance_num)
#         counts = tf.stop_gradient(counts)
#         mu = tf.nn.l2_normalize(segmented_sum/counts, axis=1)
#         # compute adjacent matrix is too slow, pre-computer before training starts
#         # label_stack = tf.one_hot(label, depth=label_depth, dtype=tf.int32)
#         # label_dilation = tf.nn.dilation2d(tf.expand_dims(label_stack, axis=0), 
#         #                                   disk_tf(neighbor_distance, label_depth), 
#         #                                   strides=[1 ,1, 1, 1], 
#         #                                   padding='SAME', data_format='NHWC', 
#         #                                   dilations=[1,1,1,1]) - 1
#         # neighbor = tf.transpose(label_dilation[0] * tf.expand_dims(label, axis=-1), perm=[2, 0, 1])
#         # def _indicator(x):
#         #     u, _ = tf.unique(tf.reshape(x, [-1]))
#         #     indicator = tf.reduce_max(tf.one_hot(u, depth=label_depth, dtype=tf.int32), axis=0)
#         #     return indicator
#         # v = tf.map_fn(_indicator, neighbor)
#         inter_mask = (1 - tf.eye(max_obj, dtype=tf.int32)) * tf.cast(adj, tf.int32)

#         ##########################
#         #### inner class loss ####
#         ##########################
        
#         mu_expand = tf.gather(mu, unique_id)
#         # mu_inner = tf.stop_gradient(mu_expand)
#         # mu_inner = tf.identity(mu_expand)
#         loss_inner = 1 - tf.reduce_sum(mu_expand * pred_flat, axis=-1)
#         loss_inner = tf.reduce_mean(loss_inner)
#         # loss_inner = tf.where(tf.math.is_nan(loss_inner), 0.0, loss_inner)

#         ##########################
#         #### inter class loss ####
#         ##########################

#         # get inter loss for each pair
#         mu_interleave = tf.tile(mu, [instance_num, 1])
#         mu_rep = tf.reshape(tf.tile(mu, [1, instance_num]), (instance_num*instance_num, -1))
#         loss_inter = tf.abs(tf.reduce_sum(mu_interleave * mu_rep, axis=-1))
#         # apply inter loss mask
#         inter_mask = tf.gather(inter_mask, unique_labels, axis=0)
#         inter_mask = tf.gather(inter_mask, unique_labels, axis=1)
#         inter_mask = K.cast_to_floatx(tf.reshape(inter_mask, [-1]))
#         loss_inter = tf.reduce_sum(loss_inter*inter_mask)/(tf.reduce_sum(inter_mask)+K.epsilon())

#         return 0, 0, loss_inner + loss_inter

#     losses = tf.map_fn(_loss, (y_true, adj_indicator, y_pred))[2]
#     return tf.reduce_mean(losses)

# def dice_embedding_loss(y_true, y_pred, adj_indicator, max_obj, margin=20):

#     '''
#     Args:
#         adj_indicator: bool matrix, representing the adjacent relationship, B x InstNum x InstNum
#         y_true: label map of size B x H x W x 1
#         y_pred: pixel embedding of size B x H x W x C
#     '''

#     y_pred = tf.nn.softmax(y_pred, axis=-1, name='embedding_normalization')
#     y_true = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
#     adj_indicator = tf.cast(adj_indicator, tf.int32)

#     def _loss(x):
#         pred, label = x[0], x[1]
#         # flatten the tensors
#         label_flat = tf.reshape(label, [-1])
#         pred_flat = tf.reshape(pred, [-1, tf.shape(pred)[-1]])

#         # grouping labels
#         unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)

#         label_depth = tf.size(unique_labels, out_type=tf.int32)
#         label_stack = tf.one_hot(label, depth=label_depth, dtype=tf.uint8)
#         label_dilated = tf.nn.dilation2d(tf.expand_dims(label_stack, axis=0), 
#                                          disk_tf(margin, label_depth, dtype=tf.uint8), 
#                                          strides=[1 ,1, 1, 1], 
#                                          padding='SAME', data_format='NHWC', 
#                                          dilations=[1,1,1,1]) - 1
#         loss = 0
#         for l in range(len(label_stack)):
#             pass


#         return 0, loss

#     losses = tf.map_fn(_loss, (y_pred, y_label))[2]
#     return tf.reduce_mean(losses)



if __name__ == '__main__':
    import numpy as np

    # test segmentation loss
    # label = 2
    # pred = np.random.uniform(0, 1, (4,10,10,label))
    # gt = np.floor(np.random.uniform(0, 1, (4,10,10,1)) * label)
    # # wbce(gt, pred)
    # dice(gt, pred)

    # # test dist_loss
    # L = dist_loss('mae')
    # gt = np.zeros((1,10,10))
    # gt[0,0:5,0:5] = 1
    # gt[0,2:7,5:8] = 2
    # m = edt(gt, normalize=True)
    # print(gt[0,:,:], m[0,:,:], m.shape)
    # pred = np.zeros((1,10,10,1)).astype(np.float32)
    # print(L(gt, pred))

    # # test seed_loss
    # L = seed_loss('binary_focal_loss')
    # gt = np.zeros((1, 5, 5))
    # gt[0, 0:2, 0:2] = 1
    # pred = np.zeros((1, 5, 5, 1))
    # pred[0, 0:2, 0:1, 0] = 1
    # print(L(gt, pred))


    # f = focal_loss()
    # pred = np.random.rand(1,2,2,2)
    # gt = np.array([[[1,0],[0,0]]])
    # print(pred, gt)
    # print(f(gt, pred))

    # f = cosine_embedding_loss(neighbor_distance=3, include_background=False)
    # pred = np.random.rand(2,10,10,2).astype(np.float32)
    # gt = np.zeros((2,10,10,1)).astype(np.int32)
    # gt[0,0:2, 0:2,:] = 1
    # gt[0,3:5, 0:2,:] = 2
    # gt[0,8:, 8:,:] = 3
    # loss = f(gt, pred)
    # print(loss)

    a = np.zeros((10,10), np.int32)
    a[0:5, 0:5] = 1
    a[6:7, 6:7] = 2
    d = np.squeeze(np.array(test(a)))
    print(d[:,:,0], d[:,:,1], d[:,:,2])
    
