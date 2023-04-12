from math import gamma
import tensorflow as tf
import tensorflow.keras.backend as K 
import numpy as np
# from instSeg.constant import *

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
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C, 'sigmoid' activated
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    return tf.reduce_mean(ce)

def bce_random(y_true, y_pred, N=2):
    '''
    Args:
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C, 'sigmoid' activated
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # generate mask, which randomly cover part of negative pixels
    c_pos = tf.cast(tf.math.count_nonzero(y_true), y_pred.dtype)
    c_neg = tf.math.minimum(tf.cast(c_pos * N, tf.float32), tf.size(y_true, tf.float32))
    mask = tf.random.uniform(tf.shape(y_true))
    mask = mask * tf.cast(y_true == 0, mask.dtype)
    values, _ = tf.math.top_k(mask, tf.cast(c_neg, tf.int32))
    mask =tf.math.logical_or(tf.greater_equal(mask, tf.reduce_min(values)), y_true == 1)
    mask = tf.cast(mask, y_pred.dtype)
    mask = tf.stop_gradient(mask)

    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    ce = ce * mask
    return tf.reduce_mean(ce)

def bce_hard(y_true, y_pred, N=2):
    '''
    Args:
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C, 'sigmoid' activated
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # generate mask, which randomly cover part of negative pixels
    c_pos = tf.cast(tf.math.count_nonzero(y_true), y_pred.dtype)
    c_neg = tf.math.minimum(tf.cast(c_pos * N, tf.float32), tf.size(y_true, tf.float32))
    mask = y_pred * tf.cast(y_true == 0, y_pred.dtype)
    values, _ = tf.math.top_k(mask, tf.cast(c_neg, tf.int32))
    mask =tf.math.logical_or(tf.greater_equal(mask, tf.reduce_min(values)), y_true == 1)
    mask = tf.cast(mask, y_pred.dtype)
    mask = tf.stop_gradient(mask)

    ce = -1 * y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    ce = ce * mask
    return tf.reduce_mean(ce)

def bce_weighted(y_true, y_pred, mode='inverse_frequency'):
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
    if mode == 'inverse_frequency':
        c_pos, c_neg = tf.cast(tf.math.count_nonzero(y_true), y_pred.dtype),  tf.cast(tf.math.count_nonzero(y_true==0), y_pred.dtype)
        w_pos = c_neg/( c_pos + c_neg)
        w_neg = c_pos/( c_pos + c_neg)
    w = tf.cast(y_true > 0, y_pred.dtype) * w_pos + tf.cast(y_true == 0, y_pred.dtype) * w_neg
    w = tf.stop_gradient(w)
    ce = tf.reduce_sum(ce * w)/tf.reduce_sum(w)
    return ce


####################
#### focal loss ####
####################

def fl(y_true, y_pred, gamma=2.0):
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

def bfl(y_true, y_pred, gamma=2.0):
    '''
    Args:
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C, 'sigmoid' activated
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    ce = -1 * y_true *  K.log(y_pred) - (1-y_true) * K.log(1-y_pred)
    w = y_true * ((1 - y_pred) ** gamma) + (1-y_true) * (y_pred ** gamma)
    w = tf.stop_gradient(w)
    ce = ce * w
    return tf.reduce_mean(ce)


###################
#### dice loss ####
###################


def dice(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C, 'sigmoid' activated
    usually C=1, each channel is equally considered
    '''
    y_true = tf.cast(y_true, y_pred.dtype)

    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])

    dice_loss = 1 - (2 * numerator + 1) / (denominator + 1)
    return tf.reduce_mean(dice_loss)

def mdice(y_true, y_pred):
    '''
    multi-class dice, each class is equally considered
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x C, 'softmax' activated
    '''
    y_true_onehot = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_onehot = K.cast_to_floatx(K.one_hot(y_true_onehot, y_pred.shape[-1]))
    y_pred = K.cast_to_floatx(y_pred)

    numerator = tf.reduce_sum(y_true_onehot * y_pred, axis=[1, 2]) 
    denominator = tf.reduce_sum(y_true_onehot + y_pred, axis=[1, 2])

    dice_loss = 1 - (2 * numerator + 1) / (denominator + 1 )
    return tf.reduce_mean(dice_loss)

def gdice(y_true, y_pred):
    '''
    generalised dice for multi-class prediction, each class is weighted by the reversed area
    Args:
        y_true: label map of size B x H x W x 1
        y_pred: feature map of size B x H x W x C, 'softmax' activated
    '''
    y_true_onehot = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_true_onehot = K.cast_to_floatx(K.one_hot(y_true_onehot, y_pred.shape[-1]))
    y_pred = K.cast_to_floatx(y_pred)

    w = tf.reduce_sum(y_true_onehot, axis=[1, 2])
    w = 1 / (w ** 2 + K.epsilon())
    w = tf.stop_gradient(w)

    numerator = tf.reduce_sum(y_true_onehot * y_pred, axis=[1, 2])
    numerator = w * numerator

    denominator = tf.reduce_sum(y_true_onehot + y_pred, axis=[1, 2])
    denominator = w * denominator

    dice_loss = 1 - (2 * tf.reduce_sum(numerator, axis=1) + 1)/ (tf.reduce_sum(denominator, axis=1) + 1)
    return tf.reduce_mean(dice_loss)

##################
#### mse loss ####
##################

def mse(y_true, y_pred):
    '''
    Args:
        y_true: label map of size B x H x W x N
        y_pred: feature map of size B x H x W x N
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    mse = tf.square(y_pred - y_true)
    mse = tf.reduce_mean(mse)
    return mse


def masked_mse(y_true, y_pred, mask):
    '''
    Args:
        y_true: label map of size B x H x W x C
        y_pred: feature map of size B x H x W x C
        mask: size B x H x W x 1
    '''
    y_true = tf.cast(y_true, y_pred.dtype)
    mse = tf.square(y_pred - y_true)
    mse = tf.reduce_mean(mse, axis=-1, keepdims=True)
    mse = mse[mask>0]
    mse = tf.reduce_mean(mse)
    return mse

######################################
#### sensitivity-specifivity loss ####
######################################

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

###################
#### embedding ####
###################

def stack2map(y_true):
    '''
    convert a stack of N mask 1 x H x W x N into a single map 1 x H x W x 1, in which different objects are indexed by different number 
        - if y_true is already a map, not affected
        - overlapped region ignores
    '''
    if tf.shape(y_true)[-1] > 1:
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.cast(y_true > 0, tf.int32)
        S = tf.reduce_sum(y_true, axis=-1, keepdims=True)
        non_overlap =  tf.cast(S == 1, tf.int32)
        # overlap = tf.cast(S > 1, tf.int32)
        y_true = tf.reduce_sum(y_true * (tf.range(tf.shape(y_true)[-1], dtype=tf.int32)[tf.newaxis,tf.newaxis,tf.newaxis,:]+1), axis=-1, keepdims=True)
        y_true = y_true * non_overlap

    return y_true

def map2stack(y_true):
    if tf.shape(y_true)[-1] == 1:
        y_true = tf.squeeze(tf.one_hot(y_true, depth=tf.reduce_max(y_true)+1), axis=-2)
        y_true = tf.stack(tf.unstack(y_true, axis=-1)[1:], axis=-1)
    return y_true

def instance_contrastive(label, adj, pred, include_background=False, mode='cosine', margin_attr=0, margin_rep=0):

    '''
    pred should be normalized before input, if mode == 'cosine'
    '''
    
    # flatten the tensors
    label_flat = tf.reshape(label, [-1])
    pred_flat = tf.reshape(pred, [-1, tf.shape(pred)[-1]])

    # if not include background, mask out background pixels
    if not include_background:
        ind = tf.greater(label_flat, 0)
        label_flat = tf.boolean_mask(label_flat, ind)
        pred_flat = tf.boolean_mask(pred_flat, ind)

    if tf.equal(tf.size(label_flat), 0):
        return 0, 0


    # grouping labels
    unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
    counts = tf.reshape(K.cast_to_floatx(counts), (-1, 1))
    instance_num = tf.size(unique_labels, out_type=tf.int32)
    # label_num = instance_num if include_background else instance_num + 1
    # compute mean embedding of each instance
    segmented_sum = tf.math.unsorted_segment_sum(pred_flat, unique_id, instance_num)
    counts = tf.cast(tf.stop_gradient(counts), segmented_sum.dtype)
    mu = segmented_sum/counts
    if mode == 'cosine':
        mu = tf.nn.l2_normalize(mu, axis=1)
    # mu = segmented_sum
    # compute adjacent matrix is too slow, pre-computer before training starts
    # inter_mask = (1 - tf.eye(max_obj, dtype=tf.int32)) * tf.cast(adj, tf.int32)
    inter_mask = tf.linalg.set_diag(adj, tf.zeros((tf.shape(adj)[0]), dtype=adj.dtype))
    inter_mask = tf.cast(inter_mask, tf.int32)

    ##########################
    #### inner class loss ####
    ##########################
    mu_expand = tf.gather(mu, unique_id)
    # loss_attr = 1 - tf.abs(tf.reduce_sum(mu_expand * pred_flat, axis=-1))
    if mode == 'cosine':
        loss_attr = 1 - tf.reduce_sum(mu_expand * pred_flat, axis=-1)
    elif mode == 'euclidean':
        loss_attr = tf.sqrt(tf.reduce_sum((mu_expand - pred_flat) ** 2, axis=-1))
    if margin_attr != 0:
        loss_attr = tf.math.maximum(loss_attr - margin_attr, 0)
    loss_attr = loss_attr * tf.squeeze(1 / (tf.gather(counts, unique_id) + 1e-12))
    loss_attr = tf.math.unsorted_segment_sum(loss_attr, unique_id, instance_num)
    loss_attr = tf.reduce_mean(loss_attr) 
    # weights = weights / (tf.reduce_sum(weights) + 1e-12)
    # loss_attr = tf.reduce_sum(loss_attr*weights)

    ##########################
    #### inter class loss ####
    ##########################

    # get inter loss for each pair
    mu_interleave = tf.tile(mu, [instance_num, 1])
    mu_rep = tf.reshape(tf.tile(mu, [1, instance_num]), (instance_num*instance_num, -1))
    if mode == 'cosine':
        loss_rep = tf.reduce_sum(mu_interleave * mu_rep, axis=-1) ** 2
    elif mode == 'euclidean':
        margin_rep = -margin_rep if margin_rep > 0 else margin_rep
        loss_rep = - tf.sqrt(tf.reduce_sum((mu_interleave - mu_rep) ** 2, axis=-1))
    if margin_rep != 0:
        loss_rep = tf.math.maximum(loss_rep - margin_rep, 0)
    # loss_rep = tf.abs(tf.reduce_sum(mu_interleave * mu_rep, axis=-1))
    # apply inter loss mask
    inter_mask = tf.gather(inter_mask, unique_labels, axis=0)
    inter_mask = tf.gather(inter_mask, unique_labels, axis=1)
    inter_mask = tf.cast(tf.reshape(inter_mask, [-1]), loss_rep.dtype)
    loss_rep = tf.reduce_sum(loss_rep*inter_mask)/(tf.reduce_sum(inter_mask)+K.epsilon())

    return loss_attr, loss_rep


def embedding_loss(y_true, y_pred, adj_indicator, config, mode='cosine'):

    '''
    Args:
        adj_indicator: bool matrix, representing the adjacent relationship, B x InstNum x InstNum
        y_true: label map of size B x H x W x 1 or stack of masks B x H x W x N 
        y_pred: pixel embedding of size B x H x W x C
    '''

    y_true = stack2map(y_true)
    y_true = tf.squeeze(y_true, axis=-1)

    if mode == 'cosine':
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    adj_indicator = tf.cast(adj_indicator, tf.int32)

    def _loss(x):
        label, adj, pred = x[0], x[1], x[2]
        loss_attr, loss_rep = instance_contrastive(label, adj, pred, config.embedding_include_bg, mode=mode, margin_attr=config.margin_attr, margin_rep=config.margin_rep)
        if config.dynamic_weighting:
            w = loss_rep/(loss_attr + 1e-7)
            w = tf.cast(tf.stop_gradient(w), loss_rep.dtype)
            return 0, 0, (loss_attr + w * loss_rep)/(1+w)
        else:
            return 0, 0, loss_attr + loss_rep

    losses = tf.map_fn(_loss, (y_true, adj_indicator, y_pred))[2]
    losses = tf.reduce_mean(losses) 

    if config.embedding_regularization > 0:
        if mode == 'cosine':
            reg = 1 - tf.reduce_max(y_pred, axis=-1)
        elif mode == 'euclidean':
            reg = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1))
        if not config.embedding_include_bg:
            ind = tf.greater(y_true, 0)
            reg = tf.boolean_mask(reg, ind)
        loss_reg = 0 if tf.equal(tf.size(reg), 0) else tf.reduce_mean(reg)
        losses = losses + config.embedding_regularization * loss_reg

    return losses


# def sparse_cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=False, dynamic_weighting=True):
#     losses = cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=include_background)
#     y_true = stack2map(y_true)
#     y_pred = tf.math.l2_normalize(y_pred, axis=-1) ** 2
#     if not include_background:
#         loss_sparse = tf.reduce_mean((1 - tf.reduce_max(y_pred, axis=-1))*tf.cast(tf.squeeze(y_true, axis=-1)>0, y_pred.dtype))
#     else:
#         loss_sparse = tf.reduce_mean(1 - tf.reduce_max(y_pred, axis=-1))
#     losses = losses + 0.1 * loss_sparse
#     return losses


# def overlap_embedding_loss(y_true, y_pred, adj_indicator, include_background=False):
#     '''
#     y_true: shape N x H x W x #obj***
#     y_pred: shape N x H x W x C
#     '''
#     loss = sparse_cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=include_background)
    
#     y_true = tf.cast(y_true, tf.int32)
#     labeled_objs = stack2map(y_true)

#     def _get_layered(x):
#         embedding, mask, labeled = x[0], x[1], x[2]
#         embedding = tf.math.l2_normalize(embedding, axis=-1)
#         mask = map2stack(mask)

#         mu = tf.math.unsorted_segment_mean(embedding, tf.squeeze(labeled, axis=-1), num_segments=tf.shape(mask)[-1]+1)
#         layer_idx = tf.argmax(mu, axis=1)
#         mask_trim = tf.pad(tf.transpose(mask, perm=[2,0,1]), paddings=tf.constant([[1,0],[0,0],[0,0]]), mode='CONSTANT', constant_values=0)
#         layered = tf.math.unsorted_segment_mean(mask_trim, layer_idx, num_segments=tf.shape(embedding)[-1]) > 0
#         layered = tf.cast(tf.transpose(layered, perm=[1,2,0]), labeled.dtype)
#         return 0, 0, layered
    
#     layered_objs = tf.map_fn(_get_layered, (y_pred, y_true, labeled_objs))[2]

    # overlap = tf.reduce_sum(tf.cast(y_true > 0, tf.int32), axis=-1) > 1
    # if tf.math.reduce_any(overlap):
    #     y_true_overlap = tf.boolean_mask(layered_objs, overlap)
    #     y_pred_overlap = tf.boolean_mask(y_pred, overlap)
    #     y_true_overlap = tf.cast(y_true_overlap, y_pred_overlap.dtype)
    #     y_pred_overlap = K.clip(y_pred_overlap, K.epsilon(), 1.0-K.epsilon())
    #     ce = -1 * y_true_overlap * K.log(y_pred_overlap) - (1 - y_true_overlap) * K.log(1 - y_pred_overlap)
    #     # ce = -1 * y_true_overlap * ((1 - y_pred_overlap) ** 2) * K.log(y_pred_overlap) - (1-y_true_overlap) * (y_pred_overlap ** gamma) * K.log(1-y_pred_overlap)
    #     loss = loss + tf.reduce_mean(ce)
    #     # loss = loss + tf.reduce_sum(ce)
    


    layered_objs = tf.stop_gradient(layered_objs)
    layered_objs = tf.cast(layered_objs, y_pred.dtype)

    # bce loss
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    # gamma = 2
    # loss_overlap = -1 * layered_objs * ((1 - y_pred) ** 2) * K.log(y_pred) - (1-layered_objs) * (y_pred ** gamma) * K.log(1-y_pred)
    # loss_overlap = -1 * layered_objs * K.log(y_pred) - (1 - layered_objs) * K.log(1 - y_pred)
    # loss_overlap = tf.reduce_mean(loss_overlap, axis=-1)
    # if not include_background:
    #     mask = tf.reduce_sum(tf.cast(y_true > 0, tf.int32), axis=-1) > 0
    #     mask = tf.cast(mask, y_pred.dtype)   
    #     loss_overlap = tf.reduce_sum(loss_overlap * mask) / (tf.reduce_sum(mask)+1e-8)
    # else:
    #     loss_overlap = tf.reduce_mean(loss_overlap)
    # dice loss

    if not include_background:
        mask = tf.reduce_sum(tf.cast(y_true > 0, tf.int32), axis=-1, keepdims=True) > 0
        mask = tf.cast(mask, y_pred.dtype)
        layered_objs = layered_objs * mask
        y_pred = y_pred * mask

    numerator = tf.reduce_sum(layered_objs * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(layered_objs + y_pred, axis=[1, 2, 3])

    loss_overlap = 1 - (2 * numerator + 1) / (denominator + 1)
    loss_overlap = tf.reduce_mean(loss_overlap)

    loss = loss + loss_overlap

    return loss

# def overlap_embedding_lossD(y_true, y_pred, adj_indicator):
#     '''
#     y_true: shape N x H x W x #obj, if you use labeled map (without overlp) use ***
#     y_pred: shape N x H x W x 1
#     '''
#     loss = sparse_cosine_embedding_loss(y_true, y_pred, adj_indicator, include_background=False)
    
#     # y_true = tf.cast(tf.cast(y_true, tf.int32)>0, tf.int32)
#     y_true = tf.cast(y_true, tf.int32)
#     labeled_objs = stack2map(y_true)

#     def _loss(x):
#         embedding, mask, labeled = x[0], x[1], x[2]
#         embedding = tf.math.l2_normalize(embedding, axis=-1)
#         if tf.shape(mask)[-1] == 1:
#             mask = tf.squeeze(tf.one_hot(mask, depth=tf.reduce_max(mask)), axis=-2)
#             mask = tf.stack(tf.unstack(mask, axis=-1)[1:], axis=-1)

#         mu = tf.math.unsorted_segment_mean(embedding, tf.squeeze(labeled, axis=-1), num_segments=tf.shape(mask)[-1]+1)
#         layer_idx = tf.argmax(mu, axis=1)
#         mask_trim = tf.pad(tf.transpose(mask, perm=[2,0,1]), paddings=tf.constant([[1,0],[0,0],[0,0]]), mode='CONSTANT', constant_values=0)
#         layered = tf.math.unsorted_segment_mean(mask_trim, layer_idx, num_segments=tf.shape(embedding)[-1]) > 0
#         layered = tf.cast(tf.transpose(layered, perm=[1,2,0]), labeled.dtype)
#         return 0, 0, layered
    
#     layered_objs = tf.map_fn(_loss, (y_pred, y_true, labeled_objs))[2]
#     loss_obj_seg = dice(layered_objs, y_pred)

#     layered_objs = tf.math.reduce_max(layered_objs, axis=-1)
#     y_pred = tf.math.reduce_max(y_pred, axis=-1)
#     loss_overall_seg = dice(layered_objs, y_pred)

#     return loss+loss_obj_seg






if __name__ == '__main__':
    a = np.zeros((1,20,20,3))
    a[0,:10,:10,0] = 1
    a[0,5:15,5:15,2] = 1
    embedding = np.zeros((1,20,20,4))
    embedding[0,:10,:10,1] = 1
    embedding[0,5:15,5:15,3] = 1
    overlap_embedding_loss(a, embedding, [])

    # print(np.sum(a[...,0]), np.sum(a[...,1]), np.sum(a[...,2]))

    # a_m = stack2map(a)
    # print(np.unique(a_m), np.sum(a_m==1), np.sum(a_m==3))

    # a_s = map2stack(a_m)
    # print(a_m.shape, a_s.shape)
    # print(np.sum(a_s[...,0]), np.sum(a_s[...,1]), np.sum(a_s[...,2]))
