from tensorflow.keras.layers import Conv2D, BatchNormalization, LayerNormalization, ConvLSTM2D, Conv2DTranspose, Conv3DTranspose
import tensorflow as tf
import numpy as np
from skimage.measure import regionprops

def rp(model, pt, input_sz=(1,512,512,1), default_set=False):

    inputs = model.input
    outputs = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if default_set:
        for module in model.layers:
            if isinstance(module, Conv2D):
                conv_weights = np.full(module.get_weights()[0].shape, 1)
                if len(module.get_weights()) > 1:
                    conv_biases = np.full(module.get_weights()[1].shape, 0.0)
                    module.set_weights([conv_weights, conv_biases])
                else:
                    module.set_weights([conv_weights])
            if isinstance(module, Conv2DTranspose):
                conv_weights = np.full(module.get_weights()[0].shape, 1)
                if len(module.get_weights()) > 1:
                    conv_biases = np.full(module.get_weights()[1].shape, 0.0)
                    module.set_weights([conv_weights, conv_biases])
                else:
                    module.set_weights([conv_weights])
            if isinstance(module, BatchNormalization):
                bn_weights = [module.get_weights()[0], module.get_weights()[1], np.full(module.get_weights()[2].shape, 0.0), np.full(module.get_weights()[3].shape, 1.0),]
                module.set_weights(bn_weights)

    input_img = tf.ones(input_sz)
    # input_img = tf.convert_to_tensor(inputs)

    with tf.GradientTape() as tf_gradient_tape:
        tf_gradient_tape.watch(input_img)
        
        output = model(input_img)
        mask = np.copy(output) * 0
        mask[0, pt[0], pt[1], :] = 1
        
        pseudo_loss = tf.reduce_mean(mask * output)
        grad = tf_gradient_tape.gradient(pseudo_loss, input_img)

    grad = np.squeeze(grad)
    H = np.sum(np.sum(grad != 0, axis=1) > 0)
    W = np.sum(np.sum(grad != 0, axis=0) > 0)
    return (H, W), grad 


# def embedding_space(embedding, instance_mask, neighbourhood=32, mode="cos"):

#     obj_emb = {}

#     homogeneity, separability = [], []

#     for r in regionprops(instance_mask):
#         if r.label == 0:
#             continue
#         rr, cc =  r.coords[:,0], r.coords[:,1]
#         emb = np.mean(embedding[rr,cc], axis=0, keepdims=True)
#         obj_emb[r.label] = emb

#         if mode == 'cos':


        

#     embedding = np.squeeze(embedding)
#     mask = np.squeeze(instance_mask)

#     embedding_flat = 
    
    
#     embedding_flat = 
#     # homogeneity, separability
#     values = np.array([1,2,3,4,5])
#     indexes = np.array([0,0,1,1,2])

#     one_hot = np.eye(np.max(indexes) + 1)[indexes]

#     counts = np.sum(one_hot, axis=0)
#     average = np.sum((one_hot.T * values), axis=1) / counts

#     print(average) # [1.5 3.5 5.]    
#     pass
