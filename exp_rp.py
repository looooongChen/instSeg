import instSeg 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


# Parameters filter 32/64
# stage #             unet             #    cunet residual      #
# 1     #       102,080 / 404,864      #   71,392 / 282,048     #
# 2     #       467,968 / 1,865,728    #  112,832 / 446,848     #
# 3     #     1,928,832 / 7,703,808    #  154,272 / 611,648     #
# 4     #     7,766,912 / 31,045,376   #  195,712 / 776,448     #
# 5     #    31,108,480 / 124,390,144  #  237,152 / 941,248     #
# 6     #   124,453,248 / 497,726,208  #  278,592 / 1,106,048   #

# receptive field and crop
# stage #    unet       #  cunet  #
# 1     #   18   9/8    #       #
# 2     #   44  22/20   #      #
# 3     #   96  48/44   #    #
# 4     #  200 100/92   #     #
# 5     #  408 204/188  #    #
# 6     #  824 412/380  #    #

#### rp and crop ####


sz = 512
input_img = tf.keras.layers.Input((sz,sz,1), name='input_img')
model, fts, props = instSeg.nets.unet(input_img, convs=2, stages=5, filters=32, up_scaling='deConv', padding='same', concatenate=False)
# model, fts, props = instSeg.nets.cunet(input_img, convs=2, stages=4, filters=32, up_scaling='deConv', padding='valid', concatenate=True, residual=True)

model.summary()

print(props)
print(input_img.shape, fts.shape)


# sz = 1024

# def count_parameters(model):
#     trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
#     nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
#     return trainableParams, nonTrainableParams

# pa_summary = pd.DataFrame(columns=['depth(stage)', 'U-Net32', 'cU-Net32', 'U-Net64', 'cU-Net64'])
# rp_summary = pd.DataFrame(columns=['depth(stage)', 'U-Net32', 'cU-Net32', 'U-Net64', 'cU-Net64'])

# for stage in [3,4,5,6]:
#     parameters, rps = [stage], [stage]
#     for filters in [32, 64]: 
        
#         input_img = tf.keras.layers.Input((sz,sz,1), name='input_img')
#         model, fts = instSeg.nets.unet(input_img, filters=filters,
#                                         stages=stage,
#                                         convs=2,
#                                         drop_rate=0,
#                                         normalization='batch',
#                                         up_scaling='upConv',
#                                         concatenate=True)
#         params, _ = count_parameters(model)
#         parameters.append(params)
#         rp, grad = instSeg.model_analyzer.rp(model, pt=(sz//2, sz//2), input_sz=(1,sz,sz,1), default_set=True)
#         rps.append(rp)

#         input_img = tf.keras.layers.Input((sz,sz,1), name='input_img')
#         model, fts = instSeg.nets.cunet(input_img, filters=filters,
#                                         stages=stage,
#                                         convs=2,
#                                         residual=True,
#                                         drop_rate=0,
#                                         normalization='batch',
#                                         up_scaling='up')
#         params, _ = count_parameters(model)
#         parameters.append(params)
#         rp, grad = instSeg.model_analyzer.rp(model, pt=(sz//2, sz//2), input_sz=(1,sz,sz,1))
#         rps.append(rp)

#     pa_summary.loc[len(pa_summary)] = parameters
#     rp_summary.loc[len(rp_summary)] = rps

# # print(pa_summary)
# pa_summary.to_csv('./parameters.csv')
# rp_summary.to_csv('./receptive field.csv')

# Unet
# stage # deConv # upConv #
# 1     # 21     # 23     #
# 2     # 45     # 49     #
# 3     # 93     # 101    #
# 4     # 189    # 205    #
# 5     # 381    # 413    #
# 6     # 765    # 829    #


# config = instSeg.ConfigParallel(image_channel=1)
# config.H = sz
# config.W = sz
# config.nstage = 3
# config.filters = 32
# config.modules = ['layered_embedding', 'foreground']
# config.embedding_dim = 16

# config.backbone = 'unet'
# config.input_normalization = None
# config.up_scaling = 'deConv'
# config.net_normalization = 'batch'
# config.dropout_rate = 0
# config.concatenate = True


# model = instSeg.InstSegParallel(config=config, model_dir='./test')
# inputs = model.model.input
# outputs = model.model(inputs)[0]
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# inputs = tf.keras.layers.Input((sz, sz, 1), name='input_img')
# outputs = Conv2D(8, 3, padding='same', activation='relu')(inputs)
# outputs = Conv2D(8, 3, padding='same', activation='relu')(outputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

# input_img = tf.keras.layers.Input((sz, sz, 1), name='input_img')
# model, fts = instSeg.nets.unet(input_img, filters=32,
#                                         stages=3,
#                                         convs=2,
#                                         drop_rate=0,
#                                         normalization='batch',
#                                         up_scaling='deConv',
#                                         concatenate=True)

# rp, grad = instSeg.analyzer.rp(model, pt=(sz//2, sz//2), input_sz=(1,sz,sz,1), default_set=True)
# print(rp)

# grad = grad/(grad.std())

# plt.imshow(grad)
# plt.show()




