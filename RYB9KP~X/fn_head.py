import tensorflow as tf
from fn_backbone import build_d9, build_Unet

def build_embedding_head(features, embedding_dim=32, name='embedding_branch'):
    with tf.variable_scope(name):
        features = tf.keras.layers.Dropout(rate=0.5)(features)
        emb = tf.keras.layers.Conv2D(embedding_dim, 3, activation='linear',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)
        emb = tf.nn.l2_normalize(emb, axis=-1)
    return emb


def build_dist_head(features, embedding_dim=32, name='dist_regression'):
    with tf.variable_scope(name):
        features = tf.keras.layers.Dropout(rate=0.5)(features)
        f2 = tf.keras.layers.Conv2D(embedding_dim, 3, activation='linear',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(features)
        dist = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(f2)
    return dist, f2



# dist -> concatenate image 
def build_dist_concat(image, feature_dist, embedding_dim=32, name='distemb_concat'):
    with tf.variable_scope(name):
        feature_dist = tf.keras.layers.Dropout(rate=0.5)(feature_dist)


        ######################### 
        ## 1.U-Net output dim  ##
        ######################### 
        feature_dist = tf.keras.layers.Conv2D(embedding_dim, 3, activation='linear',padding='same', use_bias=True, kernel_initializer='he_normal', bias_initializer='zeros')(feature_dist)

        ######################### 
        ##   Concatenate Type  ##
        ######################### 
        inputs = tf.keras.layers.concatenate([image, tf.nn.l2_normalize(feature_dist, axis=-1)], axis=3)
        f1= build_Unet(inputs)
    return build_embedding_head(f1, embedding_dim=embedding_dim)





def build_dist_embedding_concat(image, feature_dist, embedding_dim=16, name='dist_embedding_branch'):
    with tf.variable_scope(name):
        inputs = tf.keras.layers.concatenate([image, tf.nn.l2_normalize(feature_dist, axis=-1)], axis=3)
        f1, _ = build_d9(inputs, features=32, drop_rate=0.2, name="doubleHead_UNet")
    return build_embedding_head(f1, embedding_dim=16)

