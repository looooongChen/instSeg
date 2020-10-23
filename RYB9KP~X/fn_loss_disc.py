import tensorflow as tf

def discriminative_loss_single( embedding,
                                label_map,
                                neighbor,
                                delta_v, delta_d, # Hyper
                                param_var, param_dist, param_reg, # Hyper
                                include_bg=True):
    """
    build embedding loss
    :param embedding: 3 dim tensor, should be normalized
    :param label_map: 3 dim tensor with 1 channel
    :param neighbor: row N is the neighbors of object N, N starts with 1, 0 indicates the background
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    :param include_bg: weather take background as an independent object
    """

    # flatten the tensors
    label_flat = tf.reshape(label_map, [-1])    # [W*H]
    embedding_flat = tf.reshape(embedding, [-1, tf.shape(embedding)[-1]])   # [W*H, C]
    embedding_flat = tf.nn.l2_normalize(embedding_flat, axis=1) # [W*H, C]
    # weight = weight_map(tf.expand_dims(label_map, axis=0))
    # weight_flat = tf.reshape(weight, [-1, 1])

    # if not include background, mask out background pixels
    if not include_bg:
        label_mask = tf.greater(label_flat, 0)
        label_flat = tf.boolean_mask(label_flat, label_mask)
        embedding_flat = tf.boolean_mask(embedding_flat, label_mask)
        # weight_flat = tf.boolean_mask(weight_flat, label_mask)

    # grouping based on labels
    unique_labels, unique_id, counts = tf.unique_with_counts(label_flat)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)  # K
    segmented_sum = tf.unsorted_segment_sum(embedding_flat, unique_id, num_instances)


    # mean embedding of each instance
    mu = tf.math.divide(segmented_sum, tf.reshape(counts, (-1, 1))) # [K, C]
    mu = tf.nn.l2_normalize(mu, axis=1) # [K, C]
    mu_expand = tf.gather(mu, unique_id)    # [W*H, C]

    ### Calculate l_var
    distance = tf.norm(tf.subtract(mu_expand, embedding_flat), axis=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))


    ### Calculate l_dist
    
    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1]) # [K*K, C]
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances*num_instances, -1))# [K*K, C]

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)# [K*K, C]


    # compute adjacent indicator
    # indicator: bg(0) is adjacent to any object
    # 0 1 1 1 1 ...
    # 1 x x x x ...
    # 1 x x x x ...
    # ...
    bg = tf.zeros([tf.shape(neighbor)[0], 1], dtype=tf.int32) # [K, 1]
    neighbor = tf.concat([bg, neighbor], axis=1) # [K, 1+N]
    dep = num_instances if include_bg else num_instances + 1    #[K+1]

    adj_indicator = tf.one_hot(neighbor, depth=dep, dtype=tf.float32) # [K, 1+N, K+1]
    adj_indicator = tf.reduce_sum(adj_indicator, axis=1)    # [K, K+1]
    adj_indicator = tf.cast(tf.greater(adj_indicator, 0), tf.float32) # [K, K+1]

    bg_indicator = tf.one_hot(0, depth=dep, on_value=0.0, off_value=1.0, dtype=tf.float32)
    bg_indicator = tf.reshape(bg_indicator, [1, -1])    #[1, K+1]
    indicator = tf.concat([bg_indicator, adj_indicator], axis=0) #[1+K, 1+K]

    # reorder the rows and columns in the same order of unique_labels
    # if background (0) is not included, the first row and column will be ignores, since 0 is not the unique_labels
    indicator = tf.gather(indicator, unique_labels, axis=0)
    indicator = tf.gather(indicator, unique_labels, axis=1)
    inter_mask = tf.reshape(indicator, [-1, 1]) #[K*K, 1]

    # Filter out zeros from same cluster subtraction


    mu_norm = tf.norm(mu_diff, axis=1) # [K*K, 1]
    mu_norm = tf.subtract(2.*delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)
    mu_norm = tf.reshape(mu_norm,[-1, 1]) # [K*K, 1]
 
  
    l_dist = tf.multiply(mu_norm, inter_mask)
    l_dist = tf.reduce_sum(l_dist)/(tf.reduce_sum(inter_mask)+1e-12) 

    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(mu, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale*(l_var + l_dist + l_reg)
    
    return loss, l_var, l_dist, l_reg



def build_discriminative_loss(  embedding,
                                label_map,
                                neighbor,
                                delta_v, delta_d, # Hyper
                                param_var, param_dist, param_reg, # Hyper
                                include_bg=True,
                                name='disc_loss'):
    
    """
    :param embedding: [B W H C]
    :param label_map: [B W H 1]
    :param neighbor: neighbot list
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    :param include_bg: weather take background as an independent object
    """

    with tf.variable_scope(name):
        def cond(out_loss, out_var, out_dist, out_reg,   
                    embedding, label_map, neighbor, i):
            return tf.less(i, tf.shape(embedding)[0])

        def body(out_loss, out_var, out_dist, out_reg,   
                    embedding, label_map, neighbor, i):
            disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(
                                                    embedding[i],
                                                    label_map[i],
                                                    neighbor[i],
                                                    delta_v, delta_d, # Hyper
                                                    param_var, param_dist, param_reg, # Hyper
                                                    include_bg)

            out_loss = out_loss.write(i, disc_loss)
            out_var = out_var.write(i, l_var)
            out_dist = out_dist.write(i, l_dist)
            out_reg = out_reg.write(i, l_reg)

            return out_loss, out_var, out_dist, out_reg, embedding, label_map, neighbor, i + 1

        # TensorArray is a data structure that support dynamic writing
        output_ta_loss = tf.TensorArray(dtype=tf.float32,
                    size=0,
                    dynamic_size=True)
        output_ta_var = tf.TensorArray(dtype=tf.float32,
                    size=0,
                    dynamic_size=True)
        output_ta_dist = tf.TensorArray(dtype=tf.float32,
                    size=0,
                    dynamic_size=True)
        output_ta_reg = tf.TensorArray(dtype=tf.float32,
                    size=0,
                    dynamic_size=True)

        out_loss_op, out_var_op, out_dist_op, out_reg_op, _, _, _, _  = tf.while_loop(cond, body, 
                                                            [output_ta_loss, 
                                                            output_ta_var, 
                                                            output_ta_dist, 
                                                            output_ta_reg, 
                                                            embedding, 
                                                            label_map, 
                                                            neighbor,
                                                            0])
        out_loss_op = out_loss_op.stack()
        out_var_op = out_var_op.stack()
        out_dist_op = out_dist_op.stack()
        out_reg_op = out_reg_op.stack()
        
        disc_loss = tf.reduce_mean(out_loss_op)
        l_var = tf.reduce_mean(out_var_op)
        l_dist = tf.reduce_mean(out_dist_op)
        l_reg = tf.reduce_mean(out_reg_op)

        return disc_loss, l_var, l_dist, l_reg




