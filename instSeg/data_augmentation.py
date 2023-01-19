import tensorflow as tf


def gaussian_kernel(sigma):
    size = tf.math.round(sigma*2)
    g = tf.range(-size, size+1, dtype=tf.float32)
    g = tf.math.exp(-(tf.pow(g, 2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
    g_kernel = tf.tensordot(g, g, axes=0)
    return tf.stop_gradient(g_kernel / tf.reduce_sum(g_kernel))
     

def gaussian_blur(image, sigma):

    '''
    image: 3D
    '''
    
    shape = tf.shape(image)

    blurred = tf.expand_dims(image, axis=0)
    blurred = tf.cast(blurred, tf.float32)

    g_kernel = gaussian_kernel(sigma)
    g_kernel = tf.expand_dims(tf.expand_dims(g_kernel, -1), -1)
    g_kernel = tf.repeat(g_kernel, shape[-1], axis=-2)

    blurred = tf.nn.depthwise_conv2d(blurred, g_kernel, strides=[1, 1, 1, 1], padding="VALID")
    # blurred = tf.nn.conv2d(blurred, g_kernel, strides=[1, 1, 1, 1], padding="SAME")
    
    blurred = tf.squeeze(blurred, axis=0)
    blurred = tf.cast(blurred, image.dtype)
    
    return blurred


def interpolate(grid, query_points, interpolation='bilinear', border='constant'):
    """
    Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    
    Args:
        grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
        query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
        interpolation: interpolation method 'bilinear' or 'nearest'

    Returns:
        values: a 3-D `Tensor` with shape `[batch, N, channels]`
    
    Raises:
        ValueError: if the indexing mode is invalid, or if the shape of the inputs invalid.
    """

    if border == 'constant':
        grid = tf.pad(grid, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
        query_points = query_points + 1
        

    shape = tf.shape(grid)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    num_queries = tf.shape(query_points)[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    unstacked_query_points = tf.unstack(query_points, axis=2)

    for dim in [0, 1]:
        # with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        # max_idx is shape[dim + 1] - 2 so that max_idx + 1 is still a valid index into the grid.
        min_idx = tf.constant(0.0, dtype=query_type)
        max_idx = tf.cast(shape[dim + 1] - 2, query_type)

        floor = tf.minimum(tf.maximum(min_idx, tf.floor(queries)), max_idx)
        floor_int = tf.cast(floor, tf.int32)
        floors.append(floor_int)
        ceil = floor_int + 1
        ceils.append(ceil)

        # compute alpha for taking linear combinations of pixel values from the image
        # same dtype with grid
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)

    flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
    batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

    # helper function to get value from the flattened tensor
    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    # now, do the actual interpolation
    if interpolation == 'nearest':
        t = tf.less(alphas[0], 0.5)
        l = tf.less(alphas[1], 0.5)

        tl = tf.cast(tf.logical_and(t, l), tf.float32)
        tr = tf.cast(tf.logical_and(t, tf.logical_not(l)), tf.float32)
        bl = tf.cast(tf.logical_and(tf.logical_not(t), l), tf.float32)
        br = tf.cast(tf.logical_and(tf.logical_not(t), tf.logical_not(l)), tf.float32)

        interp = tf.multiply(top_left, tl) + tf.multiply(top_right, tr) \
                    + tf.multiply(bottom_left, bl) + tf.multiply(bottom_right, br)
    else:
        interp_top = alphas[1] * (top_right - top_left) + top_left
        interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
        interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def resize_image(image, size, interpolation='bilinear'):
    ''' 
    Args:
        image: shape of H x W x C
        size: (newH, newW)
    '''

    shape = tf.shape(image)
    if tf.reduce_all(shape[:2] == size):
        return image

    image = tf.expand_dims(image, axis=0) 
    batch_size, H, W, channels = 1, shape[0], shape[1], shape[2] 
    newH, newW = size[0], size[1]
    # get the query coordinates
    grid_x, grid_y = tf.meshgrid(tf.range(newW), tf.range(newH))
    grid_x = tf.cast(grid_x, tf.float32) * tf.cast(W / newW, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32) * tf.cast(H / newH, tf.float32)
    grid = tf.stack([grid_y, grid_x], axis=2)
    query_points = tf.expand_dims(grid, axis=0)
    query_points_flattened = tf.reshape(query_points, [1, newH * newW, 2])
    query_points_flattened = tf.repeat(query_points_flattened, batch_size, axis=0)
    # Compute values at the query points, then reshape the result back to the image grid.
    image_float = tf.cast(image, tf.float32)
    interpolated = interpolate(image_float, query_points_flattened, interpolation=interpolation)
    interpolated = tf.reshape(interpolated, [batch_size, newH, newW, channels])
    # interpolated = tf.cond(full_dim, lambda : image, lambda : tf.squeeze(interpolated, axis=0))

    return tf.cast(interpolated, image.dtype)

def warp_image(image, flow, interpolation="bilinear"):
    """
    Image warping using per-pixel flow vectors.

    Args:
        # image: 4-D `Tensor` with shape [batch, height, width, channels] or 3-D with shape [height, width, channels]
        # flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
        image: 3-D with shape [height, width, channels]
        flow: A 3-D float `Tensor` with shape `[height, width, 2]`.
    
    Note: the image and flow can be of type tf.half, tf.float32, or tf.float64, and do not necessarily have to be the same type.
    
    Returns:
        A 4-D/3-D Tensor with the same type as input image.
    
    Raises:
        ValueError: if height < 2 or width < 2 or the inputs have the wrong number of dimensions.
    """

    shape = tf.shape(image)
    image = tf.expand_dims(image, axis=0)
    flow = tf.expand_dims(flow, axis=0)

    batch_size, height, width, channels = 1, shape[0], shape[1], shape[2]

    # get the query coordinates
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    grid = tf.expand_dims(grid, axis=0)
    query_points = grid - flow
    query_points_flattened = tf.reshape(query_points, [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the image grid.
    interpolated = interpolate(tf.cast(image, tf.float32), query_points_flattened, interpolation=interpolation)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

    interpolated = tf.squeeze(interpolated, axis=0)

    return tf.cast(interpolated, image.dtype)


def elastic_flow(size, scale=10, strength=100):

    ''' size: shape of a 3D tensor '''
    sigma = 5
    filter_radius = 2*sigma

    size = tf.cast(size, tf.int32)
    scale = tf.cast(scale, tf.int32)
    strength = tf.cast(strength, tf.float32)

    flow = tf.random.uniform([(tf.cast(tf.math.ceil(size[0]/scale*filter_radius), tf.int32)+1)+ 1 + 2*filter_radius,
                              (tf.cast(tf.math.ceil(size[1]/scale*filter_radius), tf.int32)+1)+ 1 + 2*filter_radius,
                              2], -1, 1)                         
    flow = gaussian_blur(flow, sigma)
    flow = resize_image(flow, size[-3:-1], interpolation='bilinear')
    norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(flow ** 2, axis=-1)))
    flow = flow / norm * strength

    return flow

# def crop_and_resize(image, bbx, interpolation='bilinear'):

#     '''
#     Args:
#         image: 3D or 4D tensor, B x H x W x C or H x W x C
#         bbx: bbx cooridinates, B x 4
#     '''

#     shape = image.get_shape()
#     image_c = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image

#     bbx_ind = tf.range(0, image_c.get_shape()[0], delta=1, dtype=tf.int32)
#     image_c = tf.image.crop_and_resize(image_c, bbx, bbx_ind, image_c.get_shape()[1:3], method=interpolation.lower())

#     image_c = tf.squeeze(image_c, axis=0) if shape.ndims == 3 else image_c

#     return tf.cast(image_c, image.dtype)


# def rotate(image, angle, interpolation='bilinear'):

#     '''
#     Args:
#         image: 3D / 4D of shape B x H x W / B x H x W x C 
#         angle: scale / 1D of shape - / B  
#     '''

#     shape = image.get_shape()
#     image_q = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image

#     sz = image_q.get_shape()
#     batch_size, height, width, channels = sz[0], sz[1], sz[2], sz[3]

#     # get the query coordinates
#     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
#     grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), tf.float32)
#     query_points_flattened = tf.reshape(grid, [height * width, 2])
#     query_points_flattened = tf.expand_dims(query_points_flattened, axis=0)
#     query_points_flattened = tf.repeat(query_points_flattened, batch_size, axis=0)
#     center = tf.stop_gradient(tf.convert_to_tensor([[height/2, width/2]], dtype=tf.float32))
    
#     theta = 0.017453292519943295 * angle
#     R_m = tf.stack([(tf.cos(theta), tf.sin(theta)), (-tf.sin(theta), tf.cos(theta))], axis=0)
#     R_m = tf.expand_dims(R_m, axis=-1) if R_m.get_shape().ndims == 2 else R_m
#     R_m = tf.transpose(R_m, perm=[2,0,1])
#     R_m = tf.repeat(R_m, batch_size, axis=0) if R_m.get_shape()[0] != batch_size else R_m
#     R_m = tf.cast(R_m, tf.float32)
#     # print('aa', query_points_flattened.shape, center.shape, R_m.shape)

#     query_points_flattened = tf.unstack(query_points_flattened, axis=0)
#     R_m = tf.unstack(R_m, axis=0)

#     query_points_flattened = [tf.expand_dims(tf.linalg.matmul(P-center, R) + center, axis=0) for P, R in zip(query_points_flattened, R_m)]

#     # query_points_flattened = tf.linalg.matmul(query_points_flattened-center, R_m) + center
#     # query_points_flattened = tf.expand_dims(query_points_flattened, axis=0)
#     query_points_flattened = tf.concat(query_points_flattened, axis=0)
#     query_points_flattened = tf.where(query_points_flattened<0, tf.abs(query_points_flattened), query_points_flattened)
#     border = tf.stop_gradient(tf.convert_to_tensor([[[height, width]]], dtype=tf.float32))
#     query_points_flattened = tf.where(query_points_flattened>border, 2*border-query_points_flattened, query_points_flattened)


#     image_q = tf.cast(image_q, tf.float32)
#     interpolated = interpolate(image_q, query_points_flattened, interpolation=interpolation)
#     interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

#     interpolated = tf.squeeze(interpolated, axis=0) if shape.ndims == 3 else interpolated

#     return tf.cast(interpolated, image.dtype)


# def translate(image, offset, interpolation='bilinear'):

#     '''
#     Args:
#         image: 3D / 4D of shape B x H x W / B x H x W x C 
#         offset: 1D / 2D of shape 2 / B x 2 
#     '''

#     shape = image.get_shape()
#     image_q = tf.expand_dims(image, axis=0) if shape.ndims == 3 else image

#     sz = image_q.get_shape()
#     batch_size, height, width, channels = sz[0], sz[1], sz[2], sz[3]

#     # get the query coordinates
#     grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
#     grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), tf.float32)
#     query_points_flattened = tf.reshape(grid, [height * width, 2])
#     query_points_flattened = tf.expand_dims(query_points_flattened, axis=0)
#     query_points_flattened = tf.repeat(query_points_flattened, batch_size, axis=0)
#     offset = tf.stop_gradient(tf.convert_to_tensor(offset, dtype=tf.float32))
#     offset = tf.expand_dims(offset, axis=0) if offset.get_shape().ndims == 1 else offset
#     offset = tf.expand_dims(offset, axis=1)
    
#     query_points_flattened = query_points_flattened + offset
#     query_points_flattened = tf.where(query_points_flattened<0, tf.abs(query_points_flattened), query_points_flattened)
#     border = tf.stop_gradient(tf.convert_to_tensor([[height, width]], dtype=tf.float32))
#     query_points_flattened = tf.where(query_points_flattened>border, 2*border-query_points_flattened, query_points_flattened)

#     image_q = tf.cast(image_q, tf.float32)
#     interpolated = interpolate(image_q, query_points_flattened, interpolation=interpolation)
#     interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])

#     interpolated = tf.squeeze(interpolated, axis=0) if shape.ndims == 3 else interpolated

#     return tf.cast(interpolated, image.dtype)