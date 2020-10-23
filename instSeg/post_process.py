import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.morphology import disk, erosion
from skimage.filters import gaussian

def smooth_emb(emb, radius):
    emb = emb.copy()
    w = disk(radius)/np.sum(disk(radius))
    for i in range(emb.shape[-1]):
        emb[:, :, i] = ndimage.convolve(emb[:, :, i], w, mode='reflect')
    emb = emb /(1e-8 + np.linalg.norm(emb, ord=2, axis=-1, keepdims=True))
    return emb

# def maskViaSeed(pred, thres=0.5, min_distance=5, similarity_thres=0.7):

#     embedding = np.squeeze(pred['embedding'])
#     emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    
#     semantic = np.squeeze(np.argmax(pred['semantic'], axis=-1)) if 'semantic' in pred.keys() else np.ones(embedding.shape[:-1])
#     dist = np.squeeze(pred['dist']) * (semantic>0)
#     loc_max = peak_local_max(dist, min_distance=min_distance, threshold_rel=thres, indices=False)

#     seeds = loc_max * (semantic>0)
#     labels = np.zeros(semantic.shape, np.uint32)
#     label_obj = np.zeros(semantic.shape, np.uint32)

#     count = 1
#     for r, c in zip(*np.nonzero(seeds)):
#         if labels[r, c] != 0:
#             continue
#         label_obj = label_obj * 0
#         label_obj[r, c] = 1

#         while True:
#             dilated = dilation(label_obj, square(3))
#             mask1 = semantic == semantic[r, c]
#             mask2 = (label_obj != dilated) * (labels == 0)

#             front_r, front_c = np.nonzero(mask1 * mask2)

#             similarity = [np.dot(embedding[f_r, f_c, :], embedding[r, c, :]) for f_r, f_c in zip(front_r, front_c)]
#             add_ind = np.array([s > similarity_thres for s in similarity])

#             if np.all(add_ind == False):
#                 labels = labels + label_obj * count
#                 count += 1
#                 break
#             label_obj[front_r[add_ind], front_c[add_ind]] = 1

#     return label(labels)

def remove_noise(seg, dist, min_size=10, min_intensity=0.1):
    max_instensity = dist.max()
    props = regionprops(seg, intensity_image=dist)
    for p in props:
        if p.area < min_size:
            seg[seg==p.label] = 0
        if p.mean_intensity/max_instensity < min_intensity:
            seg[seg==p.label] = 0
    return label(seg)

def maskViaRegion(raw_pred, thres=0.7, similarity_thres=0.7, seed_distance=5):
    '''
    Args:
        raw_pred: a dict containing predictions of at least 'embedding', 'dist', optionally 'semantic'
        thres: threshold to get region from 'dist'
        similarity_thres: threshold distinguishing object in the embedding space
    '''
    embedding = np.squeeze(raw_pred['embedding'])
    # embedding = smooth_emb(embedding, 3)
    emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    semantic = np.squeeze(raw_pred['semantic']) if 'semantic' in raw_pred.keys() else np.ones(embedding.shape[:-1])
    # dist = gaussian(np.squeeze(raw_pred['dist']), sigma=3) * (semantic>0)
    # raw_pred['dist'] = erosion(raw_pred['dist'], disk(2)) 
    dist = np.squeeze(raw_pred['dist']) * (semantic>0)
    # dist = erosion(dist, dist(2))

    regions = label((dist > thres*dist.max()) * (semantic>0))
    # regions = label(peak_local_max(dist, min_distance=seed_distance, threshold_rel=thres, indices=False))
    base = 10 ** np.ceil(np.log10(len(np.unique(regions))))
    regions = (regions + semantic * base * (regions>0)).astype(np.uint32)  
    props = regionprops(regions)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]
        emb_mean = np.mean(embedding[row, col], axis=0)
        emb_mean = emb_mean/np.linalg.norm(emb_mean)
        mean[p.label] = emb_mean

    while True:
        dilated = dilation(regions, square(3))
        mask1 = (dilated - dilated % base) == semantic * base
        mask2 = (regions != dilated) * (regions == 0)
        front_r, front_c = np.nonzero(mask1 * mask2)

        similarity = [np.dot(embedding[r, c, :], mean[dilated[r, c]])
                      for r, c in zip(front_r, front_c)]
        add_ind = np.array([s > similarity_thres for s in similarity])

        if np.all(add_ind == False):
            break
        regions[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    # while True:
    #     dilated = dilation(regions, square(3))
    #     mask1 = (dilated - dilated % base) == semantic * base
    #     mask2 = (regions != dilated) * (regions == 0)
    #     front_r, front_c = np.nonzero(mask1 * mask2)

    #     if len(front_r) == 0:
    #         break
    #     regions[front_r, front_c] = dilated[front_r, front_c]

    regions = label(regions)

    return regions


# def remove_noise(l_map, d_map, min_size=10, min_intensity=0.1):
#     max_instensity = d_map.max()
#     props = regionprops(l_map, intensity_image=d_map)
#     for p in props:
#         if p.area < min_size:
#             l_map[l_map==p.label] = 0
#         if p.mean_intensity/max_instensity < min_intensity:
#             l_map[l_map==p.label] = 0
#     return label(l_map)


# from mws import MutexPixelEmbedding
# def mutex(pred):

#     embedding = np.squeeze(pred['embedding'])
#     emebdding = embedding / (1e-8 + np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True))
    
#     semantic = np.squeeze(np.argmax(pred['semantic'], axis=-1)) if 'semantic' in pred.keys() else np.ones(embedding.shape[:-1])

#     m = MutexPixelEmbedding(similarity='cos', lange_range=8, min_size=10)
#     seg = m.run(embedding, semantic>0)

#     return label(seg)

    

