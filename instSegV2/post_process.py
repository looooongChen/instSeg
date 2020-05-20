import numpy as np
from skimage.measure import regionprops, label
from skimage.measure import label as assign_label
from skimage.morphology import erosion as im_erosion
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from skimage.feature import peak_local_max

def maskViaSeed(pred, thres=0.7, min_distance=5, similarity_thres=0.7):
    assert 'embedding' in pred.keys() and 'dist' in pred.keys()
    seeds = get_seeds(pred['dist'][1], min_distance=min_distance, thres=thres)
    embedding = pred['embedding'][1]
    semantic_map = get_semantic(pred['semantic'][1]) if 'semantic' in pred.keys() else None
    return mask_from_seeds(embedding, seeds, similarity_thres=similarity_thres, mask=semantic_map)
    
def get_semantic(pred):
    pred = np.squeeze(pred)
    return np.squeeze(np.argmax(pred, axis=-1))

def get_seeds(dist_map, min_distance=5, thres=0.7):
    dist_map = np.squeeze(dist_map)
    mask = peak_local_max(dist_map, min_distance=min_distance, threshold_abs=thres * dist_map.max(), indices=False)
    return mask

def mask_from_seeds(embedding, seeds, similarity_thres=0.7, mask=None):
    
    embedding = np.squeeze(embedding)
    seeds = np.squeeze(seeds)

    if mask is not None:
        seeds = seeds * (mask>0).astype(seeds.dtype)
        # embedding = embedding * np.expand_dims(mask.astype(embedding.dtype), axis=-1)

    seeds = label(seeds)
    props = regionprops(seeds)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]
        emb_mean = np.mean(embedding[row, col], axis=0)
        emb_mean = emb_mean/np.linalg.norm(emb_mean)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3)) * (mask>0).astype(seeds.dtype)

        front = seeds != dilated
        # front = np.logical_and(seeds != dilated, seeds == 0) 
        front_r, front_c = np.nonzero(front)

        similarity = [np.dot(embedding[r, c, :], mean[dilated[r, c]])
                      for r, c in zip(front_r, front_c)]
        
        # bg = seeds[front_r, front_c] == 0
        # add_ind = np.logical_and([s > similarity_thres for s in similarity], bg)
        add_ind = np.array([s > similarity_thres for s in similarity])

        if np.all(add_ind == False):
            break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds
