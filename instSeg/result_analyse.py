import numpy as np
from skimage.measure import regionprops
from instSeg.utils import adj_matrix
from skimage.measure import label as relabel
# from sklearn.decomposition import PCA

def visualize_embedding(embedding, mask=None, metric='cos'):
    import umap

    H, W = embedding.shape[0], embedding.shape[1]
    if mask is not None:
        rr, cc = np.nonzero(mask > 0)
        embedding = embedding[rr, cc, :]
    else:
        embedding = np.reshape(embedding, (-1, embedding.shape[-1]))
    if metric == 'cos':
        embedding = embedding/np.sqrt(np.sum(embedding**2, axis=1, keepdims=True)+1e-12)

    reducer = umap.UMAP(n_neighbors=50, min_dist=0, n_components=3)
    embedding = reducer.fit_transform(embedding)
    embedding = (embedding - embedding.min())/(embedding.max() - embedding.min()) * 255
    
    if mask is not None:
        vis = np.zeros((H, W, 3))
        vis[rr, cc, :] = embedding
    else:
        vis = np.reshape(embedding, (H, W, 3))

    return vis.astype(np.uint8)

def embedding_analysis(instance, embedding, metric='cos', radius=None):
    instance = relabel(instance)
    adj = adj_matrix(np.expand_dims(instance, axis=0), radius=radius, progress_bar=False)[0]
    embedding = embedding/np.sqrt(np.sum(embedding**2, axis=1, keepdims=True)+1e-12)
    
    lmax = np.max(instance)
    adj = adj[1:lmax+1, 1:lmax+1]
    # print(radius, np.sum(adj, axis=0))

    means = {}
    consistency = {}

    for r in regionprops(instance):
        emb = embedding[r.coords[:,0], r.coords[:,1], :]
        
        mean = np.mean(emb, axis=0, keepdims=True)
        if metric == 'cos':
            mean = mean/np.sqrt(np.sum(mean**2, axis=1, keepdims=True)+1e-12)
            diff = np.arccos(np.clip(np.sum(mean * emb, axis=1), -1+1e-12, 1-1e-12))
            means[r.label] = mean
        elif metric == 'euclidean':
            diff = np.sqrt(np.sum((emb - mean) ** 2, axis=1)+1e-12)
            means[r.label] = mean
        
        consistency[r.label] = np.std(diff)

    distinctiveness = (np.copy(adj) * 0).astype(np.float32)
    for i in range(0, lmax):
        for j in range(0, lmax):
            if i == j:
                distinctiveness[i,j] = float('inf')
            elif adj[i, j] == True:
                m1, m2 = means[i+1], means[j+1]
                if metric == 'cos':
                    distinctiveness[i,j] = np.arccos(np.clip(np.abs(np.sum(m1 * m2)), 0, 1-(1e-12)))
                elif metric == 'euclidean':
                    distinctiveness[i,j] = np.sqrt(np.sum((m1 - m2) ** 2)+1e-12)
            else:
                distinctiveness[i,j] = float('inf')
    
    return consistency, distinctiveness

def false_merge(distinctiveness, thres, metric='cos'):
    t = distinctiveness.min(axis=0)
    N_false = np.sum(t < thres)
    return N_false
    
