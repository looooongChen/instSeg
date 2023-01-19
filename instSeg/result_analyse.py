from matplotlib.pyplot import axis
import numpy as np
from sklearn.decomposition import PCA

def visualize_embedding(embedding):
    H, W = embedding.shape[0], embedding.shape[1]
    embedding = embedding.reshape((H*W, embedding.shape[-1]))
    pca = PCA(n_components=3)
    pca.fit(embedding)
    embedding = pca.fit_transform(embedding)
    embedding = embedding.reshape((H, W, 3))

    embedding = (embedding - embedding.min())/(embedding.max() - embedding.min()) * 255

    return embedding.astype(np.uint8)


def embedding_homogeneity(instance, embedding, metric='cos'):

    if isinstance(instance, list):
        instance = np.array(instance)
        instance = np.moveaxis(instance, 0, -1)
    elif instance.shape[-1] == 1:
        label_stack = np.zeros((*instance.shape[:2], max(np.unique(instance))+1), dtype=np.uint8)
        X, Y = np.meshgrid(np.arange(0, instance.shape[0]), np.arange(0, instance.shape[1]), indexing='ij')
        label_stack[X.flatten(), Y.flatten(), instance.flatten()] = 1
    instance = instance > 0
    instance = instance * (np.sum(instance, axis=-1, keepdims=True) == 1)


    if metric == 'cos':
        embedding = embedding / (np.linalg.norm(embedding, ord=2, axis=-1, keepdims=True) + 1e-6)

    Diff = []

    for idx in range(instance.shape[-1]):
        rr, cc = np.nonzero(instance[:,:,idx])
        E = embedding[rr, cc, :]
        if metric == 'cos':
            mE = np.sum(E, axis=0, keepdims=True)
            mE = mE / (np.linalg.norm(mE, ord=2, axis=1, keepdims=True) + 1e-6)
            Diff.append(np.sum(E * mE, axis=-1))
        
    return Diff
    
def embedding_distinctiveness(instance, embedding, radius):
    pass