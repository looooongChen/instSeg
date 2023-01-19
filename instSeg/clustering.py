from sklearn.decomposition import PCA
import numpy as np

def pca_clustering(data, mask=None):
    '''
    data: of size H x W x C
    '''
    if mask is not None:
        rr, cc = np.nonzero(mask)
        data_flat = data[rr,cc,:]
    else:
        data_flat = data.reshape((-1, data.shape[-1]))
    pca = PCA(n_components=data.shape[-1])
    pca.fit(np.concatenate((data_flat, -data_flat), axis=0))
    data_flat = pca.transform(data_flat)

    labels_flat = np.argmax(np.absolute(data_flat), axis=-1)
    
    sign = data_flat[range(data_flat.shape[0]), labels_flat]
    labels_flat = labels_flat + 1
    labels_flat = labels_flat  + (sign < 0) * data.shape[-1]

    if mask is not None:
        labels = np.zeros(data.shape[:2])
        labels[rr,cc] = labels_flat
    else:
        labels = labels_flat.reshape(data.shape[:2])

    return labels


