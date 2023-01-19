from bbbc010 import read_BBBC010
from occ2014 import read_OCC
import instSeg
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import shutil

# color = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
# colors = ['red', 'blue', 'yellow', 'magenta', 'green', 'darkorange', 'cyan']
colors = ['red', 'blue', 'magenta', 'green', 'cyan', 'indigo']
colors = [mcolors.CSS4_COLORS[c] for c in colors] 

save_dir = './embedding_vis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_dir = './model_embedding_occ2014_vis'
model = instSeg.load_model(model_dir)


k = '550'
ds_dir = '/work/scratch/chen/Datasets/OCC2014/OCC2014'
imgs, _, masks = read_OCC(ds_dir, sz=(model.config.W, model.config.H), part='test', keys=[k])


def plot_embedding(save_path, embedding):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # ax.set_aspect("auto")
    ax.set_box_aspect(aspect = (1,1,0.9))
    ax.grid(False)
    plt.axis('off')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="lightgray", linewidths=1)

    for idx, e in embedding.items():
        ax.scatter(e[:,0], e[:,1], e[:,2], color=colors[idx % len(colors)], s=1, alpha=1, linewidths=1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)


if os.path.exists(os.path.join(save_dir, str(k))):
    shutil.rmtree(os.path.join(save_dir, str(k)))
os.makedirs(os.path.join(save_dir, str(k)))

cv2.imwrite(os.path.join(save_dir, str(k), 'img.png'), imgs[k])

vis_contour = instSeg.vis.vis_instance_contour(imgs[k], np.array(masks[k]))
cv2.imwrite(os.path.join(save_dir, str(k), 'img_gt.png'), vis_contour)

embedding = model.predict_raw(imgs[k])['embedding']
embedding = embedding/np.linalg.norm(embedding, axis=-1, keepdims=True, ord=2)
M = np.array([m>0 for m in masks[k]])
non_overlap = np.sum(M, axis=0) == 1
union = np.sum(M, axis=0) > 0

vis_embedding = (embedding-embedding.min())/(embedding.max()-embedding.min())*255
cv2.imwrite(os.path.join(save_dir, str(k), 'embedding_nonoverlap.png'), (vis_embedding*np.expand_dims(non_overlap, axis=-1)).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, str(k), 'embedding_union.png'), (vis_embedding*np.expand_dims(union, axis=-1)).astype(np.uint8))

idx_r, idx_c = np.nonzero(non_overlap)
embedding_pca = embedding[idx_r, idx_c, :]
embedding_pca = np.concatenate((embedding_pca, -embedding_pca), axis=0)
pca = PCA()
pca.fit(embedding_pca)

embedding_dict = {}
for idx, m in enumerate(M):
    idx_r, idx_c = np.nonzero(m*non_overlap)
    if len(idx_r) == 0:
        continue
    e = embedding[idx_r, idx_c, :]
    e = e/np.linalg.norm(e, axis=-1, keepdims=True, ord=2)
    # e = pca.transform(e)
    ee = np.zeros(e.shape)
    ee[:,0] = -1*e[:,0]
    ee[:,1] = e[:,1]
    ee[:,2] = -1*e[:,2]
    embedding_dict[idx] = ee*1
plot_embedding(os.path.join(save_dir, str(k), 'emb.png'), embedding_dict)

embedding_dict = {}
for idx, m in enumerate(M):
    idx_r, idx_c = np.nonzero(m*non_overlap)
    if len(idx_r) == 0:
        continue
    e = embedding[idx_r, idx_c, :]
    e = e/np.linalg.norm(e, axis=-1, keepdims=True, ord=2)
    e = pca.transform(e)
    ee = np.zeros(e.shape)
    ee[:,0] = -1*e[:,0]
    ee[:,1] = e[:,1]
    ee[:,2] = -1*e[:,2]
    embedding_dict[idx] = ee*1
plot_embedding(os.path.join(save_dir, str(k), 'emb_pca.png'), embedding_dict)
