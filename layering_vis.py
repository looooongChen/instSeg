from turtle import color
import instSeg
import numpy as np
import os
from skimage.morphology import dilation, erosion
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2


save_dir = './layering_vis'

synth_image = False
save_embedding = True
vis_img = False
vis_embedding_space = False

if synth_image:
    img1 = cv2.imread(os.path.join(save_dir, 'img7.png'), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(os.path.join(save_dir, 'img229.png'), cv2.IMREAD_UNCHANGED)

    # mask1 = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in ['./cytoplasm7/cytoplasm0.png', './cytoplasm7/cytoplasm1.png', './cytoplasm7/cytoplasm2.png']
    # mask1 = np.sum(mask1, axis=0) > 0
    mask2 = cv2.imread(os.path.join(save_dir, 'cytoplasm229/cytoplasm1.png'), cv2.IMREAD_UNCHANGED) > 0

    img = img1 * (1-mask2) + img2 * mask2

    cv2.imwrite(os.path.join(save_dir, 'img.png'), img.astype(np.uint8))



colors_obj = ['yellow', 'blue', 'cyan', 'red']
# area_colors = ['#F7B538', '#4CB944', '#58C7E7']
# overlap_colors = ['#6E7E85', '#A31621', '#000000']
colors_overlap = ['#4CB944', '#A31621', '#58C7E7']


img = cv2.imread(os.path.join(save_dir, 'img57.png'), cv2.IMREAD_UNCHANGED)
img = np.stack((img, img, img), axis=-1)
masks = [cv2.imread(os.path.join(save_dir, 'cytoplasm57', f), cv2.IMREAD_UNCHANGED)>0 for f in sorted(os.listdir(os.path.join(save_dir, 'cytoplasm57')))]

boundaries = []
for M in masks:
    boundaries.append(dilation(M) != erosion(M))
boundary = np.sum(boundaries, axis=0) > 0

area_nonoverlap = []
overlap = np.sum(masks, axis=0) > 1
for M in masks:
    area_nonoverlap.append(M*(1-overlap)*(1-boundary))
area_overlap = []
for i,j in zip([0,1,2],[1,2,3]):
    area_overlap.append(masks[i]*masks[j]*(1-boundary))

def vis_contour(ax, masks, colors):
    for M, color in zip(masks, colors):
        C, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(C) > 0:
            C = np.squeeze(C[0], axis=1)
            rr, cc = list(C[:,0]), list(C[:,1])
            rr.append(rr[0])
            cc.append(cc[0])
            ax.plot(rr, cc, color=color)

def fill_region(ax, regions, colors, pattern='.'):
    idx = 0
    for M in regions:
        print(np.sum(M))
        C, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(C) > 0:
            C = np.squeeze(C[0], axis=1)
            pg = ax.add_patch(Polygon(C, closed=True, fill=False))
            pg.set_hatch(pattern)
            pg.set_linewidth(0)
            pg.set_color(colors[idx])
            idx += 1


def get_embedding(model, img, fg, save_dir):
    raw = model.predict_raw(img)
    # instances, layered = model.postprocess(raw, post_processing='layered_embedding')
    # save vis of layering
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # vis = instSeg.vis.vis_instance_contour(imgs[k], np.array(instances))
    # cv2.imwrite(os.path.join(save_dir, 'pred.png'), vis)
    # vis = instSeg.vis.vis_instance_contour(imgs[k], np.array(masks[k]))
    # cv2.imwrite(os.path.join(save_dir, 'gt.png'), vis)
    # for i in range(layered.shape[-1]):
    #     cv2.imwrite(os.path.join(save_dir, 'layer_'+str(i)+'.png'), np.uint8(layered[:,:,i])*255)
    # fg = np.squeeze(raw['foreground']) > 0.5 
    # cv2.imwrite(os.path.join(save_dir, 'foreground.png'), np.uint8(fg*255))
    embedding = raw['layered_embedding'] if 'layered_embedding' in raw.keys() else raw['embedding']
    embedding = np.squeeze(embedding)
    for i in range(embedding.shape[-1]):
        cv2.imwrite(os.path.join(save_dir, 'layering_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*255))
        cv2.imwrite(os.path.join(save_dir, 'layering_masked_'+str(i)+'.png'), np.uint8(embedding[:,:,i]*fg*255))
    np.save(os.path.join(save_dir, 'embedding.npy'), embedding)



def get_area_embedding(embedding, mask, dimensions=[0,3,4]):
    print(dimensions)
    rr, cc = np.nonzero(mask)
    emb = embedding[rr,cc,:]
    return emb[:,dimensions]

if save_embedding:
    fg = cv2.resize((np.sum(masks, axis=0)>0).astype(np.uint8), (320, 320)) > 0.5
    model = instSeg.load_model(model_dir='/work/scratch/chen/instSeg/models_occ2014/model_unetS5_sparse_overlapTuned', load_best=None)
    get_embedding(model, img[:,:,0], fg, os.path.join(save_dir, 'initial'))
    model = instSeg.load_model(model_dir='/work/scratch/chen/instSeg/models_occ2014/model_unetS5', load_best=True)
    get_embedding(model, img[:,:,0], fg, os.path.join(save_dir, 'cos'))
    model = instSeg.load_model(model_dir='/work/scratch/chen/instSeg/models_occ2014/model_unetS5_sparse', load_best=True)
    get_embedding(model, img[:,:,0], fg, os.path.join(save_dir, 'layered'))
    model = instSeg.load_model(model_dir='/work/scratch/chen/instSeg/models_occ2014/model_unetS5_sparse_overlapTuned', load_best=True)
    get_embedding(model, img[:,:,0], fg, os.path.join(save_dir, 'overlap_tuned'))


if vis_img:
    plt.imshow(img)
    ax = plt.gca()
    fill_region(ax, area_nonoverlap, colors_obj, pattern='..')
    fill_region(ax, area_overlap, colors_overlap, pattern='**')
    vis_contour(ax, masks, colors_obj)

    plt.axis("off")
    plt.savefig(os.path.join(save_dir, 'vis.png'), dpi=300)


if vis_embedding_space:
    # from itertools import product, combinations
    # r = [0, 1]
    # ax.set_aspect("equal")
    # for s, e in combinations(np.array(list(product(r, r, r))), 2):
    #     if np.sum(np.abs(s-e)) == r[1]-r[0]:
    #         ax.plot3D(*zip(s, e), color="b")

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    embedding = np.load(os.path.join(save_dir, 'overlap_tuned', 'embedding.npy'))
    area_nonoverlap = [cv2.resize(m.astype(np.uint8), (320,320)) >0.8 for m in area_nonoverlap]
    area_overlap = [cv2.resize(m.astype(np.uint8), (320,320)) >0.8 for m in area_overlap]
    for idx, M in enumerate(area_nonoverlap):
        print(idx)
        dimensions = [0,4,7] if idx != 3 else [0,3,7]
        # dimensions = [0,1,2,3,4,5,6,7]
        emb = get_area_embedding(embedding, M, dimensions=dimensions)
        print(emb[:10])
        # emb = emb[:10]
        ax.scatter(emb[:,0], emb[:,1], emb[:,2], marker='o', s=10, alpha=0.5, linewidths=0, c=colors_obj[idx])
    for idx, M in enumerate(area_overlap):
        emb = get_area_embedding(embedding, M, dimensions=[0,4,7])
        ax.scatter(emb[:,0], emb[:,1], emb[:,2], marker='*', s=10, alpha=0.5, linewidths=0, c=colors_overlap[idx])

    ax.grid(False)
    ax.set_xticks(ticks=[0,1], labels=[0,1])
    ax.set_yticks(ticks=[0,1], labels=[0,1])
    ax.set_zticks(ticks=[0,1], labels=[0,1])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    # ax.xaxis._axinfo['juggled'] = (0,0,0)
    # ax.yaxis._axinfo['juggled'] = (1,1,1)
    # ax.zaxis._axinfo['juggled'] = (2,2,2)
    
    # ax.invert_xaxis()

    plt.savefig(os.path.join(save_dir, 'embedding.png'), dpi=300)





    