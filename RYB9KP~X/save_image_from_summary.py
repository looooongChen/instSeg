import os
import tensorflow as tf
from skimage.io import imsave
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")

def save_images_from_event(fn, tag, output_dir='./image_from_summary/'):
    """
    :param fn: tfevents file like './model/summary/train/events.out.tfevents.1574935182.maxwell'
    :param tag: name of images like 'emb_dim1-3/image'

    Possible tag (task-dependent):
    input_image/image
    ground_truth/image
    distance_map/image
    loss_embedding
    loss_embedding2
    loss_embedding_inner
    loss_embedding_inter
    emb_dim1-3/image
    emb2_dim1-3/image
    loss_dist
    output_dist/image
    loss
    lr
    """

    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        # count = 0
        # total = len(list(tf.train.summary_iterator(fn)))
        # print(total)
        for e in tqdm(tf.train.summary_iterator(fn)):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    imsave(output_fn, im)
                    # if (count/total*100%10 < 0.5):
                    #     print("Process: {}% ".format(int(count/total*100)))
                    # count += 1  

        print("Mission accomplished: {} images are saved in '{}..'".format(count, output_dir))

save_images_from_event('./model_CVPPP2017/summary/train/events.out.tfevents.1582016675.pc173', 'emb_dim1-3/image')

