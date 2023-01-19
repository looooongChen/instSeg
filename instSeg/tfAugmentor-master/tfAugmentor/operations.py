from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tfAugmentor.gaussian import *
from tfAugmentor.warp import *
from tfAugmentor.color import *


# class SyncRandomRotateRunner(Sync):

#     def run(self, images):
#         # occur = tf.random.uniform([], 0, 1) < self.probability 
#         occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
#         angle = tf.random.uniform([1], 0, 2*3.1415926)
#         for k in images.keys():
#             if k in self.Ops.keys():
#                 images[k] = self.Ops[k].run(images[k], angle, occur)

# class SyncRandomCropRunner(Sync):
    
#     def __init__(self, probability, scale, preserve_aspect_ratio):
#         super().__init__(probability)
#         if isinstance(scale, tuple) or isinstance(scale, list):
#             self.scale = scale
#         else:
#             self.scale = (scale, scale)
#         self.preserve_aspect_ratio = preserve_aspect_ratio

#     def get_bbx(self, batch_size):
#         sz = tf.random.uniform([batch_size, 2], self.scale[0], self.scale[1])
#         if self.preserve_aspect_ratio:
#             sz = tf.random.uniform([batch_size, 1], self.scale[0], self.scale[1])
#             sz = tf.concat([sz, sz], axis=-1)
#         else:
#             sz = tf.random.uniform([batch_size, 2], self.scale[0], self.scale[1])
#         offset = tf.multiply(1-sz, tf.random.uniform([batch_size, 2], 0, 1))
#         return tf.concat([offset, offset+sz], axis=1)

#     def run(self, images):
#         img = images[list(self.Ops.keys())[0]]
#         full_dim = tf.equal(tf.size(tf.shape(img)), 4)
#         batch_size = tf.cond(full_dim, lambda : tf.shape(img)[0], lambda : 1)

#         # occur = tf.random.uniform([], 0, 1) < self.probability 
#         occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
#         bbx = self.get_bbx(batch_size)
#         for k in images.keys():
#             if k in self.Ops.keys():
#                 images[k] = self.Ops[k].run(images[k], bbx, occur)


#### opeartion classed ####

class Op(metaclass=ABCMeta):

    @abstractmethod
    def run(self, item_flatten):
        pass

class SimpleOp(Op):

    def __init__(self, func, flatten_signature, image, label, probability=1):
        self.ops = []
        self.probability = probability
        items_aug = label + image
        for s in flatten_signature:
            if s in items_aug:
                self.ops.append(lambda image: func(image))
            else:
                self.ops.append(lambda x: x)

    def run(self, item_flatten):
        occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
        item_flatten = [tf.cond(occur, lambda : op(item), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten

class LeftRightFlip(SimpleOp):
    def __init__(self, flatten_signature, image, label, probability=1):
        super().__init__(tf.image.flip_left_right, flatten_signature, image, label, probability)

class UpDownFlip(SimpleOp):
    def __init__(self, flatten_signature, image, label, probability=1):
        super().__init__(tf.image.flip_up_down, flatten_signature, image, label, probability)

class Rotate90(SimpleOp):
    def __init__(self, flatten_signature, image, label, probability=1):
        super().__init__(lambda img: rotate(img, 90), flatten_signature, image, label, probability)

class Rotate180(SimpleOp):
    def __init__(self, flatten_signature, image, label, probability=1):
        super().__init__(lambda img: rotate(img, 180), flatten_signature, image, label, probability)

class Rotate270(SimpleOp):
    def __init__(self, flatten_signature, image, label, probability=1):
        super().__init__(lambda img: rotate(img, 270), flatten_signature, image, label, probability)

class Rotate(SimpleOp):
    def __init__(self, flatten_signature, image, label, angle, probability=1):
        self.ops = []
        self.probability = probability
        for s in flatten_signature:
            if s in image:
                self.ops.append(lambda image: rotate(image, angle, interpolation='bilinear'))
            elif s in label:
                self.ops.append(lambda image: rotate(image, angle, interpolation='nearest'))
            else:
                self.ops.append(lambda x: x)

class RandomRotate(Op):
    def __init__(self, flatten_signature, image, label, probability=1):
        self.ops = []
        self.probability = probability
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, angle: rotate(image, angle, interpolation='bilinear'))
                if self.info_idx is None:    
                    self.info_idx = idx
            elif s in label:
                self.ops.append(lambda image, angle: rotate(image, angle, interpolation='nearest'))
                if self.info_idx is None:    
                    self.info_idx = idx
            else:
                self.ops.append(lambda x, angle: x)

    def run(self, item_flatten):
        if self.info_idx is not None:
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            batch_size = item_flatten[self.info_idx].get_shape()[0] if item_flatten[self.info_idx].get_shape().ndims == 4 else 1
            angle = tf.random.uniform([batch_size], 0, 360)
            item_flatten = [tf.cond(occur, lambda : op(item, angle), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten

class Translate(SimpleOp):
    def __init__(self, flatten_signature, image, label, offset, probability=1):
        self.ops = []
        self.probability = probability
        for s in flatten_signature:
            if s in image:
                self.ops.append(lambda image: translate(image, offset, interpolation='bilinear'))
            elif s in label:
                self.ops.append(lambda image: translate(image, offset, interpolation='nearest'))
            else:
                self.ops.append(lambda x: x)

class RandomTranslate(Op):
    def __init__(self, flatten_signature, image, label, translation_range, probability=1):
        self.ops = []
        self.probability = probability
        self.translation_range = translation_range
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, offset: translate(image, offset, interpolation='bilinear'))
                if self.info_idx is None:    
                    self.info_idx = idx
            elif s in label:
                self.ops.append(lambda image, offset: translate(image, offset, interpolation='nearest'))
                if self.info_idx is None:    
                    self.info_idx = idx
            else:
                self.ops.append(lambda x, offset: x)

    def run(self, item_flatten):
        if self.info_idx is not None:
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            batch_size = item_flatten[self.info_idx].get_shape()[0] if item_flatten[self.info_idx].get_shape().ndims == 4 else 1
            offset = tf.random.uniform([batch_size, 2], self.translation_range[0], self.translation_range[1])
            item_flatten = [tf.cond(occur, lambda : op(item, offset), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten

class GaussianBlur(SimpleOp):

    def __init__(self, flatten_signature, image, label, sigma, probability=1):
        self.ops = []
        self.probability = probability
        for s in flatten_signature:
            if s in image:
                self.ops.append(lambda image: gaussian_blur(image, sigma))
            else:
                self.ops.append(lambda x: x)

class RandomCrop(Op):

    def __init__(self, flatten_signature, image, label, scale_range=[0.6, 0.8], preserve_aspect_ratio=False, probability=1):
        self.ops = []
        self.probability = probability
        self.scale_range = scale_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, bbx: crop_and_resize(image, bbx, interpolation='bilinear'))
                if self.info_idx is None:    
                    self.info_idx = idx 
            elif s in label:
                self.ops.append(lambda image, bbx: crop_and_resize(image, bbx, interpolation='nearest'))
                if self.info_idx is None:    
                    self.info_idx = idx 
            else:
                self.ops.append(lambda x, bbx: x)
    
    def crop_bbx(self, batch_size):
        sz = tf.random.uniform([batch_size, 2], self.scale_range[0], self.scale_range[1])
        if self.preserve_aspect_ratio:
            sz = tf.random.uniform([batch_size, 1], self.scale_range[0], self.scale_range[1])
            sz = tf.concat([sz, sz], axis=-1)
        else:
            sz = tf.random.uniform([batch_size, 2], self.scale_range[0], self.scale_range[1])
        offset = tf.multiply(1-sz, tf.random.uniform([batch_size, 2], 0, 1))
        return tf.concat([offset, offset+sz], axis=1)

    def run(self, item_flatten):
        if self.info_idx is not None:
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            batch = item_flatten[self.info_idx].get_shape()[0] if item_flatten[self.info_idx].get_shape().ndims == 4 else 1
            bbx = self.crop_bbx(batch)
            item_flatten = [tf.cond(occur, lambda : op(item, bbx), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten

class ElasticDeform(Op):

    def __init__(self, flatten_signature, image, label, scale=3, strength=100, probability=1):
        self.probability = probability
        self.scale = scale
        self.strength = strength
        self.ops = []
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, flow: warp_image(image, flow, interpolation='bilinear'))
                if self.info_idx is None:    
                    self.info_idx = idx 
            elif s in label:
                self.ops.append(lambda image, flow: warp_image(image, flow, interpolation='nearest'))
                if self.info_idx is None:    
                    self.info_idx = idx 
            else:
                self.ops.append(lambda x, flow: x)

    def run(self, item_flatten):
        if self.info_idx is not None:
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            flow = elastic_flow(item_flatten[self.info_idx].get_shape(), self.scale, self.strength)
            item_flatten = [tf.cond(occur, lambda : op(item, flow), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten

class RandomContrast(Op):
    def __init__(self, flatten_signature, image, label, contrast_range=[0.6, 1.4], probability=1):
        self.ops = []
        self.probability = probability
        self.contrast_range = contrast_range
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, f: adjust_contrast(image, f))
                if self.info_idx is None:    
                    self.info_idx = idx 
            else:
                self.ops.append(lambda x, f: x)

    def run(self, item_flatten):
        if self.info_idx is not None: 
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            batch_size = item_flatten[self.info_idx].get_shape()[0] if item_flatten[self.info_idx].get_shape().ndims == 4 else 1
            contrast = tf.random.uniform([batch_size], self.contrast_range[0], self.contrast_range[1])
            item_flatten = [tf.cond(occur, lambda : op(item, contrast), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten


class RandomGamma(Op):
    def __init__(self, flatten_signature, image, label, gamma_range=[0.6, 1.4], probability=1):
        self.ops = []
        self.probability = probability
        self.gamma_range = gamma_range
        self.info_idx = None
        for idx, s in enumerate(flatten_signature):
            if s in image:
                self.ops.append(lambda image, f: adjust_gamma(image, f))
                if self.info_idx is None:    
                    self.info_idx = idx 
            else:
                self.ops.append(lambda x, f: x)

    def run(self, item_flatten):
        if self.info_idx is not None: 
            occur = tf.less(tf.random.uniform([], 0, 1), self.probability)
            batch_size = item_flatten[self.info_idx].get_shape()[0] if item_flatten[self.info_idx].get_shape().ndims == 4 else 1
            gamma = tf.random.uniform([batch_size], self.gamma_range[0], self.gamma_range[1])
            item_flatten = [tf.cond(occur, lambda : op(item, gamma), lambda : item) for op, item in zip(self.ops, item_flatten)]
        return item_flatten