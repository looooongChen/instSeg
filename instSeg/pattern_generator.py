import cv2
import numpy as np
import random
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter


class Config(object):

    def __init__(self):

        self.img_sz = 512
        self.obj_sz = 64 # interger or 'random'
        self.max_degree = 10

        self.pattern = 'grid' # 'grid' or 'random'
        self.rotation = True
        self.translation = True
        self.shape = 'random'
        self.obj_color = (255,0,0) # None or (r,g,b)
        self.bg_color = (0,0,0) # None or (r,g,b)


class Generator(object):

    def __init__(self, config):
        self.config = config
    
    def generate_obj(self, sz, shape='random', obj_color=None, bg_color=None):
        '''
        sz: scala, size of the patch for the object
        shape: 'random', one of ['circle', 'square', 'triangle'] or a list of these options
        obj_color: color of the object
        bg_color: color of the background
        '''
        obj = np.zeros((sz, sz, 3), np.uint8)
        obj_color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)] if obj_color is None else obj_color
        if bg_color:
            for i in range(3):
                obj[:,:,i] += bg_color[i]
        R = sz // 2

        if isinstance(shape, list):
            shape = random.choice(shape)
        
        if shape == 'random':
            D = random.randint(3, self.config.max_degree)
            step = 2 * math.pi / D 
            start_angle = random.uniform(0, 2 * math.pi)

            contours = []
            for idx in range(D):
                angle = random.uniform(start_angle + idx * step, start_angle + (idx+1) * step)
                length = random.uniform(0.55 * R, 0.95 * R)
                contours.append([int(length * math.cos(angle)) + R, int(length * math.sin(angle)) + R])

            cv2.fillPoly(obj, pts = [np.array(contours)], color=obj_color)
        elif shape == 'circle':
            cv2.circle(obj, (R, R), int(0.8*R), obj_color, thickness=-1)
        elif shape == 'square':
            d = int(0.8*R) 
            contours = [[R-d, R-d], [R-d, R+d], [R+d, R+d], [R+d, R-d]]
            cv2.fillPoly(obj, pts = [np.array(contours)], color=obj_color)
        elif shape == 'triangle':
            d1, d2, d3 = int(0.8*R), int(0.5*0.8*R), int(0.866*0.8*R)
            contours = [[R, R-d1], [R+d3, R+d2], [R-d3, R+d2]]
            cv2.fillPoly(obj, pts = [np.array(contours)], color=obj_color)

        return obj

    def generate(self):

        obj_sz = self.config.obj_sz
        
        if self.config.pattern == 'grid':
            sz = int((2 * self.config.img_sz // obj_sz + 2) * obj_sz)
            img = np.zeros((sz, sz, 3), np.uint8)

            bg_color = self.config.bg_color if self.config.bg_color is not None else (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))

            for i in range(sz // obj_sz):
                for j in range(sz // obj_sz):
                    obj = self.generate_obj(obj_sz, shape=self.config.shape, obj_color=self.config.obj_color, bg_color=bg_color)
                    img[obj_sz*i:obj_sz*(i+1), obj_sz*j:obj_sz*(j+1), :] = obj
        elif self.config.pattern == 'random':
            pass

        if self.config.rotation:
            rotate_matrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=random.randint(0,360), scale=1)
            img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img.shape[1], img.shape[0]))
        
        if self.config.translation:
            Ox, Oy = img.shape[1]//2 + random.randint(- obj_sz, obj_sz), img.shape[1]//2 + random.randint(- obj_sz, obj_sz)
        else:
            Ox, Oy = img.shape[1]//2, img.shape[1]//2
        img = img[Ox - self.config.img_sz // 2 : Ox + (self.config.img_sz - self.config.img_sz // 2), Oy - self.config.img_sz // 2 : Oy + (self.config.img_sz - self.config.img_sz // 2)]
        
        
        return img




if __name__ == "__main__":

    config = Config()
    g = Generator(config)

    img = g.generate()
    cv2.imwrite('test.png', img.astype(np.uint8))

