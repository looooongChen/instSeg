import cv2
import numpy as np
import random
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

OBJ_SPEC = {'circle': {'radius': 12},
            'square': {'side': 20},
            'ellipse': {'major': 12, 'minor': 6},
            'triangle': {'side': 20}}

# OBJ_SPEC = {'circle': {'radius': 12},
#             'square': {'side': 20},
#             'ellipse': {'major': 12, 'minor': 6},
#             'triangle': {'side': 20}}

class Config(object):

    def __init__(self):

        self.img_sz = (512, 512)
        self.grid_sz = (64, 64)
        self.type = 'grid'

        self.coord_sz = (64,64)
        self.coord_resolution = (1,1)

        self.obj_spec = OBJ_SPEC
        self.obj_color = (255,0,0) # (r,g,b) or 'random'
        self.bg_color = (0,0,0) # (r,g,b)

        self.obj_shape = None
        self.obj_rotation = False
        self.obj_translation = False
        self.grid_rotation = False
        self.grid_translation = False

        self.max_degree = 10


class ObjGenerator(object):

    '''
    allow random shape, random rotation, random color
    '''

    def __init__(self, obj_spec=OBJ_SPEC, patch_sz=(16, 16), bg_color=(0,0,0)):
        '''
        sz: scala, size of the patch for the object
        shape: 'random', one of ['circle', 'square', 'triangle'] or a list of these options
        obj_color: color of the object
        bg_color: color of the background
        '''
        self.obj_spec = obj_spec
        self.patch_sz = patch_sz
        self.bg_color = bg_color

    def generate(self, shape=None, color='random', rotation=False, translate=False):
        
        img = np.zeros((self.patch_sz[0], self.patch_sz[1], 3), np.uint8)
        for i in range(3):
            img[:,:,i] += self.bg_color[i]
        obj_color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)] if color is None else color

        if shape is None:
            shape = random.choice(list(self.obj_spec.keys()))
        
        # opencv coordinate system (x,y), x for column, y for row
        C = [self.patch_sz[1]//2, self.patch_sz[0]//2]
        # if shape == 'random':
        #     D = random.randint(3, self.config.max_degree)
        #     step = 2 * math.pi / D 
        #     start_angle = random.uniform(0, 2 * math.pi)

        #     contours = []
        #     for idx in range(D):
        #         angle = random.uniform(start_angle + idx * step, start_angle + (idx+1) * step)
        #         length = random.uniform(0.55 * R, 0.95 * R)
        #         contours.append([int(length * math.cos(angle)) + R, int(length * math.sin(angle)) + R])

        #     cv2.fillPoly(obj, pts = [np.array(contours)], color=obj_color)

        if translate is not False:
            C[0], C[1] = C[0] + random.randint(0, translate), C[1] + random.randint(0, translate)

        if rotation is False:
            angle = 0
        elif rotation == 'random':
            angle = random.randint(0, 259) 
        elif isinstance(rotation, int):
            angle = random.choice(list(range(0,361,rotation)))
        else:
            angle = 0

        if shape == 'circle':
            R = self.obj_spec[shape]['radius']
            cv2.circle(img, C, int(R), obj_color, thickness=-1)
        elif shape == 'ellipse':
            Rmajor, Rminor = self.obj_spec[shape]['major'], self.obj_spec[shape]['minor']
            cv2.ellipse(img, C, (Rmajor, Rminor), angle, 0, 360, obj_color, thickness=-1)
        else: 
            if shape == 'square':
                L = self.obj_spec[shape]['side'] // 2
                coords = np.array([[-L, -L], [-L, L], [L, L], [L, -L]])
            elif shape == 'triangle':
                L = self.obj_spec[shape]['side']
                coords = [[0, -3**0.5/3*L], [L/2, 3**0.5/6*L], [-L/2, 3**0.5/6*L]]
            if angle != 0:
                angle = np.radians(angle)
                Rotate = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
                coords = np.matmul(Rotate, coords.T).T
            coords = coords + np.expand_dims(np.array(C), axis=0)
            coords = coords.astype(np.int64)
            cv2.fillPoly(img, pts=[coords], color=obj_color)

        return img



class GridGenerator(object):

    def __init__(self, config):
        self.config = config
        self.obj_generator = ObjGenerator(obj_spec=config.obj_spec, patch_sz=config.grid_sz, bg_color=config.bg_color)
    

    def generate(self, num=None):

        grid_sz = self.config.grid_sz
        img_sz = self.config.img_sz

        if self.config.type == 'grid':
            tmp_sz = [int((2 * img_sz[0] // grid_sz[0] + 2) * grid_sz[0]), int((2 * img_sz[1] // grid_sz[1] + 2) * grid_sz[1])]
            img = np.zeros((tmp_sz[0], tmp_sz[1], 3), np.uint8)
            for i in range(tmp_sz[0] // grid_sz[0]):
                for j in range(tmp_sz[1] // grid_sz[1]):
                    obj = self.obj_generator.generate(shape=self.config.obj_shape, color=self.config.obj_color, rotation=self.config.obj_rotation, translate=self.config.obj_translation)
                    img[grid_sz[0]*i:grid_sz[0]*(i+1), grid_sz[1]*j:grid_sz[1]*(j+1), :] = obj

            if self.config.grid_rotation:
                rotate_matrix = cv2.getRotationMatrix2D(center=(tmp_sz[1]/2, tmp_sz[0]/2), angle=random.randint(0,360), scale=1)
                img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(tmp_sz[1], tmp_sz[0]))
            
            if self.config.grid_translation:
                Ox, Oy = tmp_sz[0]//2 + random.randint(- grid_sz[0], grid_sz[0]), tmp_sz[1]//2 + random.randint(- grid_sz[1], grid_sz[1])
            else:
                Ox, Oy = tmp_sz[0]//2, tmp_sz[1]//2
            img = img[Ox - img_sz[0] // 2 : Ox + (img_sz[0] - img_sz[0] // 2), Oy - img_sz[1] // 2 : Oy + (img_sz[1] - img_sz[1] // 2)]

            return img
        
        elif self.config.type == 'pair':
            img = np.zeros((img_sz[0], img_sz[1], 3), np.uint8)
            obj = self.obj_generator.generate(shape=self.config.obj_shape, color=self.config.obj_color, rotation=self.config.obj_rotation, translate=self.config.obj_translation)
            i, j =  random.randint(0, img_sz[0] // grid_sz[0] - 1), random.randint(0, img_sz[1] // grid_sz[1] - 2)
            img[grid_sz[0]*i:grid_sz[0]*(i+1), grid_sz[1]*j:grid_sz[1]*(j+1), :] = obj
            img[grid_sz[0]*i:grid_sz[0]*(i+1), grid_sz[1]*(j+1):grid_sz[1]*(j+2), :] = obj

            return img
        
        elif self.config.type == 'quartet':
            img = np.zeros((img_sz[0], img_sz[1], 3), np.uint8)
            obj = self.obj_generator.generate(shape=self.config.obj_shape, color=self.config.obj_color, rotation=self.config.obj_rotation, translate=self.config.obj_translation)
            i, j =  random.randint(0, img_sz[0] // grid_sz[0] - 1), random.randint(0, img_sz[1] // grid_sz[1] - 4)
            for k in range(4):
                img[grid_sz[0]*i:grid_sz[0]*(i+1), grid_sz[1]*(j+k):grid_sz[1]*(j+1+k), :] = obj

            return img

        elif self.config.type == 'stain':
            coord_sz = self.config.coord_sz
            coord_resolution = self.config.coord_resolution
            
            assert coord_sz[0] <= img_sz[0] and coord_sz[1] <= img_sz[1]
            img = np.zeros((img_sz[0], img_sz[1], 3), np.uint8)
            coord = np.zeros((img_sz[0], img_sz[1], 2), np.int32)
            
            # sampling
            Nrow, Ncol = img_sz[0]//grid_sz[0], img_sz[1]//grid_sz[1]
            num = int(Nrow*Ncol*0.01) if num is None else num
            num = min(max(num, 1), Nrow*Ncol)
            pt = np.squeeze(np.array(random.sample(range(Nrow*Ncol), num)))
            xx, yy = pt // Nrow, pt % Ncol
            if grid_sz[0] != 1:
                xx = xx * grid_sz[0] + np.random.randint(0, grid_sz[0], size=xx.shape)
            if grid_sz[1] != 1:
                yy = yy * grid_sz[1] + np.random.randint(0, grid_sz[1], size=xx.shape)

            img[xx, yy, :] = self.config.obj_color
            coord[xx, yy, :] = np.array([xx % self.config.coord_sz[0] + 1, yy % self.config.coord_sz[1] + 1]).T

            if coord_resolution[0] != 1 or coord_resolution[1] != 1:
                coord = block_reduce(coord, (coord_resolution[0], coord_resolution[1], 1), np.max)

            return img, coord


            

        




if __name__ == "__main__":

    config = Config()
    # config.obj_shape = 'square'
    # config.obj_rotation = True
    # config.obj_translation = False
    # config.grid_rotation = False
    # config.grid_translation = False

    config.img_sz = (8, 8)
    config.grid_sz = (4, 4)
    config.coord_sz = (4, 4)
    config.coord_resolution = (4,4)
    config.type = 'stain'
    g = GridGenerator(config)

    img, coord = g.generate(4)

    print(img[:,:,0])

    print(coord[:,:,0])
    print(coord[:,:,1])

    # xx, yy = np.nonzero(img[:,:,0])
    # print(xx, coord[xx, yy, 0])
    # print(yy, coord[xx, yy, 1])




    

