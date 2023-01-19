import random
import numpy as np
import cv2
from random import shuffle

class Chess(object):

    def __init__(self, 
                 image_size=(512,512),
                 period=32,
                 radius=14,
                 shapes=[3,4,5,6,'inf'],
                 patten_translation=False,
                 patten_rotation=False,
                 object_rotation=False):

        assert radius < period
        self.image_size = image_size
        self.period = period
        self.radius = radius
        self.shapes = shapes
        self.patten_translation = patten_translation
        self.patten_rotation = patten_rotation
        self.object_rotation = object_rotation

    def generate(self, number):
        
        images = []
        for i in range(number):
            
            image = np.zeros((*self.image_size, 1), np.uint8)

            x = list(range(0, int(self.image_size[1] * 1.5), self.period))
            y = list(range(0, self.image_size[0], self.period))
            y_reverse = [-c for c in reversed(y)]
            y = y_reverse[:-1] + y
            coords = np.array(np.meshgrid(x, y, indexing='xy'))
            coords = coords.reshape((2,-1)) + self.period // 2
            coords = np.moveaxis(coords, 0, -1)
            # pattern rotation
            if self.patten_rotation:
                A = random.uniform(0, np.pi/2)
                R = np.array([[np.cos(A), -np.sin(A)],[np.sin(A), np.cos(A)]])
                coords = np.matmul(coords, R.T)
            # pattern translate
            if self.patten_translation:
                start = np.array([[random.randint(0, self.period), random.randint(0, self.period)]])
                coords = coords + start

            # coords = coords[coords[:,0]>0]
            # coords = coords[coords[:,0]<self.image_size[1]]
            # coords = coords[coords[:,1]>0]
            # coords = coords[coords[:,1]<self.image_size[0]]

            S = self.shapes[random.randint(0, len(self.shapes)-1)]
            S = 3 if S < 3 else S
            S = 32 if S > 32 else S
            coords_obj = np.array([[np.cos(i*2*np.pi/S - np.pi/2), np.sin(i*2*np.pi/S- np.pi/2)] for i in range(S)]) * self.radius

            if self.object_rotation:
                A = random.uniform(0, 2*np.pi/3)
                R = np.array([[np.cos(A), -np.sin(A)],[np.sin(A), np.cos(A)]])
                coords_obj = np.matmul(coords_obj, R.T)

            polygons = np.array([C + coords_obj for C in coords])
            # idxX = np.logical_and(np.all(polygons[:,:,0] > 0, axis=1), np.all(polygons[:,:,0] < self.image_size[1], axis=1))
            # idxY = np.logical_and(np.all(polygons[:,:,1] > 0, axis=1), np.all(polygons[:,:,1] < self.image_size[0], axis=1))
            # idx = np.logical_and(idxX, idxY)
            # polygons = polygons[idx]

            cv2.fillPoly(image, pts=np.round(polygons).astype(np.int32), color=(255))

            # image[coords[:,1], coords[:,0]] = 255
            images.append(image)
        
        return images

class Cluster(object):

    def __init__(self, 
                 image_size=(384,384),
                 period=32,
                 radius=14,
                 shapes=[3,4,5,6,'inf'],
                 pattern=[1,2,3],
                 distance=1,
                 patten_rotation=False,
                 object_rotation=False):

        assert radius < period
        self.image_size = image_size
        self.period = period
        self.radius = radius
        self.shapes = shapes
        self.distance = distance
        self.pattern = pattern
        self.patten_rotation = patten_rotation
        self.object_rotation = object_rotation

    def generate(self, number, position=None):
        
        Gh, Gw = self.image_size[0] // self.period, self.image_size[1] // self.period

        images = []
        for i in range(number):
            
            image = np.zeros((*self.image_size, 1), np.uint8)

            if position is None:
                Hmin = random.randint(0, Gh-self.distance-1)
                Wmin = random.randint(0, Gw-self.distance-1)
            else:
                Hmin = min(int(position[0]), Gh-self.distance-1)
                Wmin = min(int(position[1]), Gw-self.distance-1)
                
            Hmin = Hmin * self.period + self.period // 2 
            Wmin = Wmin * self.period + self.period // 2 

            coords = np.array([[Wmin, Hmin], [Wmin+self.period*self.distance, Hmin], [Wmin+self.period*self.distance, Hmin+self.period*self.distance], [Wmin, Hmin+self.period*self.distance]])
            if isinstance(self.pattern, list):
                coords = coords[self.pattern]
            else:
                idx = list(range(4))
                shuffle(idx)
                coords = coords[idx[:self.pattern]]

            # pattern rotation
            if self.patten_rotation:
                A = random.uniform(0, np.pi/2)
                R = np.array([[np.cos(A), -np.sin(A)],[np.sin(A), np.cos(A)]])
                coords = np.matmul(coords, R.T)


            S = self.shapes[random.randint(0, len(self.shapes)-1)]
            S = 3 if S < 3 else S
            S = 32 if S > 32 else S
            coords_obj = np.array([[np.cos(i*2*np.pi/S - np.pi/2), np.sin(i*2*np.pi/S- np.pi/2)] for i in range(S)]) * self.radius

            if self.object_rotation:
                A = random.uniform(0, 2*np.pi/3)
                R = np.array([[np.cos(A), -np.sin(A)],[np.sin(A), np.cos(A)]])
                coords_obj = np.matmul(coords_obj, R.T)

            polygons = np.array([C + coords_obj for C in coords])

            cv2.fillPoly(image, pts=np.round(polygons).astype(np.int32), color=(255))
            images.append(image)
        
        return images

if __name__ == "__main__":
    import cv2
    from skimage.morphology import dilation, disk
    g = RepeatShape(image_size=(384, 384), shapes=[32])
    images = g.generate(number=1)
    # g = Cluster(image_size=(384,384), shapes=[32], pattern=[1,2,3], distance=5)
    # images = g.generate(number=1, position=(3,3))
    for idx in range(len(images)):
        cv2.imwrite('./three55_'+str(idx)+'.png', images[idx])