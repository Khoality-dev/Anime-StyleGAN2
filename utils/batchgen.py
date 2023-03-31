import io
import os
import random
import cv2
import numpy as np
from PIL import Image

from utils.imglib import load_sampling

def load_bfile(h5_file, name):
    img = np.asarray(Image.open(io.BytesIO(np.array(h5_file[name]))))
    img = cv2.cvtColor(cv2.resize(img, (256,256)), cv2.COLOR_RGB2BGR)
    return img

def load_bfiles(h5_file, names):
    img = []
    for name in names:
        img.append(load_bfile(h5_file, name))
    return img

class BatchGen():
    def __init__(self, path, img_size, batch_size, latent_size):

        self.path = path
        self.h5_file = None
        
        if os.path.isdir(path):
            self.keys = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        else:
            self.h5_file = h5py.File(path, 'r')
            self.keys = self.h5_file.keys()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.latent_size = latent_size
        return
    
    def next_batch_Generator(self):
        z = np.random.normal(size = (self.batch_size, self.latent_size))
        noise = np.random.normal(size = (self.batch_size, self.img_size[0], self.img_size[1],1))
        return z, noise

    def next_batch_Discriminator(self):
        real_samples = random.sample(self.keys, self.batch_size)
        if (self.h5_file != None):
            real_samples = np.array(load_bfiles(self.h5_file, real_samples)) / 127.5 - 1
        else:
            real_samples = np.array(load_sampling(self.path, real_samples, H = self.img_size[0], W = self.img_size[1]))
            
        z = np.random.normal(size = (self.batch_size, self.latent_size))
        noise = np.random.normal(size = (self.batch_size, self.img_size[0], self.img_size[1],1))

        return real_samples, z, noise

    def get_size(self):
        return len(self.keys)