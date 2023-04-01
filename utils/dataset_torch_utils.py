import os
import cv2
import torch
import numpy as np
import io
import h5py
from PIL import Image
import torchvision

class Torch_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, resolution = (256, 256), type = "h5"):
        self.root_dir = root_dir
        self.resolution = resolution
        self.type = type
        if (self.type == "h5"):
            self.h5_file = h5py.File(root_dir, 'r')
            self.images = list(self.h5_file.keys())
        else:
            self.images = os.listdir(root_dir)  
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        if (self.type == "h5"):
            image = np.asarray(Image.open(io.BytesIO(np.array(self.h5_file[image_name]))))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_path = os.path.join(self.root_dir, image_name)
            image = cv2.imread(image_path)

        cv2.resize(image, self.resolution)
        return torchvision.transforms.ToTensor()((image / 127.5) - 1).to(torch.float32)