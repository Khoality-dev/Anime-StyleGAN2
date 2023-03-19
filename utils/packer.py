#Credit for https://blade6570.github.io/soumyatripathy/hdf5_blog.html

import argparse
import os
from alive_progress import alive_bar
import h5py
import numpy as np

from utils.imglib import load_image, load_images

def packer(args):
    data_dir = args.src
    target_h5 = args.dst
    img_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and (os.path.splitext(f)[1] == '.png' or os.path.splitext(f)[1] == '.jpg')]
    h5_file = h5py.File(target_h5, "w")
    print("Packing up dataset...")
    with alive_bar(len(img_files)) as bar:
        h5_file.create_dataset('mode', data = args.mode)
        for file_name in img_files:
            path = os.path.join(data_dir, file_name)
            if (args.mode == 1):
                with open(path, "rb") as img_f:
                    data = img_f.read()
            else:
                data = load_image(path)
            h5_file.create_dataset(file_name, data = np.array(data))
            bar()
    print("Done!")
    h5_file.close()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest = 'src', default = "Dataset")
    parser.add_argument('-d', dest = 'dst', default = "Dataset.h5")
    parser.add_argument('-m', dest = 'mode',help='Write Mode, 0: png, 1: binary (default: 1)', default = 1)
    args = parser.parse_args()
    packer(args)