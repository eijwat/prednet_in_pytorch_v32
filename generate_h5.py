import os
import argparse

import numpy as np
import h5py

usage = 'Usage: python {} INPUT_FILE [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate hickle file from images.',
                                 usage=usage)
parser.add_argument('image_list', action='store', nargs=None, 
                    type=str, help='Input image list file.')
parser.add_argument('-c', '--output', action='store', nargs=None, 
                    type=str, default='output.h5', help='Output hickle file.')

args = parser.parse_args()
hf = h5py.File(args.output, 'a')
image_files = open(args.image_list).read().splitlines()

for f in image_files:
    print(f)
    binary_data = open(f, 'rb').read()
    hf.create_dataset(os.path.basename(f), data=np.asarray(binary_data))
    hf.flush()

hf.close()
