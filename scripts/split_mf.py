#!/usr/bin/env python3


import numpy as np
import sys
from os import listdir
from os.path import isfile, join, realpath
from pathlib import Path
from shutil import copyfile


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} MF_DATA_DIR TRAIN:TEST:VAL')
        sys.exit(1)

    data_dir = sys.argv[1].rstrip('/')

    split_ratios = list(map(lambda x: int(x) / 100, sys.argv[2].split(':')))
    files = np.array([realpath(join(data_dir, f)) for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('.jpg')])

    # Shuffle the files and create indices to get random items.
    np.random.shuffle(files)
    split_indices = (np.cumsum(split_ratios) * len(files)).astype(int)[:-1]

    # Create three subsets: train, val, test respectively.
    split_data = np.split(files, split_indices)

    data_names = {
        0: 'train',
        1: 'val',
        2: 'test'
    }

    for index, subset in enumerate(split_data):
        subset_name = data_names[index]
        filename = join(data_dir, subset_name)
        with open(f'{filename}.txt', 'w') as file:
            print(f'Created {file.name}')
            file.write('\n'.join(subset) + '\n')
            file.close()
