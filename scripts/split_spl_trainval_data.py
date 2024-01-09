#!/usr/bin/env python3

"""
Takes all files in a given directory containing train-val data and splits them 
into separate folder. A ratio of 0.1 indicates 10% validation, 90% training.

Example:
$ ./scripts/split_spl_trainval_data.py ./datasets/SPLObjDetectDatasetV2/trainval 0.1
"""


import numpy as np
from os import listdir
from os.path import isfile, join
import sys
from pathlib import Path
from shutil import copyfile

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} TRAINVAL_INPUT_DIR SPLIT_RATIO')
        sys.exit(1)

    trainval_dir, split_ratio = sys.argv[1:]
    split_ratio = float(split_ratio)

    images_dir = join(trainval_dir, 'images')
    labels_dir = join(trainval_dir, 'labels')

    # Create new sibling folders called 'train' and 'val' relative to 
    # the 'trainval' folder.
    parent_dir = trainval_dir.rstrip('/').rsplit('/', 1)[0]
    out_dir_train = join(parent_dir, 'train')
    out_dir_val = join(parent_dir, 'val')
    for dirpath in [out_dir_train, out_dir_val]:
        # Create another images/ and labels/ folder inside each output dir
        Path(join(dirpath, 'images')).mkdir(parents=True, exist_ok=True)
        Path(join(dirpath, 'labels')).mkdir(parents=True, exist_ok=True)

    files = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]
    split_size = int(len(files) * split_ratio)
    validation_files_to_move = set(np.random.choice(files, size=split_size))

    # Move to validation folder along with corresponding labels
    for img_filename in files:
        out_dir = out_dir_val if img_filename in validation_files_to_move else out_dir_train
        label_filename = '{}.txt'.format(img_filename.rsplit('.', 1)[0])
        copyfile(join(images_dir, img_filename), join(out_dir, 'images', img_filename))
        copyfile(join(labels_dir, label_filename), join(out_dir, 'labels', label_filename))
