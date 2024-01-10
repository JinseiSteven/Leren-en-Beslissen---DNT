

"""
Splits the dataset images into a training, validation and optional testing splits
and saves the file paths to .txt files, to be used with the Ultralytics YOLO format.

Usage: data_splitter.py [--sourcepath ./path] [--splits int [int... ]]
"""

import argparse
import numpy as np
import os
from pathlib import Path
from random import shuffle


def split_image_names(sourcepath, split):

    images = []
    for path, _, files in os.walk(sourcepath):
        for name in files:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join('./', os.path.relpath(path, sourcepath), name))

    # shuffle images and split threeways
    shuffle(images)
    return np.split(images, 
                    [int(len(images) * (split[0] / 100)), 
                     int(len(images) * ((split[0] + split[1]) / 100))])

def write_txt_files(sourcepath, split_name, image_list):
    
    # create new txt file and write image paths to it
    with open(os.path.join(sourcepath, split_name + '.txt'), 'w') as file:
        for _, image in enumerate(sorted(image_list)):
            file.write(image)
            file.write('\n')

def main(args):

    # split the total images based on given splits
    splits = split_image_names(args.sourcepath, args.splits)

    # create txt files: train, val and test
    for name, split in zip(['train', 'val', 'test'], splits):
        print(f"{name}.txt contains {len(split)} items.")
        write_txt_files(args.sourcepath, name, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcepath', type=Path, default='.', help="The folder where the images and labels folders reside (default: current directory).", )
    parser.add_argument('--splits', default=[70, 20, 10], nargs='+', type=int, help="The way the data should be split, into: training, validation and testing respectively. Testing split is optional")
    args = parser.parse_args()

    if len(args.splits) < 2 or len(args.splits) > 3:
        print("\nError\n-\n\nSplit list must contain 2 or 3 values.\n\n-\nFor help execute: data_splitter.py --help")
        exit()
    if (sum(args.splits) != 100):
        print("\nError\n-\n\nElements of split list must sum up to 100\n\n-\nFor help execute: data_splitter.py --help")
        exit()
    if not os.path.exists(os.path.join(args.sourcepath, 'images')):
        print("\nError\n-\n\nNo 'images' folder found in source folder.\n\n-\nFor help execute: data_splitter.py --help")
        exit()

    main(args)