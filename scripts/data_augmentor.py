
'''
Image Augmenter class used for generating augmented images from a dataset.
WIP WIP WIP

Usage: data_augmentor.py [--sourcepath ./path] --aug_type str [str... ]
'''

import argparse
import os
from pathlib import Path
from PIL import Image
from random import shuffle
try:
    from torchvision import transforms
except ImportError:
    print('Package `torchvision` not installed, run `pip3 install torchvision`')
    exit(1)


class Image_Augmentor:
    def __init__(self, sourcepath):
        self.sourcepath = sourcepath
        self.augmentations = {'colorjitter': transforms.ColorJitter(brightness=(.5), saturation=(0.5,1.5),hue=(-0.5,0.5)), 
                              'gaussian_blur': transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
                              'adjust_sharpness': transforms.RandomAdjustSharpness(sharpness_factor=5),
                              'posterize': transforms.RandomPosterize(bits=5),
                              'random_rotation': []}

        images = [os.path.join(sourcepath, 'images', file) for file in os.listdir(os.path.join(sourcepath, 'images')) 
                  if file.endswith(('.png', '.jpg', '.jpeg'))]
        shuffle(images)

        self.images = images
    
    def new_dir(self, prefix):
        folder_path = os.path.join(self.sourcepath, 'images', prefix)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)


    def apply_augmentations(self, augmentations, ratio=1):
        transform = transforms.Compose(list(set([self.augmentations[x] for x in augmentations])))

        prefix = '_'.join(sorted(augmentations))
        
        self.new_dir(prefix)
        
        for ix in range(len(self.images) * ratio):
            img = Image.open(self.images[ix])
            transformed_image = transform(img)

            split_path = os.path.split(self.images[ix])
            transformed_image.save(os.path.join(split_path[0], prefix, prefix + split_path[1]))
            print(os.path.join(split_path[0], prefix, prefix + split_path[1]))


def main(args):
    Augmentor = Image_Augmentor(args.sourcepath)
    Augmentor.apply_augmentations(args.aug_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcepath', type=Path, default='.', help="The folder where the images folder resides (default: current directory).", )
    parser.add_argument('--aug_type', required=True, choices=['colorjitter', 'gaussian_blur', 'adjust_sharpness', 'posterize'], nargs='+', help="The type of augmentation to be applied to the images.")
    parser.add_argument('--ratio', type=float, help="The percentage of images on which the augmentation will be applied.")
    args = parser.parse_args()

    args.aug_type = list(set(args.aug_type))
    if not os.path.exists(os.path.join(args.sourcepath, 'images')):
        print("\nError\n-----\nNo 'images' folder found in source folder.\n")
        exit()

    main(args)