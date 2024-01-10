
'''
Image Augmenter class used for generating augmented images from a dataset.
WIP WIP WIP

Usage: data_augmentor.py [--sourcepath ./path] --augments str [str... ]
'''

import argparse
import os
from pathlib import Path
from PIL import Image
from random import shuffle
from time import sleep
try:
    from torch import Tensor
    from torchvision.tv_tensors import BoundingBoxes
    from torchvision.transforms import v2
    from tqdm import tqdm
except ImportError as e:
    print(f"\nError\n-----\n{e}.\n")
    exit(1)


class Image_Augmentor:
    """
    A class used for generating augmented images from a dataset.

    ...

    Attributes
    ----------
    sourcepath : Path
        path to the dataset folder in which both an images and a labels folder resides
    augmentations : dict[str, torchvision.transform]
        dictionary containing all available augmentations:
            - colorjitter
            - gaussian_blur
            - adjust_sharpness
            - posterize
            - random_rotation
    images : list[Path]
        list containing paths to all image files in the 'images' folder

    Methods
    -------
    apply_augmentations(self, augmentations, prefix=None, ratio=0.1):
        Applies the chosen augmentations to the specified ratio of images. 
        Saves the results to a new subdirectory with the chosen prefix.
    new_dir(self, destination):
        Creates a new folder at the specified path, relative to the sourcepath.
    yolo_to_bbox(self, x, y, w, h, W, H):
        Converts a boundary box from YOLOv8 format to a standard (x, y, x, y) format.
    bbox_to_yolo(self, x1, y1, x2, y2, W, H):
        Converts a boundary box from a standard (x, y, x, y) format to YOLOv8 format.
    """

    def __init__(self, sourcepath):
        """
        Parameters
        ----------
        sourcepath : Path
            path to the dataset folder in which both an images and a labels folder resides
        """
        self.sourcepath = sourcepath
        self._augmentations = {'colorjitter': v2.ColorJitter(brightness=(.5), saturation=(0.5,1.5),hue=(-0.5,0.5)), 
                              'gaussian_blur': v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)), 
                              'adjust_sharpness': v2.RandomAdjustSharpness(sharpness_factor=5),
                              'posterize': v2.RandomPosterize(bits=5),
                              'random_rotation': v2.RandomRotation(degrees=(0, 180))}

        images = [os.path.join(sourcepath, 'images', file) for file in os.listdir(os.path.join(sourcepath, 'images')) 
                  if file.endswith(('.png', '.jpg', '.jpeg'))]
        shuffle(images)

        self._images = images

    def apply_augmentations(self, augmentations, prefix=None, ratio=0.1):
        """
        Applies the chosen augmentations to the specified ratio of images. 
        Saves the results to a new subdirectory with the chosen prefix.

        If the argument `prefix` isn't passed in, the combination of augmentation names is used.
        If the argument `ratio` isn't passed in, a ratio of 10% is used.

        Parameters
        ----------
        augmentations : list[str]
            A list of all augmentations to be applied.
        prefix : str, optional
            Prefix to be used for the images and the images and labels folders.
        ratio : float, optional
            The ratio of images to be augmented
        """

        # combining the different augmentations chosen
        transform = v2.Compose(list(set([self._augmentations[x] for x in augmentations])))

        # checking if a prefix has been given, if not: prefix will be names of all augmentations
        if not prefix:
            prefix = '_'.join(sorted(augmentations))

        # creating new directories for the images and labels
        self.new_dir(os.path.join('images', prefix))
        self.new_dir(os.path.join('labels', prefix))

        # iteratively creating N * ratio new images
        total_images = int(len(self._images) * ratio)
        for ix, _ in zip(range(total_images), tqdm(range(total_images), desc="Augmenting Images... ")):
            
            # splitting the path to create the image and label paths
            split_path = os.path.split(self._images[ix])
            image_path = os.path.join(split_path[0], prefix, prefix + '_' + split_path[1])
            label_path = os.path.join(self.sourcepath, 'labels', prefix, prefix + '_' + Path(split_path[1]).stem + '.txt')
            old_label_path = os.path.join(self.sourcepath, 'labels', Path(split_path[1]).stem + '.txt')
            
            # skipping any files which don't have a corresponding label file
            if not os.path.exists(old_label_path):
                continue

            # opening the image and assessing the size
            img = Image.open(self._images[ix]) 
            H, W = v2.functional.get_size(img)

            # opening the corresponding labels file and extracting the boundary boxes (converted from YOLOv8 format)
            ids, boxes = [], []
            with open(old_label_path) as file:
                for bbox in file:
                    id, x, y, w, h = bbox.split(' ')
                    boxes.append(self.yolo_to_bbox(float(x), float(y), float(w), float(h), W, H))
                    ids.append(id)

            boxes = BoundingBoxes(Tensor(boxes), format="XYXY", canvas_size=(H, W))

            # applying the transformation on both the image and the boundary boxes
            transformed_image, transformed_boxes = transform(img, boxes)

            # saving the image inside the newly created directory
            transformed_image.save(image_path)

            # creating a new .txt file containing the new boundary boxes (YOLOv8 format)
            with open(label_path, 'w') as label_file:
                for id, box in zip(ids, transformed_boxes.tolist()):
                    label_file.write(id)
                    box = self.bbox_to_yolo(*box, W, H)
                    for element in box:
                        label_file.write(" " + f"{element:.8f}")
                    label_file.write('\n')

    def new_dir(self, destination):
        """
        Creates a new folder at the specified path, relative to the sourcepath.

        Parameters
        ----------
        destination : Path
            The destination path where the folder should be created.
        """
        folder_path = os.path.join(self.sourcepath, destination)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    # converting the YOLO format to bounding boxes
    def yolo_to_bbox(self, x, y, w, h, W, H):
        """
        Converts a boundary box from YOLOv8 format to a standard (x, y, x, y) format.

        Parameters
        ----------
        x : float
            The x-coordinate of the center of the boundary box. Normalized.
        y : float
            The y-coordinate of the center of the boundary box. Normalized.
        w : float
            Boundary box width. Normalized.
        h : float
            Boundary box height. Normalized.
        W : int
            Width of the image.
        H : int
            Height of the image.
        """
        x1, x2 = (x - (w / 2.)) * W, (x + (w / 2.)) * W
        y1, y2 = (y - (h / 2.)) * H, (y + (h / 2.)) * H
        return [x1, y1, x2, y2]
    
    # converting the bounding boxes to YOLO format
    def bbox_to_yolo(self, x1, y1, x2, y2, W, H):
        """
        Converts a boundary box from YOLOv8 format to a standard (x, y, x, y) format.

        Parameters
        ----------
        x1 : float
            The x-coordinate of the bottom-left corner of the boundary box.
        y1 : float
            The y-coordinate of the bottom-left corner of the boundary box.
        x2 : float
            The x-coordinate of the top-right corner of the boundary box.
        y2 : float
            The y-coordinate of the top-right corner of the boundary box.
        W : int
            Width of the image.
        H : int
            Height of the image.
        """
        x = ((x1 + x2) / 2.0) / W
        y = ((y1 + y2) / 2.0) / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        return [x ,y ,w ,h]


def main(args):
    Augmentor = Image_Augmentor(args.sourcepath)
    Augmentor.apply_augmentations(args.augments, prefix=args.prefix, ratio=args.ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcepath', type=Path, default='.', help="the folder where the images folder resides (default: current directory)", )
    parser.add_argument('--prefix', type=str, help="the prefix to be used for the the augmented images and the augmented image and label folders")
    parser.add_argument('--augments', required=True, choices=['colorjitter', 'gaussian_blur', 'adjust_sharpness', 'posterize', 'random_rotation'], nargs='+', help="the type of augmentation to be applied to the images")
    parser.add_argument('--ratio', type=float, default=0.1, help="the ratio of images on which the augmentation will be applied (default: 0.1)")
    args = parser.parse_args()

    args.augments = list(set(args.augments))
    args.prefix = args.prefix.replace(" ", "")
    if not os.path.exists(os.path.join(args.sourcepath, 'images')):
        print(f"\nError\n-----\nNo 'images' folder found in path: {args.sourcepath}.\n")
        exit()

    main(args)