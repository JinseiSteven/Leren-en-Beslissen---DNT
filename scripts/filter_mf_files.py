#!/usr/bin/env python3

"""
The results below shows a run on this dataset using the train.py with epoch=1 
that shows the goal_post and pen_post classes have 0 accuracy and recall,
therefore we only want to gather the ball and robot files and put them inside
another folder.

#########################################################################
# Dataset_maker_faire train.py run with epoch=1
#########################################################################
Class        Images  Instances      Box(P          R      mAP50  mAP50-95)
  all           369        324      0.204     0.0702      0.058     0.0225
 ball           369         56      0.429      0.143      0.118     0.0322
robot           369        210      0.387      0.138      0.105      0.054
goal_post       369         32          0          0          0          0
pen_spot        369         26          0          0    0.00947     0.0036
"""

import sys
from os import listdir
from os.path import join
from pathlib import Path
from shutil import copyfile

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} DATASET_MAKER_FAIRE_PATH')
        sys.exit(1)

    data_dir = sys.argv[1]
    data_cleaned_dir = join(data_dir.rsplit('/', 1)[0], 'data_cleaned')

    # Make sure the new data_cleaned sibling directory exists.
    Path(data_cleaned_dir).mkdir(parents=True, exist_ok=True)
    Path(join(data_cleaned_dir, 'images')).mkdir(parents=True, exist_ok=True)
    Path(join(data_cleaned_dir, 'labels')).mkdir(parents=True, exist_ok=True)

    # Loop over all image files in the specified data_dir.
    files = [f for f in listdir(data_dir) if f.endswith(('.jpg', 'png', '.jpeg'))]
    for img_filename in files:
        # Open the corresponding .txt file
        try:
            txt_filename = f'{img_filename.rsplit(".", 1)[0]}.txt'
            lines = open(join(data_dir, txt_filename)).readlines()
        except FileNotFoundError:
            continue

        filtered_lines = []

        # Remove rows that are not class index 0 (ball) or robot (1)
        for line in lines:
            class_index = int(line.split(' ')[0])
            if class_index in [0, 1]:
                filtered_lines.append(line)

        # If we have some remaining lines left, then they are index 0 or 1,
        # and thus we can keep this file by copying it over to the data_cleaned.
        if len(filtered_lines) > 0:
            copyfile(join(data_dir, img_filename), join(data_cleaned_dir, 'images', img_filename))
            copyfile(join(data_dir, txt_filename), join(data_cleaned_dir, 'labels', txt_filename))
