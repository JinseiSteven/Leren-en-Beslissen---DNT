#!/usr/bin/env python3

"""
Converts the 'Dataset_maker_faire' annotation data files to the
'SPLObjDetectDatasetV2' dataset its data format.

See https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format for
more info about the format specifie for the SPLObjDetectDatasetV2 dataset.

Usage: ./convert_dataset_annotations.py file1.xml file2.xml ... fileN.xml
"""

import sys
import os

try:
    import xmltodict
except ImportError:
    print('Package `xmltodict` not installed, run `pip3 install xmltodict`')
    sys.exit(1)

CLASS_IDS = {
    'ball': 0,
    'robot': 1,
    'goalpost': 2,
    'goalspot': 3,
}

def convert_files(filepaths: list[str]) -> None:
    total_success_files = 0
    total_filepaths = len(filepaths)

    for index, filepath in enumerate(filepaths):
        print(f'[{index + 1}/{total_filepaths}] {filepath}')

        # Read XML file and convert to python dict
        xml_dict = xmltodict.parse(open(filepath, 'r').read())

        # If there are no annotations, remove the file immediately
        if 'object' not in xml_dict['annotation']:
            print('No detected objects present, removing file')
            os.remove(filepath)
            continue

        total_success_files += 1

        # Get the full image width
        image_width = int(xml_dict['annotation']['size']['width'])
        image_height = int(xml_dict['annotation']['size']['height'])

        # Sometimes it's one dict, sometimes an array of dict.
        # Convert a single dict to an array with one item such that
        # we always have arrays.
        if not isinstance(xml_dict['annotation']['object'], list):
            xml_dict['annotation']['object'] = [xml_dict['annotation']['object']]

        # Loop over each detected object annotation
        for obj in xml_dict['annotation']['object']:
            # The new dataset does not contain the 'centerspot' class, 
            # therefore we ignore it.
            if obj['name'] == 'centerspot':
                continue

            # Get the numerical class id for the new dataset based on the name
            # of the current dataset.
            class_id = CLASS_IDS[obj['name']]

            # Calculate the bounding box width and height.
            bbox_width = int(obj['bndbox']['xmax']) - int(obj['bndbox']['xmin'])
            bbox_height = int(obj['bndbox']['ymax']) - int(obj['bndbox']['ymin'])

            # Calcualte the midpoint of the bounding box.
            # Divide by the image widht/height to scale it between 0 and 1
            midpoint_x = (int(obj['bndbox']['xmin']) + (bbox_width / 2)) / image_width
            midpoint_y = (int(obj['bndbox']['ymin']) + (bbox_height / 2)) / image_height

            # Rescale the bounding box such that it's between 0 and 1
            bbox_width_scaled = bbox_width / image_width
            bbox_height_scaled = bbox_height / image_height

            contents = f'{class_id} {midpoint_x} {midpoint_y} {bbox_width_scaled} {bbox_height_scaled}'

            new_filepath = '{}.txt'.format(filepath.rsplit('.', 1)[0])
            new_file = open(new_filepath, 'w')
            new_file.write(contents)
            new_file.close()

    print(f'Converted {total_success_files} files successfully.')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'Usage: {sys.argv[0]} file1.xml file2.xml ... fileN.xml')
        sys.exit(1)

    convert_files(sys.argv[1:])
    print('Done')
    sys.exit(0)
