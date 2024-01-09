#!/usr/bin/env python3

from ultralytics import YOLO, settings
import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments and return them.
    """
    parser = argparse.ArgumentParser(
        prog = 'DNT train.py',
        description = 'Main file to train the neural network.',
        epilog = 'This program is used for the Leren & Beslissen course at the University of Amsterdam.')

    parser.add_argument('--yolo-config',
                        required=True,
                        help='YAML YOLO config passed to model()')

    parser.add_argument('--train-config',
                        required=True,
                        help='YAML training config passed to model.train()')

    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs to train for')

    parser.add_argument('--device',
                        help='Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')

    return parser.parse_args()

if __name__ == '__main__':
    settings.update({
        'runs_dir': ROOT_DIR,
        'datasets_dir': os.path.join(ROOT_DIR, 'datasets')
    })

    args = parse_arguments()

    model = YOLO(args.yolo_config)
    results = model.train(data=args.train_config, epochs=args.epochs, device=args.device)
    metrics = model.val()
