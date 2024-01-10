# Leren en Beslissen DNT

Machine learning project focussing on SPL specific object detection using YOLOv8 and additional optimizations. This is a project conducted by students from the University of Amsterdam.

# Installation

```
$ git clone https://github.com/JinseiSteven/Leren_en_Beslissen_DNT.git
$ cd Leren_en_Beslissen_DNT

# Run this if you want to create a virtual env for the dependencies
$ python3 -m venv env
$ source ./env/bin/activate

$ pip3 install -r requirements.txt
```

# Datasets

To train the model with the dataset, please request the `custom` dataset from
one of the authors mentioned at the [Authors](#authors) section.

- Download the `customs.zip` and unzip
- Put the custom/ folder inside the datasets/ folder in the git repo
- Run `./scripts/data_splitter.py --sourcepath datasets/custom/`

Continue to the [Usage](#usage) section.

# Usage

### `train.py`

This file can be used to train the neural network with [YOLOv8](https://github.com/ultralytics/ultralytics).

Make sure to download one of the `YOLOv8x` models specified at the GitHub page
and place it inside the root of this repository.

Example run:

```
$ ./train.py --yolo-config=./yolov8n.yaml --train-config=./config/train_custom.yaml --epochs=3
```

**NOTE:** Apple Silicon users can specify `--device=mps`, see [Apple M1 and M2 MPS Training](https://docs.ultralytics.com/modes/train/#apple-m1-and-m2-mps-training)

See `./train.py --help` for all possible arguments.

# Authors

- Joost Weerheim (13769758)
- Kim Koomen (14621312)
- Ross Geurts (14599996)
- Stephan Visser (13977571)
