# NeurIPS 2025 Submission 12267
## An Overview of Prototype Formulations for Interpretable Deep Learning

This is an anonymous code repository for NeurIPS 2025 12267

# Installation
This repository is built on pytorch and pytorch-lightning. We provide an installation script.
```bash
# Assumes CONDA is installed
bash ./install.sh
```

This repository will install all the required dependencies.
In case you are not using conda, please install the packages as listed in the script.


## Dataset Preparation
Unpack the datasets into the following structure:
```
./data/
├── CUB_200_2011/
│   ├── images/
│   ├── bounding_boxes.txt
│   ├── classes.txt
│   ├── image_class_labels.txt
│   ├── train_test_split.txt
│   └── ... any other cub200 files
├── OxfordFlowers102/
│   └── flowers-102/
│       ├── jpg/
│       ├── iamgelabels.mat
│       └── setid.mat
└── stanford_cars/
    ├── cars_test/
    ├── cars_train/
    └── cars_test_annos_withlabels.mat
```

Use the script in `./utils` to create bounding_box crops and offline augmentations akin to ProtoPNet.

# Usage Guide
This repository uses `ml-collections` for hyperparameter configuration. You can run an experiment with

```bash
python main.py --config configs/<CONFIG_FILE>.py
```

We use ml-collections CLI features to overwrite config parameters. For additional examples see `./multi_run.sh`
```bash
python main.py --config configs/hyperpg_resnet50_birds.py --config.model.pm_name=gaussian --config.training.num_epochs=30 --config.seed=1337
```

# Implementation Overview
We aim for a modular implementation. The Prototype Formulations are implemented in `prob_proto.models.prototype_modules`
The training loop itself is implemented with pytorch lightning in `prob_proto.pl_training`
