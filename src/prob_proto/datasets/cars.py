import os

import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_factory import register_dataset


@register_dataset(name="cars")
class StanfordCarsDataset(torchvision.datasets.StanfordCars):
    def __init__(self, data_dir: str, train: bool = True, augment: bool = True):
        split = "train" if train else "test"

        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        if train and augment:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(224),
                ]
            )

        super().__init__(data_dir, split, transform, None, False)
