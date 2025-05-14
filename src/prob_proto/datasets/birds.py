import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_factory import register_dataset


@register_dataset(name="birds")
class Cub200Dataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, augment: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images_crop")

        if "images" in data_dir:
            self.image_dir = data_dir
            self.data_dir = os.path.join(data_dir, "..")

        self.is_train = train

        self.transf = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        if self.is_train and augment:
            self.transf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        self.all_data = self.load_data()

    def load_data(self):
        train_test_annotation = pd.read_csv(
            os.path.join(self.data_dir, "train_test_split.txt"),
            sep=" ",
            header=None,
            index_col=0,
            names=["is_train"],
        )

        img_paths = pd.read_csv(
            os.path.join(self.data_dir, "images.txt"),
            sep=" ",
            header=None,
            index_col=0,
            names=["path"],
        )

        class_labels = pd.read_csv(
            os.path.join(self.data_dir, "image_class_labels.txt"),
            sep=" ",
            header=None,
            index_col=0,
            names=["class_id"],
        )

        annotations = pd.concat(
            [train_test_annotation, img_paths, class_labels], axis=1
        )

        # Create filtered Copy, not a View!
        all_data = annotations[annotations["is_train"] == self.is_train].copy()
        all_data["img"] = all_data["path"].apply(self._load_img)
        return all_data

    def _load_img(self, img_path):
        img = Image.open(os.path.join(self.image_dir, img_path)).convert("RGB")

        return self.transf(img)

    def __getitem__(self, index):
        obj = self.all_data.iloc[index].copy()
        image, label = obj["img"], obj["class_id"] - 1

        return image, label

    def __len__(self):
        return self.all_data.shape[0]
