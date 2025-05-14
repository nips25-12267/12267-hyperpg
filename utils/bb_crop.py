import os

import pandas as pd
from PIL import Image

DATA_DIR = "./data"
CUB_200_DIR_NAME = "CUB_200_2011"

bb_info = pd.read_csv(
    os.path.join(DATA_DIR, CUB_200_DIR_NAME, "bounding_boxes.txt"),
    sep=" ",
    header=None,
    index_col=0,
)

img_paths = pd.read_csv(
    os.path.join(DATA_DIR, CUB_200_DIR_NAME, "images.txt"),
    sep=" ",
    header=None,
    index_col=0,
    names=["path"],
)

img_paths["path"] = img_paths["path"].astype("str")

full_df = pd.concat([bb_info, img_paths], axis=1)


OUTDIR = os.path.join(DATA_DIR, CUB_200_DIR_NAME, "images_crop")
os.makedirs(OUTDIR, exist_ok=True)


def crop_img(s):
    x, y, w, h, p = s.to_numpy()
    img = Image.open(os.path.join(DATA_DIR, CUB_200_DIR_NAME, "images", p))
    cropped_img = img.crop((x, y, x + w, y + h))
    os.makedirs(os.path.dirname(os.path.join(OUTDIR, p)), exist_ok=True)
    cropped_img.save(os.path.join(OUTDIR, p))


full_df.apply(crop_img, axis=1)
