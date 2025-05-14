import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "flowers"
    dataset.data_dir = "data/OxfordFlowers102"
    dataset.num_classes = 102

    return dataset
