import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "cars"
    dataset.data_dir = "data/"
    dataset.num_classes = 196

    return dataset
