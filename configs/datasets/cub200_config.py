import ml_collections


def get_config():
    dataset = ml_collections.ConfigDict()
    dataset.name = "birds"
    dataset.data_dir = "data/CUB_200_2011"
    dataset.num_classes = 200

    return dataset
