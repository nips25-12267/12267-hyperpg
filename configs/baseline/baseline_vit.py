import ml_collections

from configs import default_config
from configs.datasets import cub200_config


def get_config():
    config = default_config.get_default_configs()
    config.dataset = cub200_config.get_config()

    model = config.model
    model.name = "baseline"
    model.backbone = "vit"
    model.num_classes = config.dataset.num_classes

    training = config.training
    training.batch_size = 16
    # training.num_epochs = 100

    return config
