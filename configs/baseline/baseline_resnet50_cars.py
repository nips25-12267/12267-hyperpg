import ml_collections

from configs import default_config
from configs.datasets import cars_config


def get_config():
    config = default_config.get_default_configs()

    config.dataset = cars_config.get_config()

    model = config.model
    model.name = "baseline"
    model.backbone = "resnet50"
    model.num_classes = config.dataset.num_classes

    training = config.training
    training.batch_size = 48

    return config
