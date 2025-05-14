import ml_collections

from configs import default_config
from configs.datasets import cub200_config


def get_config():
    config = default_config.get_default_configs()

    config.dataset = cub200_config.get_config()

    model = config.model
    model.name = "protopnet"
    model.backbone = "resnet50"
    model.num_classes = config.dataset.num_classes
    model.num_protos = 10
    model.latent_channels = 128
    model.pm_name = "l2"

    training = config.training
    training.batch_size = 48
    training.num_epochs = 200

    loss = config.loss
    loss.append(
        {"name": "cluster-l2", "weight": 0.8}
    )
    loss.append({"name": "separation-l2", "weight": 0.08})

    return config
