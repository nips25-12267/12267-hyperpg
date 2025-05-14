import ml_collections

from configs import wandb_config
from configs.datasets import cars_config, cub200_config, flowers_config


def get_default_configs():
    config = ml_collections.ConfigDict()

    config.wandb = wandb_config.get_config()
    config.dataset = cub200_config.get_config()
    config.training = training = ml_collections.ConfigDict()

    training.batch_size = 64
    training.num_epochs = 50
    training.val_epochs = 5
    training.checkpoint_epochs = 5  # training.val_epochs * 2

    # Fill in SubConfig
    config.model = ml_collections.ConfigDict()

    config.loss = [
        {"name": "cross_entropy", "weight": 1.0},
    ]

    config.seed = 42
    return config


def get_config():
    return get_default_configs()
