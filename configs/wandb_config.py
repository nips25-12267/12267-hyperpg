import ml_collections


def get_config():
    wandb_cfg = ml_collections.ConfigDict()
    wandb_cfg.project = "prob-proto"

    return wandb_cfg
