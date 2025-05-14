import torch.nn as nn

_LOSS_REPOSITORY = {}


def register_loss(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _LOSS_REPOSITORY:
            raise ValueError(
                f"Already registered loss function with name: {local_name}"
            )
        _LOSS_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_loss(loss_cfg: dict | list):
    if not isinstance(loss_cfg, list):
        cls = _LOSS_REPOSITORY[loss_cfg.pop("name")]
        return cls(**loss_cfg)

    return MultiObjectiveLoss(loss_cfg)


class MultiObjectiveLoss(nn.Module):
    def __init__(self, loss_cfg_list):
        super().__init__()
        self.loss_obj_list = [create_loss(cfg) for cfg in loss_cfg_list]

    def forward(self, *args, **kwargs):
        loss_dict = {}
        for loss in self.loss_obj_list:
            loss_dict.update(loss(*args, **kwargs))
        return loss_dict
