import torch
from torch import nn

_PM_REPOSITORY = {}


def register_prototype(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _PM_REPOSITORY:
            raise ValueError(
                f"Already registered prototype module with name: {local_name}"
            )
        _PM_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_prototype(pm_name: str, *args, **kwargs):
    cls = _PM_REPOSITORY[pm_name]

    return cls(*args, **kwargs)


class PrototypeModule(nn.Module):
    def __init__(self, num_classes: int, num_prototypes: int, latent_channels: int):
        super().__init__()
        self.prototypes = nn.Parameter(
            torch.rand(num_classes, num_prototypes, latent_channels, 1, 1)
        )

    def forward(self, feats: torch.tensor):
        """Compute Prototype Similarity Maps

        Args:
            feats (torch.tensor): input features
        """
        raise NotImplementedError()
