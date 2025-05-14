import torch.nn as nn

_BACKBONE_REPOSITORY = {}


def register_backbone(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _BACKBONE_REPOSITORY:
            raise ValueError(f"Already registered model with name: {local_name}")
        _BACKBONE_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def create_backbone(name: str):
    cls = _BACKBONE_REPOSITORY[name]

    return cls()


class BackboneABC(nn.Module):
    def __init__(
        self, weights, modules, channels: int, width: int, height: int
    ) -> None:
        super().__init__()

        self.transforms = weights.transforms()
        self.out_channels = channels
        self.out_width = width
        self.out_height = height
        self.net = nn.Sequential(*modules)

    def forward(self, img_batch):
        batch_process = self.transforms(img_batch)
        return self.net(batch_process)


class BackboneNonSeqABC(nn.Module):
    def __init__(
        self, weights, modules, channels: int, width: int, height: int
    ) -> None:
        super().__init__()

        self.transforms = weights.transforms()
        self.out_channels = channels
        self.out_width = width
        self.out_height = height
        self.net = modules

    def forward(self, img_batch):
        batch_process = self.transforms(img_batch)
        return self.net(batch_process)
