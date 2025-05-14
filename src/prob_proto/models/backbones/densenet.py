import torch
import torchvision

from .backbone_factory import BackboneNonSeqABC, register_backbone


@register_backbone(name="dense121")
class DenseNetBackbone(BackboneNonSeqABC):
    def __init__(self) -> None:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        densenet = torchvision.models.densenet121(weights=weights)
        modules = densenet.features

        super().__init__(weights, modules, channels=1024, width=7, height=7)
