import torch
import torchvision

from .backbone_factory import BackboneABC, register_backbone


@register_backbone(name="convnext")
class ConvNextBackbone(BackboneABC):
    def __init__(self) -> None:
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        convnext = torchvision.models.convnext_base(weights=weights)
        modules = list(convnext.children())[:-2]
        super().__init__(weights, modules, channels=1024, width=7, height=7)
