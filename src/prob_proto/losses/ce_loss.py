import torch

from .loss_factory import register_loss


@register_loss(name="cross_entropy")
class CrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self, weight: float):
        super().__init__()
        self.lambda_weight = weight

    def forward(self, pred: torch.Tensor, label: torch.Tensor, *args, **kwargs):
        ce_loss = super().forward(pred, label)
        return {"ce": self.lambda_weight * ce_loss}
