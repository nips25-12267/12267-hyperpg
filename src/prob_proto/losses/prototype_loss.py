import torch
import torch.nn.functional as F
from torch import nn

from .loss_factory import register_loss


class HyperPGLoss_BaseClass(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight


@register_loss(name="cluster")
class ClusterLoss(HyperPGLoss_BaseClass):
    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        density: torch.Tensor,
    ):
        """Cluster Loss from ProtoPNet

        Args:
            pred (torch.Tensor): Model Prediction. [Batch x Classes]
            label (torch.Tensor): Ground Truth Label with Class IDs. [Batch]
            density (torch.Tensor): Prototype Densities. [Batch x Classes x Proto p. Class x H x W]


        Returns:
            dict: loss dictionary
        """
        max_proto, _ = torch.max(density, dim=-1)
        max_proto, _ = torch.max(max_proto, dim=-1)
        # shape [B, C, P]

        max_proto_per_class, _ = torch.max(max_proto, dim=-1)
        # shape [B, C]

        select_sims = max_proto_per_class[torch.arange(max_proto.shape[0]), label]

        return {"cluster": self.weight * -1 * select_sims.mean()}


@register_loss(name="separation")
class SeparationLoss(HyperPGLoss_BaseClass):
    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        density: torch.Tensor,
    ):
        """Separation Loss from ProtoPNet

        Args:
            pred (torch.Tensor): Model Prediction. [Batch x Classes]
            label (torch.Tensor): Ground Truth Label with Class IDs. [Batch]
            density (torch.Tensor): Prototype Densities. [Batch x Classes x Proto p. Class x H x W]

        Returns:
            dict: loss dictionary
        """
        max_proto, _ = torch.max(density, dim=-1)
        max_proto, _ = torch.max(max_proto, dim=-1)
        # shape B, C, P

        mask = torch.ones_like(max_proto, dtype=torch.bool)
        # B, C, P
        mask[torch.arange(max_proto.size(0)), label, :] = False

        f_sim = max_proto[mask]
        # R
        filtered_sim = f_sim.view(max_proto.shape[0], max_proto.shape[1] - 1, -1)
        # B, C-1, P
        # C-1, because the ground truth class was just filtered out

        max_sim, _ = torch.max(filtered_sim, dim=-1)

        return {"separation": self.weight * max_sim.mean()}


@register_loss(name="cluster-l2")
class L2ClusterLoss(HyperPGLoss_BaseClass):
    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        density: torch.Tensor,
    ):
        """Cluster Loss from ProtoPNet on the basis of the L2 distance

        Args:
            pred (torch.Tensor): Model Prediction. [Batch x Classes]
            label (torch.Tensor): Ground Truth Label with Class IDs. [Batch]
            density (torch.Tensor): In this case the distance. [Batch x Classes x Proto p. Class x H x W]


        Returns:
            dict: loss dictionary
        """
        min_proto, _ = torch.min(density, dim=-1)
        min_proto, _ = torch.min(min_proto, dim=-1)
        # shape [B, C, P]

        min_proto_per_class, _ = torch.min(min_proto, dim=-1)
        # shape [B, C]

        select_sims = min_proto_per_class[torch.arange(min_proto.shape[0]), label]

        return {"cluster": self.weight * select_sims.mean()}


@register_loss(name="separation-l2")
class L2SeparationLoss(HyperPGLoss_BaseClass):
    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        density: torch.Tensor,
    ):
        """Separation Loss from ProtoPNet on the basis of the L2 distance

        Args:
            pred (torch.Tensor): Model Prediction. [Batch x Classes]
            label (torch.Tensor): Ground Truth Label with Class IDs. [Batch]
            density (torch.Tensor): In this case the distance. [Batch x Classes x Proto p. Class x H x W]

        Returns:
            dict: loss dictionary
        """
        min_proto, _ = torch.min(density, dim=-1)
        min_proto, _ = torch.min(min_proto, dim=-1)
        # shape B, C, P

        mask = torch.ones_like(min_proto, dtype=torch.bool)
        # B, C, P
        mask[torch.arange(min_proto.size(0)), label, :] = False

        f_sim = min_proto[mask]
        # R
        filtered_sim = f_sim.view(min_proto.shape[0], min_proto.shape[1] - 1, -1)
        # B, C-1, P
        # C-1, because the ground truth class was just filtered out

        max_sim, _ = torch.min(filtered_sim, dim=-1)

        return {"separation": self.weight * -1 * max_sim.mean()}
