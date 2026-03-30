import torch
from torch import nn

class ProtoLoss(nn.Module):
    """
    Supervised instance-level contrastive learning which directly contrast each instance with
    the corresponding emotion prototype.
    """
    def __init__(self, feature_dim, num_classes):
        super(ProtoLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, features, labels, prototypes, temperature=1.0):
        """
        Args:
            features: hidden vector of shape [bs, n_views, ...].
            labels: ground truth of shape [bs].
            prototypes: prototypes of shape [n_classes, ...].
        """
        
        if len(features.shape) != 2:
            raise ValueError('`features` needs to be [bsz, feature_dim]')

        # Normalize prototypes and features
        prototypes = nn.functional.normalize(prototypes, p=2, dim=1)
        features = nn.functional.normalize(features, p=2, dim=1)

        # The distance between the feature of an instance and the prototype of the corresponding label.
        # [bs * n_views, 1]
        pos_dist = torch.sum(features * prototypes[labels], dim=1, keepdim=True)

        # The distance between the feature of an instance and the prototypes of all labels.
        # [bs * n_views, n_classes]
        all_dist = torch.matmul(features, prototypes.transpose(0, 1))

        # contrastive loss
        # [bs * n_views]
        loss = (torch.log(torch.sum(torch.exp(all_dist / temperature), dim=1)) - pos_dist.squeeze(-1) / temperature).mean()

        return loss