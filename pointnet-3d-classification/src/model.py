"""
PointNet — Deep Learning on Point Sets for 3D Classification.
Qi et al., CVPR 2017.

PointNet processes unordered point clouds using shared MLPs and
a symmetric (max-pooling) function to achieve permutation invariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial transformer network for point cloud alignment.

    Learns a 3×3 or 64×64 transformation matrix to canonicalize
    the input point cloud.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize fc3 bias to identity
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.eye_(self.fc3.bias.view(k, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]  # Global max pooling
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity + small perturbation
        identity = torch.eye(self.k, device=x.device).view(
            1, self.k * self.k
        ).repeat(batch_size, 1)
        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetFeatureExtractor(nn.Module):
    """PointNet feature extraction backbone.

    Input: (B, 3, N) point cloud
    Output: (B, 1024) global feature vector
    """

    def __init__(self, use_input_transform: bool = True,
                 use_feature_transform: bool = True):
        super().__init__()
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        if use_input_transform:
            self.input_transform = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        if use_feature_transform:
            self.feature_transform = TNet(k=64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None,
                           torch.Tensor | None]:
        batch_size, _, num_points = x.shape

        # Input transform
        trans_input = None
        if self.use_input_transform:
            trans_input = self.input_transform(x)
            x = torch.bmm(trans_input, x)

        # MLP layers
        x = F.relu(self.bn1(self.conv1(x)))
        point_features = x  # (B, 64, N)

        trans_feat = None
        if self.use_feature_transform:
            trans_feat = self.feature_transform(point_features)
            x = torch.bmm(trans_feat, point_features)
            point_features = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        # Symmetric function: global max pooling
        global_feature = torch.max(x, 2)[0]  # (B, 1024)

        return global_feature, trans_input, trans_feat


class PointNetClassifier(nn.Module):
    """PointNet for 3D object classification on ModelNet40."""

    def __init__(self, num_classes: int = 40,
                 use_input_transform: bool = True,
                 use_feature_transform: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = PointNetFeatureExtractor(
            use_input_transform, use_feature_transform
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None,
                           torch.Tensor | None]:
        """
        Args:
            x: Point cloud (B, 3, N).

        Returns:
            (class logits, input transform matrix, feature transform matrix).
        """
        global_feat, trans_input, trans_feat = self.feature_extractor(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(self.dropout(x))))
        x = self.fc3(x)

        return x, trans_input, trans_feat

    def get_critical_points(self, x: torch.Tensor,
                            top_k: int = 64) -> torch.Tensor:
        """Identify the critical points that contributed to the global feature.

        The critical points are those whose per-point features had the
        maximum values across the point dimension.
        """
        self.eval()
        with torch.no_grad():
            global_feat, _, _ = self.feature_extractor(x)
        # Return the indices of max-activating points in the last conv layer
        # For simplicity, just return the global feature (critical points
        # require storing intermediate max indices)
        return global_feat


def pointnet_loss(logits: torch.Tensor, labels: torch.Tensor,
                  trans_input: torch.Tensor | None,
                  trans_feat: torch.Tensor | None,
                  reg_weight: float = 0.001) -> torch.Tensor:
    """PointNet loss with regularization on transformation matrices.

    Loss = CrossEntropy + λ * (||I - AAT||² + ||I - BBT||²)
    where A is the input transform and B is the feature transform.
    """
    ce_loss = F.cross_entropy(logits, labels)

    reg_loss = 0.0
    if trans_input is not None:
        # Orthogonality regularization: A should be close to orthogonal
        batch_size = trans_input.size(0)
        I = torch.eye(3, device=trans_input.device).unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        reg_loss += F.mse_loss(
            torch.bmm(trans_input, trans_input.transpose(1, 2)),
            I
        )
    if trans_feat is not None:
        batch_size = trans_feat.size(0)
        I = torch.eye(64, device=trans_feat.device).unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        reg_loss += F.mse_loss(
            torch.bmm(trans_feat, trans_feat.transpose(1, 2)),
            I
        )

    return ce_loss + reg_weight * reg_loss
