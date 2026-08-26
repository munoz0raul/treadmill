"""
model.py — Treadmill speed regressor built on a Kinetics-400 video backbone.

Input:  (batch, 3, T, 112, 112)  float32  [0..1]  RGB clip (T frames)
Output: (batch, 1)               float32  speed in km/h

Backbone: torchvision `mc3_18` (mixed 2D/3D ResNet, ~11.7M params) pretrained
on Kinetics-400 — which already contains walking / running / treadmill actions.
We drop its 400-way classification head and attach a small regression head.

Kinetics normalisation (mean/std) is baked in as buffers so the ONNX graph and
the board receive plain [0..1] RGB clips — no host-side normalisation needed.

Training is two-phase (see train.py):
  1. freeze_backbone() → train only the head so random head weights don't wreck
     the pretrained features.
  2. unfreeze_backbone() → fine-tune everything at a lower LR.
"""

import torch
import torch.nn as nn
from torchvision.models.video import mc3_18, MC3_18_Weights

# Kinetics-400 normalisation (from MC3_18_Weights.KINETICS400_V1.transforms())
KIN_MEAN = [0.43216, 0.394666, 0.37645]
KIN_STD  = [0.22803, 0.22145, 0.216989]


class SpeedCNN(nn.Module):
    """Name kept as SpeedCNN for pipeline compatibility; now a mc3_18 regressor."""

    def __init__(self, n_frames=8, in_h=112, in_w=112, pretrained=True):
        super().__init__()
        self.n_frames = n_frames

        weights = MC3_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = mc3_18(weights=weights)
        feat_dim = self.backbone.fc.in_features            # 512
        self.backbone.fc = nn.Identity()                   # strip 400-way head

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Normalisation baked in — input is plain [0..1] RGB, shape (B,3,T,H,W)
        self.register_buffer("mean", torch.tensor(KIN_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std",  torch.tensor(KIN_STD).view(1, 3, 1, 1, 1))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: (B, 3, T, H, W) in [0..1]
        x = (x - self.mean) / self.std
        feat = self.backbone(x)          # (B, 512)
        return self.head(feat)           # (B, 1)


if __name__ == "__main__":
    m = SpeedCNN(n_frames=8)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}  (trainable: {trainable:,})")
    m.freeze_backbone()
    head_only = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Head-only trainable: {head_only:,}")
    x = torch.zeros(2, 3, 8, 112, 112)
    print(f"Output shape: {m(x).shape}")
