"""RGB-led height fusion experiments, not an MMRAD reproduction.

The only measured modalities are RGB and nDSM. `valid` is raster validity
metadata, never a building/repair mask. Gates are learned weights, not calibrated
confidence. No downloads or checkpoint reads occur in the constructor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

VERSION = "rgb_height_r1_v0.1.0"


@dataclass(frozen=True)
class R1Config:
    variant: str = "rgb"  # rgb, gated, gated_boundary
    width: int = 128
    freeze_backbone_bn: bool = True


def conv(cin, cout, kernel=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class RGBHeightR1(nn.Module):
    def __init__(self, config: R1Config | None = None):
        super().__init__()
        self.config = config or R1Config()
        if self.config.variant not in ("rgb", "gated", "gated_boundary"):
            raise ValueError("Unknown R1 variant")
        if self.config.width < 8 or self.config.width % 8:
            raise ValueError("width must be a positive multiple of 8")
        width = self.config.width
        # Shared modules are created first and in identical order in all variants.
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (256, 512, 1024, 2048)])
        self.smooth = nn.ModuleList([conv(width, width) for _ in range(4)])
        self.detail = conv(64, 32, 1)
        self.decode = nn.Sequential(conv(width + 32, 64), conv(64, 64))
        self.mask_head = nn.Conv2d(64, 1, 1)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        if self.config.variant != "rgb":
            self.height_stem = nn.Sequential(conv(1, 16, stride=2), conv(16, 32, stride=2))
            self.height_stages = nn.ModuleList([nn.Identity(), conv(32, 64, stride=2),
                                               conv(64, 96, stride=2), conv(96, 128, stride=2)])
            self.height_project = nn.ModuleList([nn.Conv2d(c, width, 1, bias=False) for c in (32, 64, 96, 128)])
            self.gates = nn.ModuleList([nn.Sequential(conv(2 * width + 1, 32, 1),
                                                     nn.Conv2d(32, 1, 1)) for _ in range(4)])
            for gate in self.gates:
                nn.init.constant_(gate[-1].bias, -2.0)
        if self.config.variant == "gated_boundary":
            self.boundary_features = conv(64, 32)
            self.boundary_head = nn.Conv2d(32, 1, 1)
            self.boundary_feedback = nn.Conv2d(32, 1, 1, bias=False)
            nn.init.zeros_(self.boundary_feedback.weight)
        self.train(self.training)

    def load_rgb_pretrained(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Only the ImageNet classifier is omitted. Anything else must match.
        state = {k: v for k, v in state.items() if k not in ("fc.weight", "fc.bias")}
        self.backbone.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # Preserve pretrained statistics at small batch size.
        return self

    def _decode(self, pyramid, stem, size):
        top = None
        for index in reversed(range(4)):
            top = pyramid[index] if top is None else pyramid[index] + resize(top, pyramid[index].shape[-2:])
            top = self.smooth[index](top)
        features = self.decode(torch.cat((resize(top, stem.shape[-2:]), self.detail(stem)), 1))
        logits = self.mask_head(features)
        boundary = None
        if self.config.variant == "gated_boundary":
            shape = self.boundary_features(features)
            boundary = resize(self.boundary_head(shape), size)
            logits = logits + self.boundary_feedback(shape)
        return resize(logits, size), boundary

    def forward(self, rgb, ndsm=None, valid=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB must have shape B,3,H,W")
        size = rgb.shape[-2:]
        x = (rgb - self.rgb_mean) / self.rgb_std
        b = self.backbone
        stem = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(stem)
        features = []
        for layer, lateral in zip((b.layer1, b.layer2, b.layer3, b.layer4), self.lateral):
            x = layer(x)
            features.append(lateral(x))
        rgb_logits, rgb_boundary = self._decode(features, stem, size)
        if self.config.variant == "rgb":
            return {"logits": rgb_logits, "rgb_logits": rgb_logits,
                    "boundary_logits": None, "gates": []}
        if ndsm is None:
            ndsm = torch.zeros_like(rgb[:, :1])
            valid = torch.zeros_like(ndsm)
        if ndsm.shape != rgb[:, :1].shape:
            raise ValueError("nDSM must match RGB spatial/batch dimensions")
        finite = torch.isfinite(ndsm)
        if valid is None:
            valid = finite.to(rgb.dtype)
        if valid.shape != ndsm.shape:
            raise ValueError("Validity mask shape mismatch")
        valid = torch.nan_to_num(valid.to(rgb.dtype), nan=0., posinf=0., neginf=0.).clamp(0, 1) * finite
        height = torch.nan_to_num(ndsm, nan=0., posinf=0., neginf=0.).clamp(0, 1) * valid
        height = self.height_stem(height)
        fused, gates = [], []
        for feature, stage, project, gate in zip(features, self.height_stages, self.height_project, self.gates):
            height = stage(height)
            delta = resize(project(height), feature.shape[-2:])
            coverage = F.adaptive_avg_pool2d(valid, feature.shape[-2:])
            weight = torch.sigmoid(gate(torch.cat((feature, delta, coverage), 1))) * coverage
            fused.append(feature + weight * delta)
            gates.append(weight)
        logits, boundary = self._decode(fused, stem, size)
        # Exact sample-level fallback even if future decoder implementations change.
        missing = valid.sum((1, 2, 3), keepdim=True) == 0
        logits = torch.where(missing, rgb_logits, logits)
        if boundary is not None:
            boundary = torch.where(missing, rgb_boundary, boundary)
        return {"logits": logits, "rgb_logits": rgb_logits,
                "boundary_logits": boundary, "gates": gates}

    def manifest(self):
        return {"version": VERSION, "config": asdict(self.config),
                "parameters": sum(p.numel() for p in self.parameters()),
                "module_parameters": {n: sum(p.numel() for p in m.parameters()) for n, m in self.named_children()},
                "gate_is_calibrated_confidence": False,
                "independent_rgb_path": True, "validity_is_semantic_label": False}


def boundary_target(target):
    """Inner 1-pixel binary-mask contour; image border is not a fake contour."""
    return target - (-F.max_pool2d(-target, 3, 1, 1))


def segmentation_loss(logits, target):
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = 1 - ((2 * (probability * target).sum((1, 2, 3)) + 1e-6) /
                (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)).mean()
    return .5 * bce + .5 * dice


def r1_loss(output, target, variant, rgb_weight=.3, boundary_weight=.2):
    main = segmentation_loss(output["logits"], target)
    aux = segmentation_loss(output["rgb_logits"], target) if variant != "rgb" else main * 0
    edge = main * 0
    if variant == "gated_boundary":
        target_edge = boundary_target(target.float())
        edge = segmentation_loss(output["boundary_logits"], target_edge)
    total = main + rgb_weight * aux + boundary_weight * edge
    return total, {"main": main.detach(), "rgb_aux": aux.detach(), "boundary": edge.detach()}
