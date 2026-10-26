"""B1: native-grid detail residual and fourfold contour orientation.

An original, lightweight frame-field-inspired extension, NOT a reproduction of
Girard et al. Nor a polygon decoder. q=(cos(4 theta),sin(4 theta)) represents
unoriented orthogonal axes; it does not force buildings to have right angles.
All legacy B0 keys retain their meaning. Zero residual initialization preserves
the inherited mask exactly. Invalid/missing height never becomes semantic GT.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .rgb_height_r1 import RGBHeightR1, R1Config, conv, resize


class RGBHeightShape(RGBHeightR1):
    version = 'b1_native_detail_fourfold_v1'

    def __init__(self):
        super().__init__(R1Config('gated'))
        self.shape_rgb = nn.Sequential(conv(3, 16), conv(16, 16))
        self.shape_detail = nn.Sequential(conv(80, 32), conv(32, 32))
        self.shape_edge = nn.Conv2d(32, 1, 1)
        self.shape_field = nn.Conv2d(32, 2, 1)
        # Feed the predicted edge/directions into a learned mask correction.
        # It is not threshold-based smoothing or forced rectification.
        self.shape_residual = nn.Conv2d(35, 1, 1)
        nn.init.zeros_(self.shape_residual.weight)
        nn.init.zeros_(self.shape_residual.bias)

    def load_b0(self, state):
        incompatible = self.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys
        assert set(incompatible.missing_keys) == {k for k in self.state_dict() if k.startswith('shape_')}

    def _decode_with_features(self, pyramid, stem, size):
        top = None
        for index in reversed(range(4)):
            top = pyramid[index] if top is None else pyramid[index] + resize(top, pyramid[index].shape[-2:])
            top = self.smooth[index](top)
        decoded = self.decode(torch.cat((resize(top, stem.shape[-2:]), self.detail(stem)), 1))
        return resize(self.mask_head(decoded), size), decoded

    def _refine(self, logits, decoded, native):
        detail = self.shape_detail(torch.cat((resize(decoded, native.shape[-2:]), native), 1))
        edge, field = self.shape_edge(detail), self.shape_field(detail)
        residual = self.shape_residual(torch.cat((detail, edge.sigmoid(), F.normalize(field, dim=1, eps=1e-4)), 1))
        return logits + residual, edge, field

    def forward(self, rgb, ndsm=None, valid=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected RGB B,3,H,W')
        size = rgb.shape[-2:]
        normalized = (rgb - self.rgb_mean) / self.rgb_std
        b = self.backbone
        stem = b.relu(b.bn1(b.conv1(normalized)))
        x = b.maxpool(stem)
        features = []
        for layer, lateral in zip((b.layer1, b.layer2, b.layer3, b.layer4), self.lateral):
            x = layer(x); features.append(lateral(x))
        rgb_base, rgb_decoded = self._decode_with_features(features, stem, size)
        native = self.shape_rgb(normalized)
        rgb_logits, rgb_edge, rgb_field = self._refine(rgb_base, rgb_decoded, native)
        if ndsm is None:
            ndsm, valid = torch.zeros_like(rgb[:, :1]), torch.zeros_like(rgb[:, :1])
        if ndsm.shape != rgb[:, :1].shape:
            raise ValueError('Height shape mismatch')
        finite = torch.isfinite(ndsm)
        if valid is None: valid = finite.to(rgb.dtype)
        if valid.shape != ndsm.shape: raise ValueError('Validity shape mismatch')
        valid = torch.nan_to_num(valid.to(rgb.dtype), nan=0., posinf=0., neginf=0.).clamp(0, 1) * finite
        height = torch.nan_to_num(ndsm, nan=0., posinf=0., neginf=0.).clamp(0, 1) * valid
        height = self.height_stem(height)
        fused, gates = [], []
        for feature, stage, project, gate in zip(features, self.height_stages, self.height_project, self.gates):
            height = stage(height)
            delta = resize(project(height), feature.shape[-2:])
            coverage = F.adaptive_avg_pool2d(valid, feature.shape[-2:])
            weight = torch.sigmoid(gate(torch.cat((feature, delta, coverage), 1))) * coverage
            fused.append(feature + weight * delta); gates.append(weight)
        base, decoded = self._decode_with_features(fused, stem, size)
        logits, edge, field = self._refine(base, decoded, native)
        missing = valid.sum((1, 2, 3), keepdim=True) == 0
        return dict(logits=torch.where(missing, rgb_logits, logits), rgb_logits=rgb_logits,
                    boundary_logits=torch.where(missing, rgb_edge, edge),
                    field=torch.where(missing, rgb_field, field), gates=gates)
