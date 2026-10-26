import torch
import torch.nn.functional as F
from usbf.models.rgb_height_r1 import boundary_target
from .validity import masked_r1_loss,masked_segmentation_loss

def interior(valid, radius):
    unknown = F.pad((~valid.bool()).float(), (radius, radius, radius, radius), value=1.)
    return F.max_pool2d(unknown, 2 * radius + 1, 1) == 0

def contour_targets(target, valid):
    with torch.no_grad():
        target = torch.where(valid.bool(), target.float(), 0.)
        domain = interior(valid, 6)
        coords = torch.arange(-3, 4, device=target.device, dtype=torch.float32)
        kernel = torch.exp(-coords.square() / (2 * 1.2 ** 2)); kernel /= kernel.sum()
        smooth = F.conv2d(F.conv2d(target, kernel[None, None, None, :], padding=(0, 3)),
                          kernel[None, None, :, None], padding=(3, 0))
        derivative = torch.tensor([-.5, 0., .5], device=target.device)
        gx = F.conv2d(smooth, derivative[None, None, None, :], padding=(0, 1))
        gy = F.conv2d(smooth, derivative[None, None, :, None], padding=(1, 0))
        amplitude = (gx.square() + gy.square()).sqrt()
        nx, ny = gx / amplitude.clamp_min(1e-6), gy / amplitude.clamp_min(1e-6)
        field = torch.cat((nx**4 - 6 * nx.square() * ny.square() + ny**4,
                           4 * nx * ny * (nx.square() - ny.square())), 1)
        xx, yy, xy = [F.avg_pool2d(z, 5, 1, 2) for z in (gx.square(), gy.square(), gx * gy)]
        coherence = ((xx - yy).square() + 4 * xy.square()).sqrt() / (xx + yy).clamp_min(1e-6)
        edge = boundary_target(target)
        near = F.max_pool2d(edge, 5, 1, 2) > 0
        support = domain & near & (amplitude > .03) & (coherence > .6)
        return dict(edge=edge, edge_domain=domain, field=field, field_domain=support)

def shape_loss(output, target, valid):
    main, parts = masked_r1_loss(output, target, valid, 'gated')
    # Disable autocast here: continuous orientation targets/reductions use FP32.
    with torch.autocast(device_type=target.device.type, enabled=False):
        truth = contour_targets(target, valid)
        edge = masked_segmentation_loss(output['boundary_logits'], truth['edge'], truth['edge_domain'])
        support = truth['field_domain'].float()
        field = output['field'].float()
        assert torch.isfinite(field).all()
        norm = field.square().sum(1, keepdim=True).clamp_min(1e-8).sqrt()
        cosine = ((field / norm) * truth['field']).sum(1, keepdim=True).clamp(-1, 1)
        direction = ((1 - cosine + .1 * (norm - 1).square()) * support).sum() / support.sum().clamp_min(1)
    return main + .2 * edge + .1 * direction, {**parts, 'edge': edge.detach(), 'direction': direction.detach()}
