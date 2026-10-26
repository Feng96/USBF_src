import torch
import torch.nn.functional as F
from usbf.models.rgb_height_r1 import boundary_target

def safe_domain(valid, radius=1):
    """Exclude neighbors of unknown pixels; outer raster edge matches legacy."""
    return F.max_pool2d((~valid.bool()).float(), 2 * radius + 1, 1, radius) == 0

def masked_segmentation_loss(logits, target, valid):
    valid = valid.bool()
    # Where BEFORE sigmoid/BCE: NaNs in ignored positions cannot poison reduction.
    logits = torch.where(valid, logits.float(), 0.)
    target = torch.where(valid, target.float(), 0.)
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite values in supervised domain")
    n = valid.sum()
    if not bool(n):
        return logits.sum() * 0
    probability = logits.sigmoid() * valid
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / n
    dims = (1, 2, 3)
    dice = 1 - (2 * (probability * target).sum(dims) + 1e-6) / (probability.sum(dims) + target.sum(dims) + 1e-6)
    return .5 * bce + .5 * dice[valid.sum(dims) > 0].mean()

def masked_r1_loss(output, target, valid, variant):
    main = masked_segmentation_loss(output["logits"], target, valid)
    aux, edge = main * 0, main * 0
    if variant != "rgb":
        aux = masked_segmentation_loss(output["rgb_logits"], target, valid)
    if variant == "gated_boundary":
        clean = torch.where(valid.bool(), target.float(), 0.)
        edge = masked_segmentation_loss(output["boundary_logits"], boundary_target(clean), safe_domain(valid))
    return main + .3 * aux + .2 * edge, {"main": main.detach(), "rgb_aux": aux.detach(), "boundary": edge.detach()}

class MaskedCounts:
    def __init__(self, device):
        self.values = torch.zeros(10, dtype=torch.float64, device=device)

    def add(self, logits, target, valid):
        valid = valid.bool()
        prediction, truth = (logits >= 0) & valid, (target >= .5) & valid
        # Erode by contour radius + matching radius. No unknown-region edges or
        # matches enter either boundary denominator or numerator.
        domain = safe_domain(valid, 2)
        ep = (boundary_target(prediction.float()) > 0) & domain
        et = (boundary_target(truth.float()) > 0) & domain
        mp = ep & (F.max_pool2d(et.float(), 3, 1, 1) > 0)
        mt = et & (F.max_pool2d(ep.float(), 3, 1, 1) > 0)
        self.values += torch.stack(((prediction & truth).sum(), (prediction & ~truth & valid).sum(),
                                    (~prediction & truth & valid).sum(), (~prediction & ~truth & valid).sum(),
                                    mp.sum(), ep.sum(), mt.sum(), et.sum(), valid.sum(), domain.sum())).double()

    def result(self):
        tp, fp, fn, tn, mp, ep, mt, et, pixels, boundary_pixels = self.values.tolist()
        precision, recall = mp / max(1, ep), mt / max(1, et)
        bf = 1. if ep == et == 0 else 2 * precision * recall / max(1e-12, precision + recall)
        return {"iou": (tp / (tp + fp + fn) if tp + fp + fn else 1.) if pixels else None,
                "boundary_f1_linf_1px": bf if boundary_pixels else None,
                "fp_pixels": fp, "fn_pixels": fn, "tp_pixels": tp, "tn_pixels": tn,
                "supervised_pixels": pixels, "boundary_domain_pixels": boundary_pixels}
