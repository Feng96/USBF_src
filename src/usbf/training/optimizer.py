

def optimizer(model):
    import torch
    groups = [dict(params=[p for n,p in model.named_parameters() if n.startswith('backbone.')], lr=2e-6, initial_lr=2e-6),
              dict(params=[p for n,p in model.named_parameters() if not n.startswith(('backbone.', 'shape_'))], lr=2e-5, initial_lr=2e-5)]
    new = [p for n,p in model.named_parameters() if n.startswith('shape_')]
    if new: groups.append(dict(params=new, lr=1e-4, initial_lr=1e-4))
    return torch.optim.AdamW(groups, weight_decay=1e-4)
