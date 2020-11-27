from itertools import combinations, product

import torch

trs = list(combinations(range(2, 5), 2)) + [None]
flips = list(range(2, 5)) + [None]
rots = list(range(1, 4)) + [None]
transform_list = list(product(flips, rots))


def simple_tta(x):
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function
    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])
    return out


def apply_simple_tta(model, x, average=True):
    todos = simple_tta(x)
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out
    return torch.stack(out).mean(dim=0)


def revert_tta_factory(flip, rot):
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise
