import numpy as np
from scipy.signal import correlate2d

def conv2d(x, weight, padding=0):
    import torch as _torch
    xd = x.data
    wd = weight.data
    B, C, H, W = xd.shape
    OC, IC, kh, kw = wd.shape
    if isinstance(padding, int):
        ph = pw = padding
    else:
        ph, pw = padding
    out = np.zeros((B, OC, H, W))
    xp = np.pad(xd, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    for b in range(B):
        for oc in range(OC):
            acc = np.zeros((H, W))
            for ic in range(IC):
                acc += correlate2d(xp[b, ic], wd[oc, ic], mode='valid')
            out[b, oc] = acc
    return _torch.Tensor(out, x.device.type)


def interpolate(x, size, mode='bicubic', align_corners=None):
    """Shim substitute for torch.nn.functional.interpolate (bicubic/area
    only, via skimage). Not numerically identical to real torch -- only
    used here to verify the multi-scale pyramid PLUMBING (shapes, pipeline
    correctness) offline; the real notebook uses real torch's interpolate."""
    import torch as _torch
    from skimage.transform import resize as _sk_resize
    xd = x.data
    B, C, H, W = xd.shape
    order = 3 if mode == 'bicubic' else 1
    anti_alias = (mode == 'area')
    out = np.zeros((B, C, size[0], size[1]), dtype=xd.dtype)
    for b in range(B):
        for c in range(C):
            out[b, c] = _sk_resize(xd[b, c], size, order=order,
                                    anti_aliasing=anti_alias, mode='edge')
    return _torch.Tensor(out, x.device.type)
