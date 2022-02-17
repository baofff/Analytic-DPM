import torch


def linear_interpolate(a, b, steps):
    a_shape = a.shape
    a = a.detach().cpu().view(-1)
    b = b.detach().cpu().view(-1)
    res = []
    for aa, bb in zip(a, b):
        res.append(torch.linspace(aa, bb, steps=steps).unsqueeze(dim=1))
    res = torch.cat(res, dim=1)
    res = res.view(len(res), *a_shape)
    return res


def rect_interpolate(a, b, c, steps):
    a = a.detach().cpu()
    b = b.detach().cpu()
    c = c.detach().cpu()
    ab = linear_interpolate(a, b, steps) - a
    ac = linear_interpolate(a, c, steps) - a
    res = []
    for st in ac:
        res.append(ab + st)
    res = torch.cat(res, dim=0) + a
    return res
