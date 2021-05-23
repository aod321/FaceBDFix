import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


def find_float_boundary(maskdt, width=3):
    # Extract boundary from instance mask
    # maskdt = Shape(N, 1, h, w)
    #     maskdt = torch.Tensor(maskdt)
    n, c, h, w = maskdt.shape
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder,
                             stride=1, padding=width // 2)
    bml = torch.abs(boundary_mask - width * width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width * width / 2)
    return fbmask


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
    sdets = sdets.clone()
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets


def get_dets(maskdt, patch_size, iou_thresh=0.27, out_fmt='cxcywh'):
    """Generate patch proposals from the coarse mask.
    Args:
        maskdt (array): B, 1, H, W
        patch_size (int): [description]
        iou_thresh (float, optional): useful for nms. Defaults to 0.3.
    Returns:
        array: filtered bboxs. shape B, N, 4. each row contain x1, y1,
            x2, y2, score.
    """
    fbmask = find_float_boundary(maskdt)
    n, c, h, w = maskdt.shape
    # Shape(N, 1, 512, 512)
    sdets = []
    for i in range(n):
        mask = fbmask[i]
        nonzero_indexes = torch.nonzero(mask, as_tuple=True)
        score = mask[nonzero_indexes]
        nonzero_indexes = torch.nonzero(mask, as_tuple=False)
        ones = torch.ones((len(nonzero_indexes), 2), device=maskdt.device) * patch_size
        temp = torch.cat([nonzero_indexes[:, 2:3], nonzero_indexes[:, 1:2],
                          ones], dim=1)
        dets = torchvision.ops.box_convert(temp, in_fmt='cxcywh', out_fmt='xyxy')
        inds = torchvision.ops.nms(dets, score, iou_thresh)
        sdet = _force_move_back(dets[inds], h, w, patch_size)
        sdet = torchvision.ops.box_convert(sdet, in_fmt='xyxy', out_fmt=out_fmt)
        sdets.append(sdet)
    sdets = torch.stack(sdets)
    return sdets


def affine_crop(img, points=None, is_label=False, theta=None):
    # c_x, c_y, w, h = points.shape
    b, c, h, w = img.shape
    img_in = img.to(img.device)
    if points is not None:
        n, _ = points.shape
        assert points.shape == (n, 4)
        theta = torch.zeros((n, 2, 3), dtype=torch.float32, device=img.device, requires_grad=False).detach()
        for k in range(n):
            c_w, c_h, r_w, r_h = points[k]
            theta[k, 0, 0] = (r_w - 1) / (w - 1)
            theta[k, 0, 2] = -1 + (2 * c_w) / (w - 1)
            theta[k, 1, 1] = (r_h - 1) / (h - 1)
            theta[k, 1, 2] = -1 + (2 * c_h) / (h - 1)
    elif theta is not None:
        theta = theta
        n = theta.shape[0]
    else:
        raise RuntimeError('Expect theta or point not None')

    if n:
        grid = F.affine_grid(theta, [n, c, w, h], align_corners=True).type_as(theta)
        if is_label:
            patches = F.grid_sample(input=img_in.repeat([n, 1, 1, 1]), grid=grid, align_corners=True,
                                mode='nearest', padding_mode='zeros')
        else:
            patches = F.grid_sample(input=img_in.repeat([n, 1, 1, 1]), grid=grid, align_corners=False,
                        mode='bilinear', padding_mode='zeros')

    else:
        print("[WARNING] no patches cropped")
        patches = torch.zeros(0, 3, w, h)

    return patches, theta


def affine_mapback(preds, theta, is_label=False):
    # paste the images back
    n = theta.shape[0]
    _, c, r, r = preds.shape
    ones = torch.tensor([[0., 0., 1.]], device=preds.device).repeat(n, 1, 1).detach()
    rtheta = torch.cat([theta, ones], dim=1).detach()
    rtheta = torch.inverse(rtheta)
    rtheta = rtheta[:, 0:2]
    assert rtheta.shape == (n, 2, 3)
    del ones
    grid = F.affine_grid(rtheta, [n, c, r, r], align_corners=True).type_as(theta)
    if is_label:
        patches = F.grid_sample(input=preds, grid=grid, align_corners=True,
                                mode='nearest', padding_mode='zeros')
    else:
        patches = F.grid_sample(input=preds, grid=grid, align_corners=True,
                                mode='bilinear', padding_mode='zeros')
    patches, _ = torch.max(patches, dim=0, keepdim=True)
    return patches


class F1Accuracy(object):
    def __init__(self, num=2):
        super(F1Accuracy, self).__init__()
        self.hist_list = []
        self.num = num

    def fast_histogram(self, a, b, na, nb):
        '''
        fast histogram calculation
        ---
        * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
        '''
        assert a.shape == b.shape, (a.shape, b.shape)
        assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
        # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
        hist = np.bincount(
            nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
            minlength=na * nb).reshape(na, nb)
        assert np.sum(hist) == a.size
        return hist

    def collect(self, input, target):
        hist = self.fast_histogram(input.cpu().numpy(), target.cpu().numpy(),
                                   self.num, self.num)
        self.hist_list.append(hist)

    def calc(self):
        if self.hist_list:
            hist_sum = np.sum(np.stack(self.hist_list, axis=0), axis=0)
            A = hist_sum[1:self.num, :].sum()
            B = hist_sum[:, 1:self.num].sum()
            intersected = hist_sum[1:self.num, :][:, 1:self.num].sum()
            F1 = 2 * intersected / (A + B)
            self.hist_list.clear()
            return F1
        else:
            raise RuntimeError('No datas')