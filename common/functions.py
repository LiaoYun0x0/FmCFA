import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.einops import rearrange, repeat
import cv2

def mask_border(m, b, v):
    m[:, :b]=v
    m[:, :, :b]=v
    m[:, :, :, :b]=v
    m[:, :, :, :, :b]=v
    m[:, -b:0]=v
    m[:, :, -b:0]=v
    m[:, :, :, -b:0]=v
    m[:, :, :, :, -b:0]=v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd]= v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0-bd:] = v
        m[b_idx, :, w0-bd:] = v
        m[b_idx, :, :, h1-bd:] = v
        m[b_idx, :, :, h1-bd:] = v

@torch.no_grad()
def mutual_nearest_neighbor_match(
    scores,
    axes_lengths: dict,
    border: int,
    match_threshold: float,
    mask0=None,
    mask1=None,
    use_bins=False
):
    if use_bins:
        conf_matrix = scores[:, :-1, :-1].exp()
    else:
        conf_matrix = scores

    mask = conf_matrix > match_threshold
    mask = rearrange(
        mask,
        'B (H0 W0) (H1 W1) -> B H0 W0 H1 W1',
        **axes_lengths
    )
    if mask0 is not None:
        mask_border(mask, border, False)
    else:
        mask_border_with_padding(mask, border, False, mask0, mask1)

    mask = rearrange(
        mask,
        'B H0 W0 H1 W1 -> B (H0 W0) (H1 W1)',
        axes_lengths
    )

    max0 = conf_matrix.max(dim=2, keepdim=True)[0]
    max1 = conf_matrix.max(dim=1, keepdim=True)[0]

    mask = mask * (conf_matrix == max0) * (conf_matrix == max1)

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix[b_ids, i_ids, j_ids]
    matches = torch.stack([b_ids, i_ids, j_ids]).T

    return matches[mconf != 0], mconf[mconf != 0]

def assignments_to_matches(assignments, use_bins=False):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    return mids

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid

def grid_positions(h, w, device='cpu', matrix=False):
    rows = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)

    cols = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)

    if matrix:
        return torch.stack([cols, rows], dim=0)
    else:
        return torch.cat([cols.view(1, -1), rows.view(1, -1)], dim=0)

def upscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = coord * 2.0 + 0.5
    return coord

def downscale(coord, scaling_steps):
    for _ in range(scaling_steps):
        coord = (coord - .5) / 2.
    return coord

def normalize(coord, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord.device).float()
    coord_norm = (coord - c) / c
    return coord_norm

def denormalize(coord_norm, h, w):
    c = torch.Tensor([(w-1)/2., (h-1)/2.]).to(coord_norm.device)
    coord = coord_norm * c + c
    return coord

def ind2coord(ind, w):
    ind = ind.unsqueeze(-1)
    x = ind % w
    y = ind // w
    coord = torch.cat([x, y], -1).float()
    return coord

def test_ind2coord(assignments, w, use_bins=True):
    if use_bins:
        assignments = assignments[:, :-1, :-1]
    mask = assignments > 0
    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mids = torch.stack([b_ids, i_ids, j_ids]).T

    mkpts0 = ind2coord(i_ids, w)
    mkpts1 = ind2coord(j_ids, w)

    return mkpts0, mkpts1

def make_grid(cols,rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    xs = np.tile(xs[np.newaxis,:],(rows,1))
    ys = np.tile(ys[:,np.newaxis],(1,cols))
    grid = np.concatenate([xs[...,np.newaxis],ys[...,np.newaxis]],axis=-1).copy()
    return grid 

def _transform_inv(img,mean,std):
    img = img * std + mean
    img  = np.uint8(img * 255.0)
    img = img.transpose(1,2,0)
    return img

def draw_match(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts[::16]:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match2(match_mask,query,refer,x, y, homo_filter=True,patch_size=8):
    grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    _pts = grid[match_mask]
    query = cv2.copyMakeBorder(query, 0, 800-512, 0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    out_img = np.concatenate([query,refer],axis=1).copy()
    query_pts = []
    refer_pts = []
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in _pts:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[0] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[0] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if query_pts.shape[0] > 0:
        xx_mean = np.mean((refer_pts - query_pts-x-512)[:, 0], axis=0)
        yy_mean = np.mean((refer_pts - query_pts-y)[:, 1], axis=0)
        cv2.rectangle(out_img, (x+512,y), (x+512+512, y+512), (0, 255, 255), 2)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def draw_match3(matches,query,refer,x, y, homo_filter=True,patch_size=8):
    matches = matches.detach().cpu().numpy()
    out_img = np.concatenate([query,refer],axis=1).copy()
    wq = query.shape[1]
    wr = refer.shape[1]
    qcols = wq // patch_size
    rcols = wr // patch_size
    query_pts = []
    refer_pts = []
    for pt in matches[::16,:]:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + (pt[1] // qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size + query.shape[1]
        y1 = patch_size/2 + (pt[2] // rcols) * patch_size
        query_pts.append([x0,y0])
        refer_pts.append([x1,y1])
        # cv2.line(out_img,(x0,y0),(x1,y1),(0,255,0),2)
    query_pts = np.asarray(query_pts,np.float32)
    refer_pts = np.asarray(refer_pts,np.float32)
    if homo_filter and query_pts.shape[0] > 4:
        H,mask = cv2.findHomography(query_pts,refer_pts,cv2.RANSAC,ransacReprojThreshold=16)
        for i in range(query_pts.shape[0]):
            if mask[i]:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
            else:
                cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,0,255),1)
    else:
        for i in range(query_pts.shape[0]):
            cv2.line(out_img,(int(query_pts[i,0]),int(query_pts[i,1])),(int(refer_pts[i,0]),int(refer_pts[i,1])),(0,255,0),1)
    cv2.putText(out_img, str(len(query_pts)), (100,100), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 3)
    return out_img

def get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + torch.div(pt[1], qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + torch.div(pt[2], rcols) * patch_size
        query_pts.append(torch.Tensor([x0, y0]))
        refer_pts.append(torch.Tensor([x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()
    else:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts


def batch_get_mkpts(matches, query, refer, patch_size=8):
    #grid = make_grid(match_mask.shape[1],match_mask.shape[0])
    #_pts = grid[match_mask]
    #matches = match_mask.detach().cpu().numpy()
    query_pts = []
    refer_pts = []
    wq = query.shape[2]
    wr = refer.shape[2]
    qcols = wq // patch_size
    rcols = wr // patch_size
    """
    for pt in matches:
        x0 = patch_size/2 + (pt[1] % qcols) * patch_size
        y0 = patch_size/2 + torch.div(pt[1], qcols) * patch_size
        x1 = patch_size/2 + (pt[2] % rcols) * patch_size
        y1 = patch_size/2 + torch.div(pt[2], rcols) * patch_size
        query_pts.append(torch.Tensor([pt[0], x0, y0]))
        refer_pts.append(torch.Tensor([pt[0], x1, y1]))
    if len(query_pts) > 0:
        query_pts = torch.stack(query_pts).cuda()
        refer_pts = torch.stack(refer_pts).cuda()

    """
    x0 = patch_size/2 + (matches[:, 1] % qcols) * patch_size
    y0 = patch_size/2 + torch.div(matches[:, 1], qcols) * patch_size
    x1 = patch_size/2 + (matches[:, 2] % qcols) * patch_size
    y1 = patch_size/2 + torch.div(matches[:, 2], qcols) * patch_size
    query_pts = torch.cat((matches[:, 0].unsqueeze(1), x0.unsqueeze(1), y0.unsqueeze(1)), 1)
    refer_pts = torch.cat((matches[:, 0].unsqueeze(1), x1.unsqueeze(1), y1.unsqueeze(1)), 1)

    if len(query_pts) < 1:
        query_pts = torch.Tensor(0, 2).cuda()
        refer_pts = torch.Tensor(0, 2).cuda()

    return query_pts, refer_pts






