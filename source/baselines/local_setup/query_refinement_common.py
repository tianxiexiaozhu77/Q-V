"""Fixed, gradient-free transfer primitives. No test labels or fitted weights.

Center80 is applied after each model's native pixel preprocessing. Query direction
uses unit support embeddings; query norms are preserved for learned decoders and
unnormalized dot-product heads. This extends phase6 to different native spaces.
"""
import numpy as np
import os
import torch
from torch.nn import functional as F

ETA = .3


def crop_ratio_from_view(view):
    """Resolve the deterministic center-crop side ratio used by an evaluator.

    ``center80`` remains the public view name for backwards compatibility.  A
    sensitivity worker can set ``QUERY_REFINEMENT_CROP_RATIO`` to a finite
    value in (0, 1]; existing runs therefore retain exactly the old 0.8
    geometry when the variable is absent.
    """
    if view == 'identity':
        return None
    if view != 'center80':
        raise ValueError('Expected identity or center80 view')
    value = float(os.environ.get('QUERY_REFINEMENT_CROP_RATIO', '0.8'))
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError('QUERY_REFINEMENT_CROP_RATIO must lie in (0, 1]')
    return value

def center_pixels(pixels, view='identity'):
    if view == 'identity':
        return pixels
    ratio = crop_ratio_from_view(view)
    if pixels.ndim != 4 or not pixels.is_floating_point():
        raise ValueError('Expected floating NCHW pixels and identity/center80')
    h, w = pixels.shape[-2:]
    ch, cw = int(h*ratio), int(w*ratio)
    top, left = (h-ch)//2, (w-cw)//2
    return F.interpolate(pixels[..., top:top+ch, left:left+cw], size=(h,w),
                         mode='bilinear', align_corners=False, antialias=False)

def fit_directions(features, labels):
    x = torch.as_tensor(features).detach().to(device='cpu', dtype=torch.float32)
    y = torch.as_tensor(labels).detach().to(device='cpu', dtype=torch.float32)
    if y.ndim != 2 or not torch.all((y == 0) | (y == 1)):
        raise ValueError('Expected binary support labels [N,C]')
    if x.ndim == 2:
        x = x[:,None,:].expand(-1,y.shape[1],-1)
    if x.ndim != 3 or x.shape[:2] != y.shape or not torch.isfinite(x).all():
        raise ValueError('Invalid support embeddings')
    if (torch.linalg.vector_norm(x,dim=-1) < 1e-12).any():
        raise ValueError('Zero support embedding')
    x = F.normalize(x,dim=-1)
    positives = y.sum(0)
    negatives = len(y)-positives
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError('Each class needs both positive and negative supports')
    pm = (x*y[...,None]).sum(0)/positives[:,None]
    nm = (x*(1-y)[...,None]).sum(0)/negatives[:,None]
    difference = pm-nm
    norms = difference.norm(dim=-1)
    direction = F.normalize(difference,dim=-1)
    return dict(direction=direction,directions=direction,positive_means=pm,
                negative_means=nm,direction_norms=norms,
                positive_counts=positives.to(torch.int64),negative_counts=negatives.to(torch.int64),
                zero_direction_classes=norms < 1e-12)

def refine_query(query, direction, eta=ETA, preserve_norm=True):
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0.:
        raise ValueError('eta must be a finite nonnegative value')
    if eta == 0.:
        return query
    direction = torch.as_tensor(direction,device=query.device,dtype=query.dtype)
    norms = query.norm(dim=-1,keepdim=True)
    if (norms < 1e-12).any():
        raise ValueError('Zero original query')
    result = F.normalize(F.normalize(query,dim=-1)+eta*direction,dim=-1)
    if preserve_norm:
        result = result*norms
    return torch.where(direction.norm(dim=-1,keepdim=True)<1e-12,query,result)

def fuse_center_maps(identity_maps, center_maps, input_size):
    """Inverse crop in native original-image map frame, then coverage-weight mean.

    Native maps already include each adapter's inverse resize/crop. Their finite
    rectangle defines native coverage (BioViL-T has NaN crop margins). Center80
    has no contribution outside its geometric crop. No new map normalization.
    """
    import cv2
    output = []
    ih, iw = (input_size,input_size) if isinstance(input_size,int) else input_size
    ratio = float(os.environ.get('QUERY_REFINEMENT_CROP_RATIO', '0.8'))
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError('QUERY_REFINEMENT_CROP_RATIO must lie in (0, 1]')
    ch,cw = int(ih*ratio),int(iw*ratio)
    top,left = (ih-ch)//2,(iw-cw)//2
    for original,zoom in zip(identity_maps,center_maps):
        original,zoom = np.asarray(original),np.asarray(zoom)
        if original.shape != zoom.shape:
            raise ValueError('View map shapes differ')
        fused = original.copy()
        for k,(a,b) in enumerate(zip(original,zoom)):
            valid=np.isfinite(a)
            yy,xx=np.where(valid)
            if not len(xx):raise ValueError('Empty native coverage')
            y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
            if not valid[y0:y1,x0:x1].all():raise ValueError('Nonrectangular native coverage')
            if not np.array_equal(np.isfinite(b),valid):raise ValueError('Native view coverage differs')
            h,w=y1-y0,x1-x0
            cx=(np.arange(w,dtype=np.float32)+.5)*iw/w
            cy=(np.arange(h,dtype=np.float32)+.5)*ih/h
            goodx=(cx>=left)&(cx<left+cw)
            goody=(cy>=top)&(cy<top+ch)
            mx=(cx-left)/cw*w-.5
            my=(cy-top)/ch*h-.5
            gridx,gridy=np.meshgrid(mx,my)
            restored=cv2.remap(b[y0:y1,x0:x1].astype(np.float32),gridx,gridy,
                              cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            cover=goody[:,None]&goodx[None,:]
            region=fused[k,y0:y1,x0:x1]
            region[cover]=(a[y0:y1,x0:x1][cover]+restored[cover])*.5
        output.append(fused)
    return output

def self_test():
    x=torch.arange(3*10*10,dtype=torch.float32).reshape(1,3,10,10)
    assert center_pixels(x,'identity') is x
    assert torch.equal(center_pixels(x,'center80'),F.interpolate(x[:,:,1:9,1:9],(10,10),mode='bilinear',align_corners=False))
    f=fit_directions(torch.tensor([[1.,0.],[0.,1.]]),[[1],[0]])
    q=torch.tensor([[3.,4.]])
    assert refine_query(q,f['directions'],0) is q
    assert torch.allclose(refine_query(q,f['directions']).norm(dim=-1),q.norm(dim=-1))
    a=np.zeros((1,10,10),np.float32);b=np.full_like(a,2)
    z=fuse_center_maps([a],[b],10)[0]
    assert np.all(z[:,1:9,1:9]==1) and z.sum()==64
    a[:,0]=np.nan;b[:,0]=np.nan
    z=fuse_center_maps([a],[b],10)[0]
    assert np.isnan(z[:,0]).all() and np.isfinite(z[:,1:]).all()
    print('query refinement primitives: passed')

if __name__=='__main__':self_test()
