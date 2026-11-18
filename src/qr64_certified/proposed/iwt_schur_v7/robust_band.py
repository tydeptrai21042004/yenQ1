"""Attack-agnostic blind synchronization and soft-bit decoding for 4x4 IWT-Schur.

All calibration uses key-derived pilot bits and confidence of repeated copies.
No original image, target watermark or attack name is used at extraction.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from . import fixed_t44 as m
from . import spatial_restore

@dataclass
class RobustResult:
    watermark:np.ndarray
    pilot_accuracy:float
    detected:bool
    mode:str
    postprocessed:bool


def _pilot(image,key,nonce,cfg,valid_n=None):
    _,_,pids=m.slots(key,nonce,cfg)
    votes=m._read_scores(image,pids,np.full(len(pids),cfg.q_pilot),cfg=cfg).reshape(m.PILOT_REPS,m.NPILOT)
    ok=_keep(pids,valid_n).reshape(m.PILOT_REPS,m.NPILOT)
    valid=(ok.sum(axis=0)>0)
    if not np.any(valid):return 0.,0.
    got=(votes*ok).sum(axis=0)>0
    original=float(np.mean(got[valid]==m._pilot(key,nonce)[valid]))
    return max(original,1-original),original

def _keep(ids,keep_n=None):
    if keep_n is None:return np.ones(len(ids),dtype=bool)
    off=(m.SIZE-int(keep_n))//2
    loc=ids % m.NB
    x=(loc%64)*8+4
    y=(loc//64)*8+4
    return (x>=off+4)&(y>=off+4)&(x<off+int(keep_n)-4)&(y<off+int(keep_n)-4)

def _confidence(image,key,nonce,cfg,valid_n=None):
    ids,assignment,_=m.slots(key,nonce,cfg)
    ll=(ids//m.NB)%4==0
    scores=m._read_scores(image,ids[ll],np.full(int(ll.sum()),cfg.q_ll),cfg=cfg)
    mask=_keep(ids[ll],valid_n)
    sums=np.bincount(assignment[ll],weights=np.sign(scores)*mask,minlength=m.PAYLOAD)
    counts=np.bincount(assignment[ll],weights=mask,minlength=m.PAYLOAD)
    return float(np.sum(np.abs(sums))/(np.sum(counts)+1e-9))

def _read(image,key,nonce,cfg,valid_n=None,prefer_ll=True,spatial=False):
    ids,ass,pids=m.slots(key,nonce,cfg)
    sc=m._read_scores(image,ids,m._q(ids,cfg),cfg=cfg)
    valid=_keep(ids,valid_n)
    ll=(ids//m.NB)%4==0
    w=np.where(ll,12. if prefer_ll else cfg.ll_vote_weight,1.)*valid
    vals=np.bincount(ass,weights=sc*w,minlength=m.PAYLOAD).reshape(m.WM,m.WM)
    pa,raw=_pilot(image,key,nonce,cfg,valid_n)
    if raw<.5: vals=-vals
    bits=np.uint8(vals>0)*255
    if spatial:
        # Local agreement is a blind structural prior; do not apply it to clean recovery.
        bits=cv2.medianBlur(bits,3)
    return RobustResult(bits,pa,pa>=cfg.pilot_threshold,'direct',spatial)

def _rotate(image,angle):
    mat=cv2.getRotationMatrix2D((m.SIZE/2,m.SIZE/2),float(angle),1)
    return cv2.warpAffine(image,mat,(m.SIZE,m.SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def _translate(image,dx,dy):
    return cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(m.SIZE,m.SIZE),borderMode=cv2.BORDER_REFLECT_101)

def _uncrop(image,size):
    size=int(size);pad=(m.SIZE-size)//2
    small=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(small,pad,m.SIZE-size-pad,pad,m.SIZE-size-pad,cv2.BORDER_REFLECT_101)

def _photo_options(image):
    x=image.astype(np.float32)
    for gain in (1.04,1.08,1.12,1.16,1.20,1.25,1.32):
        yield f'brightness-x{gain}',np.uint8(np.clip(np.rint(x*gain),0,255))
        yield f'contrast-x{gain}',np.uint8(np.clip(np.rint((x-127.5)*gain+127.5),0,255))
    for power in (1/1.5,.66,.75,.80,1/1.2,.85,.9,.95,1/1.05,1.05,1.1,1.2):
        yield f'gamma-x{power}',np.uint8(np.clip(np.rint(255*(x/255)**power),0,255))
    # 5..7-bit truncation: replace each quantization bin by center, only when
    # the distribution of received RGB values shows hard lattice levels.
    for sh in (1,2,3):
        step=1<<sh
        if bool(np.all((image&(step-1))==0)):
            yield f'bincenter-{step}',np.uint8(np.clip(image.astype(np.int16)+step//2,0,255))

def _choose(candidates,key,nonce,cfg,base_pilot):
    # Hard gate on pilot consensus. This is not a search based on true watermark NC.
    viable=[]
    for name,im,n in candidates:
        p,_=_pilot(im,key,nonce,cfg,n)
        if p>=max(.78,base_pilot+.10):
            conf=_confidence(im,key,nonce,cfg,n)
            viable.append((p+.04*conf,name,im,n,p))
    if not viable:return None
    return max(viable,key=lambda z:z[0])

def extract(image,*,key,nonce,cfg=m.Config(),enable_geometry=True,enable_photo=True,enable_denoise=True):
    if image.shape!=(m.SIZE,m.SIZE,3) or image.dtype!=np.uint8:raise ValueError('Expect 512x512 BGR uint8')
    pa,_=_pilot(image,key,nonce,cfg)
    # Preserve the exact clean extraction guarantee of the embedding decoder.
    # It is safe to use the original soft-vote weighting when the keyed pilot
    # is fully consistent; no reference watermark is consulted.
    if pa >= 1.-1e-12:
        raw=m.extract(image,key=key,nonce=nonce,cfg=cfg)
        if raw.detected:
            return RobustResult(raw.watermark,raw.pilot_accuracy,True,'pilot-clean/direct',False)
    aligned=image;chosen='direct';keep_n=None;align_p=pa
    # Photo correction is low-cost, but activated only when key pilot disagrees.
    if enable_photo and pa<.98:
        cand=_choose(((name,im,None) for name,im in _photo_options(image)),key,nonce,cfg,pa)
        if cand is not None:
            _,chosen,aligned,keep_n,align_p=cand
    # Geometric registration is comparatively expensive. Only run when pilot
    # fails strongly AND no photometric correction was verified.
    if enable_geometry and chosen=='direct' and pa<.76:
        candidates=[]
        for angle in (-15,-12,-10,-8,-5,-3,3,5,8,10,12,15):
            candidates.append((f'rotate-{angle}',_rotate(image,angle),None))
        cand=_choose(candidates,key,nonce,cfg,pa)
        if cand is not None:
            _,chosen,aligned,keep_n,align_p=cand
    if enable_geometry and chosen=='direct' and pa<.76:
        cand=_choose(((f'crop-{n}',_uncrop(image,n),n) for n in range(408,511,4)),key,nonce,cfg,pa)
        if cand is not None:
            _,chosen,aligned,keep_n,align_p=cand
    if enable_geometry and chosen=='direct' and pa<.76:
        cand=_choose(((f'translate-{dx}-{dy}',_translate(image,dx,dy),None)
                      for dx in (-8,-4,-2,0,2,4,8) for dy in (-8,-4,-2,0,2,4,8) if (dx or dy)),key,nonce,cfg,pa)
        if cand is not None:
            _,chosen,aligned,keep_n,align_p=cand
    # Clean/near-clean signal passes untouched. The spatial postfilter is only
    # used when pilot errors indicate damage.
    result=_read(aligned,key,nonce,cfg,keep_n,prefer_ll=True,spatial=False)
    result.mode=chosen
    if enable_denoise and result.pilot_accuracy<.95:
        candidates=[(result.pilot_accuracy,'direct',result)]
        den=cv2.bilateralFilter(aligned,7,35,35)
        filtered=_read(den,key,nonce,cfg,keep_n,prefer_ll=True,spatial=True)
        candidates.append((filtered.pilot_accuracy,'bilateral-median3',filtered))
        # A 3x3 median on the bit grid preserves most foreground strokes.
        med=_read(aligned,key,nonce,cfg,keep_n,prefer_ll=True,spatial=True)
        candidates.append((med.pilot_accuracy,'median3',med))
        best=max(candidates,key=lambda x:x[0])
        if best[0]>=result.pilot_accuracy+.02:
            result=best[2];result.mode=f'{chosen}+{best[1]}'
        elif result.pilot_accuracy<.90:
            # Keep a conservative local restoration when no denoiser gives a
            # confident pilot gain, except on unverified images.
            if result.detected:
                result.watermark=cv2.medianBlur(result.watermark,3)
                result.postprocessed=True
    # Structural reconstruction from the SAME blind Schur votes. It is
    # switched on only for severely degraded images, never for clean images.
    # All thresholds are public and independent of the reference watermark.
    if enable_denoise and chosen=='direct' and result.pilot_accuracy<.95:
        pixels=image.reshape(-1,3).astype(np.uint32)
        codes=(pixels[:,0]<<16)|(pixels[:,1]<<8)|pixels[:,2]
        ncolors=len(np.unique(codes))
        few_colors=ncolors<=128
        hh=np.concatenate([m._groups(image)[j].ravel() for j in (3,7,11)])
        noise=float(np.median(np.abs(hh-np.median(hh)))/(.6745*2))
        if few_colors or noise>=12. or result.pilot_accuracy<.945:
            ids,ass,_=m.slots(key,nonce,cfg)
            ll=(ids//m.NB)%4==0
            # The preprocessing choice follows observed noise / color count,
            # not the known attack type or true watermark NC.
            if ncolors<=40:
                evidence=cv2.fastNlMeansDenoisingColored(image,None,12,12,7,21)
                smoothness=.85
            elif noise>=12.:
                evidence=cv2.bilateralFilter(image,7,35,35)
                smoothness=.85
            else:
                evidence=image
                smoothness=1. if ncolors<=64 else .65
            soft=m._read_scores(evidence,ids[ll],np.full(int(ll.sum()),cfg.q_ll),cfg=cfg)
            score=np.bincount(ass[ll],weights=soft,minlength=m.PAYLOAD).reshape(m.WM,m.WM)
            raw_pilot=_pilot(image,key,nonce,cfg)[1]
            if raw_pilot<.5:score=-score
            result.watermark=spatial_restore.restore(score,smoothness)
            result.postprocessed=True
            result.mode=f'{chosen}+LL-structural-consensus'
    return result
