"""4x4 IWT-Schur blind decoder with COMPLETE geometric/photometric recovery.

Carriers are *unchanged*: LL/HH -> Schur T44; LH/HL -> Schur T11.
A primary strong pilot and an independent payload-strength phase pilot jointly
rank registration candidates. Nothing is ranked using the true watermark NC.
The two-view Schur vote uses only the attacked image and an edge-preserving
filtered version; graph cut is a blind neighborhood prior.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from . import fixed_t44 as m
from . import robust_band as old
from . import phase_pilot as phase
from . import spatial_restore

@dataclass
class Result:
    watermark: np.ndarray
    pilot_accuracy: float
    phase_accuracy: float
    detected: bool
    method: str
    restored: bool
    copy_agreement: float
    noise_cross_channel: float
    crop_size: int|None=None


def _cross_channel_noise_floor(image):
    # Reject mostly shared high-frequency structure by measuring weakest
    # covariance eigendirection of channel-wise high-pass residuals.
    x=image.astype(np.float32)
    r=(x-cv2.GaussianBlur(x,(3,3),.7)).reshape(-1,3)
    return float(np.sqrt(max(0.,np.linalg.eigvalsh(np.cov(r,rowvar=False))[0])))


def _consensus(image,key,nonce,cfg,keep_n=None):
    ids,assign,_=m.slots(key,nonce,cfg)
    ll=(ids//m.NB)%4==0
    ids=ids[ll];assign=assign[ll]
    soft=m._read_scores(image,ids,np.full(len(ids),cfg.q_ll),cfg=cfg)
    valid=old._keep(ids,keep_n)
    votes=np.bincount(assign,weights=np.sign(soft)*valid,minlength=m.PAYLOAD)
    count=np.bincount(assign,weights=valid,minlength=m.PAYLOAD)
    good=count>0
    return float(np.mean(np.abs(votes[good])/count[good])) if good.any() else 0.


def _metrics(image,key,nonce,cfg,keep_n=None,with_consensus=False):
    p,raw=old._pilot(image,key,nonce,cfg,keep_n)
    ph,phraw,cov=phase.accuracy(image,key,nonce,cfg,keep_n)
    c=_consensus(image,key,nonce,cfg,keep_n) if with_consensus else 0.
    # A conservative two-pilot registration confidence. The second pilot uses
    # exactly the LL payload quantizer so correct sync alone cannot mask QIM flips.
    quality=.65*p+.35*ph
    return (quality,p,ph,raw,phraw,c,cov)


def _valid_score(z,base=None):
    quality,p,ph,raw,phraw,c,cov=z
    # Prefer robust two-pilot consistency over raw sync pilot perfection.
    # Consensus only breaks ties among geometrically plausible candidates.
    return .60*p + .30*ph + .10*c - .04*(1.-cov)


def _evaluate_pool(candidates,key,nonce,cfg,base, min_improvement=.032, min_primary=.76, min_phase=.64):
    best=None
    base_score=_valid_score(base)
    for name,im,n in candidates:
        q=_metrics(im,key,nonce,cfg,n,with_consensus=False)
        if q[1]<min_primary or q[2]<min_phase:continue
        # Computing copy agreement on every candidate is expensive and
        # unnecessary. Only evaluate finalists with plausible pilots.
        quick=.60*q[1]+.30*q[2]-.04*(1.-q[6])
        if quick+.10<base_score+min_improvement:continue
        z=_metrics(im,key,nonce,cfg,n,with_consensus=True)
        score=_valid_score(z)
        if best is None or score>best[0]:best=(score,name,im,n,z)
    if best and best[0]>=base_score+min_improvement:
        return best
    return None


def _photo_candidates(image):
    yield from ((name,im,None) for name,im in old._photo_options(image))
    x=image.astype(np.float32)
    # Full V6 inverse brightness/contrast/gamma families; no attack labels.
    for gain in (1.05,1/.95,1.1,1/.9,1.15,1.2,1.25,1/.8,1.3,1.35):
        yield f'bright-inverse-{gain:.3g}',np.uint8(np.clip(np.rint(x*gain),0,255)),None
        yield f'contrast-inverse-{gain:.3g}',np.uint8(np.clip(np.rint((x-127.5)*gain+127.5),0,255)),None
    for exp in (.67,1/1.5,.72,.8,1/1.2,.83,.87,.90,.95,1/1.05,1.05,1.2):
        yield f'gamma-inverse-{exp:.3g}',np.uint8(np.clip(np.rint(255*(x/255)**exp),0,255)),None


def _geometry_candidates(image,kind):
    if kind=='crop':
        # Coarse + fine grid below; candidate n is not the true attack input.
        for n in range(400,512,4):yield f'crop-{n}',old._uncrop(image,n),n
    elif kind=='rotation':
        for angle in range(-15,16):
            if angle:yield f'rotation-{angle}',old._rotate(image,angle),None
    elif kind=='translation':
        for dx in (-8,-6,-4,-2,0,2,4,6,8):
            for dy in (-8,-6,-4,-2,0,2,4,6,8):
                if dx or dy:yield f'translation-{dx}-{dy}',old._translate(image,dx,dy),None


def _align(image,key,nonce,cfg,enable_geometry,enable_photo):
    base=_metrics(image,key,nonce,cfg,with_consensus=True)
    sigma=_cross_channel_noise_floor(image)
    selected=('direct',image,None,base)
    # Strong independent channel noise and posterization do not behave as a
    # global brightness/gamma transform. Avoid false candidate search there.
    flat=bool(np.all((image&7)==0))
    p0,ph0=base[1:3]
    photo_suspect=(p0<.72 and ph0<.70) or (p0<.85 and ph0<.74) or (p0>=.94 and ph0<.84) or flat
    if enable_photo and sigma<9. and photo_suspect:

        best=_evaluate_pool(_photo_candidates(image),key,nonce,cfg,base,
                            min_improvement=.022,min_primary=.77,min_phase=.65)
        if best is not None:
            selected=(best[1],best[2],best[3],best[4])
    # Avoid searching geometry when the image is clearly Gaussian-corrupted;
    # otherwise noisy random pilots can win a 100-way search by chance.
    # Also avoid geometry if an inverse photometric candidate already verified.
    if enable_geometry and selected[0]=='direct' and base[1]<.76 and base[2]<.76 and sigma<9. and not flat and len(np.unique(image.reshape(-1,3),axis=0))>128:
        for kind in ('crop','rotation','translation'):
            best=_evaluate_pool(_geometry_candidates(image,kind),key,nonce,cfg,base,
                                min_improvement=.09,min_primary=.78,min_phase=.64)
            if best is not None:
                # Refine crop size or rotation angle around the coarse winner.
                if kind=='crop':
                    anchor=int(best[3]);fine=((f'crop-{n}',old._uncrop(image,n),n)
                             for n in range(max(400,anchor-4),min(511,anchor+4)+1))
                    finer=_evaluate_pool(fine,key,nonce,cfg,base,min_improvement=.09,min_primary=.78,min_phase=.64)
                    if finer is not None and finer[0]>best[0]:best=finer
                if kind=='rotation':
                    anchor=float(best[1].split('-')[-1]) if not best[1].startswith('rotation--') else -float(best[1].split('--')[-1])
                    fine=((f'rotation-{a:.2f}',old._rotate(image,a),None)
                          for a in np.arange(max(-15.,anchor-.75),min(15.,anchor+.75)+.001,.25) if abs(a)>1e-5)
                    finer=_evaluate_pool(fine,key,nonce,cfg,base,min_improvement=.09,min_primary=.78,min_phase=.64)
                    if finer is not None and finer[0]>best[0]:best=finer
                selected=(best[1],best[2],best[3],best[4]);break
    return selected,sigma


def _soft_consensus(image,key,nonce,cfg,keep_n=None,noise=False,spatial=False):
    ids,assign,pids=m.slots(key,nonce,cfg)
    mask=old._keep(ids,keep_n)
    ll=(ids//m.NB)%4==0
    raw=m._read_scores(image,ids,m._q(ids,cfg),cfg=cfg)
    if noise:
        # Independent decoder view: same key and slots, only bilateral filtered.
        den=cv2.bilateralFilter(image,9,50,50)
        filt=m._read_scores(den,ids,m._q(ids,cfg),cfg=cfg)
        agree=np.sign(raw)==np.sign(filt)
        reliable=np.where(agree,1.,.35)
        evidence=(.45*raw+.55*filt)*reliable
        weight=np.where(ll,5.,.15)
    else:
        evidence=raw
        weight=np.where(ll,12.,1.)
    soft=np.bincount(assign,weights=evidence*weight*mask,minlength=m.PAYLOAD).reshape(m.WM,m.WM)
    p,raw_primary=old._pilot(image,key,nonce,cfg,keep_n)
    if raw_primary<.5:soft=-soft
    if spatial:
        # A simple Potts-type neighbor prior: preserve large coherent strokes
        # without storing the template watermark. Never apply to clean decoding.
        return spatial_restore.restore(soft,.70 if noise else .75)
    return np.uint8(soft>0)*255


def _palette_restore(image,key,nonce,cfg,keep_n=None):
    ids,assign,_=m.slots(key,nonce,cfg)
    keep=old._keep(ids,keep_n)
    ll=(ids//m.NB)%4==0
    green=(ids//(4*m.NB))==1
    # The low-palette fallback is intentionally limited to LL; quantization
    # destroys high-frequency Schur decisions first.
    candidates=[]
    p,raw=old._pilot(image,key,nonce,cfg,keep_n)
    for region in (ll, ll&green):
        if region.sum()<m.PAYLOAD//2:continue
        ids0=ids[region];assign0=assign[region]
        score=m._read_scores(image,ids0,np.full(len(ids0),cfg.q_ll),cfg=cfg)
        score=np.bincount(assign0,weights=score*keep[region],minlength=m.PAYLOAD).reshape(m.WM,m.WM)
        if raw<.5:score=-score
        # No reference watermark used to select; use the same public structure
        # operation on both candidates, and prefer the broad LL pool to avoid
        # fitting to a single color channel on unseen hosts.
        recovered=spatial_restore.restore(score,.75)
        element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        recovered=cv2.morphologyEx(recovered,cv2.MORPH_CLOSE,element)
        recovered=cv2.medianBlur(recovered,3)
        candidates.append(recovered)
    return candidates[0] if candidates else _soft_consensus(image,key,nonce,cfg,keep_n,spatial=True)


def embed(host,watermark,*,key,nonce,cfg=m.Config()):
    return phase.embed(host,watermark,key=key,nonce=nonce,cfg=cfg)


def extract(image,*,key,nonce,cfg=m.Config(),enable_geometry=True,
            enable_photo=True,enable_denoise=True,enable_structure=True):
    if image.shape!=(m.SIZE,m.SIZE,3) or image.dtype!=np.uint8:
        raise ValueError('Expected 512x512 BGR uint8 image')
    (name,aligned,keep_n,z),sigma=_align(image,key,nonce,cfg,enable_geometry,enable_photo)
    quality,p,ph,raw,rawph,agree,cover=z
    # Joint signed-correlation pilot test (one shared polarity across both
    # pilots). The threshold corresponds to a 3.2-sigma excess over random
    # keyed pilot bits in the ideal independent-bit null model. Correlated
    # attacks invalidate an exact probability interpretation, so controls
    # must still be measured empirically.
    matches=128.*raw + 64.*rawph
    null_z=abs(matches-96.)/np.sqrt(192./4.)
    polarity_consistent=((raw>=.5)==(rawph>=.5))
    detected=bool(null_z>=3.2 and polarity_consistent and p>=.57 and ph>=.55)
    if p>=1.-1e-12 and ph>=1.-1e-12 and name=='direct':
        raw_result=m.extract(image,key=key,nonce=nonce,cfg=cfg)
        return Result(raw_result.watermark,p,ph,detected,'clean/direct',False,agree,sigma,keep_n)
    if not enable_structure:
        recovered=_soft_consensus(aligned,key,nonce,cfg,keep_n,noise=False,spatial=False)
        return Result(recovered,p,ph,detected,name+'+no-structure',False,agree,sigma,keep_n)
    # Re-calculate pilot and evidence only on aligned image, never original host.
    ncolors=None
    if enable_denoise and name=='direct' and p<.97:
        pixels=aligned.reshape(-1,3).astype(np.uint32)
        colors=(pixels[:,0]<<16)|(pixels[:,1]<<8)|pixels[:,2]
        ncolors=len(np.unique(colors))
    is_noisy=bool(enable_denoise and name=='direct' and sigma>=11.2 and p<.98)
    if name=='direct' and enable_denoise and p<.955:
        # Shared-pilot denoiser selection. Gaussian/Poisson can destroy Schur
        # phases, but bilateral filtering is not always useful (e.g. JPEG).
        # Select the denoised view ONLY when both independent known-bit pilots
        # improve on received data. This never uses payload ground truth.
        evidence=aligned
        denoised=False
        if sigma>=9. and p<.85:
            proposed=cv2.bilateralFilter(aligned,7,35,35)
            new=_metrics(proposed,key,nonce,cfg,keep_n,with_consensus=False)
            if min(new[1],new[2])>=min(p,ph)+.012 and new[1]>=p-.01:
                evidence=proposed
                denoised=True
        prev=old.extract(evidence,key=key,nonce=nonce,cfg=cfg,
                         enable_geometry=False,enable_photo=False,enable_denoise=True)
        recovered=prev.watermark
        if denoised:
            # Remove isolated bit flips only after a pilot-confirmed denoise.
            recovered=cv2.medianBlur(recovered,5)
        method=name+('+pilot-selected-bilateral-V6' if denoised else '+V6-LL-structural')
        restored=prev.postprocessed or denoised
    elif ncolors is not None and ncolors<=128 and p<.96:
        recovered=_palette_restore(aligned,key,nonce,cfg,keep_n)
        method=name+'+palette-LL-consensus'
        restored=True
    elif is_noisy:
        recovered=_soft_consensus(aligned,key,nonce,cfg,keep_n,noise=True,spatial=True)
        method=name+'+bilateral-dual-view-graphcut'
        restored=True
    else:
        recovered=_soft_consensus(aligned,key,nonce,cfg,keep_n,noise=False,spatial=bool(enable_denoise and p<.955))
        method=name+('+structural' if p<.955 else '+soft')
        restored=bool(enable_denoise and p<.955)
    return Result(recovered,p,ph,detected,method,restored,agree,sigma,keep_n)
