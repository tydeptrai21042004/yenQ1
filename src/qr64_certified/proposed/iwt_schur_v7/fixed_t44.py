"""Blind IWT--Schur 4x4 carrier selected by subband.
Algebraically ordered eigenvalues of symmetric 4x4 IWT blocks.
LL and HH use Schur T[4,4]; LH and HL use Schur T[1,1].
Extraction requires only received image, key, nonce, and public settings.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,hmac
import cv2
import numpy as np
from .iwt import iwt2,iiwt2

SIZE=512; WM=64; PAYLOAD=WM*WM; BLOCK=4; BANDSIZE=SIZE//2
NB=(BANDSIZE//BLOCK)**2; NGROUPS=12
NPILOT=128; PILOT_REPS=3

@dataclass(frozen=True)
class Config:
    ll_copies:int=2
    detail_copies:int=2
    q_ll:float=18.
    q_detail:float=12.
    q_pilot:float=32.
    iterations:int=5
    pilot_threshold:float=.69
    ll_vote_weight:float=6.
    # Algebraically ordered real eigenvalues, indices 0..3 for LL,LH,HL,HH.
    # Default corresponds to T44,T11,T11,T44. Controls only for ablations.
    carrier_indices:tuple[int,int,int,int]=(3,0,0,3)
    def __post_init__(self):
        assert 1<=self.ll_copies and self.ll_copies*PAYLOAD+NPILOT*PILOT_REPS <= 3*NB
        assert 0<=self.detail_copies and self.detail_copies*PAYLOAD <= 9*NB
        assert min(self.q_ll,self.q_detail,self.q_pilot)>0
        assert len(self.carrier_indices)==4 and all(0<=int(x)<=3 for x in self.carrier_indices)

@dataclass
class Detection:
    watermark:np.ndarray
    pilot_accuracy:float
    detected:bool
    raw_bits:np.ndarray

def read_watermark(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None:raise FileNotFoundError(path)
    if im.ndim==3 and im.shape[2]==4:
        alpha=im[:,:,3:4].astype(float)/255.
        bgr=np.uint8(np.rint(im[:,:,:3]*alpha+255*(1-alpha)))
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    elif im.ndim==3: gray=cv2.cvtColor(im[:,:,:3],cv2.COLOR_BGR2GRAY)
    else:gray=im
    if gray.shape!=(WM,WM):raise ValueError('Watermark must have dimensions 64x64')
    return np.uint8(gray>=128)*255

def _rng(key,nonce):
    v=hashlib.sha256(('IWT-Schur-4x4-band-T44-T11-v1|'+key+'|'+nonce).encode()).digest()
    return np.random.default_rng(int.from_bytes(v[:16],'big'))

def _pilot(key,nonce):
    v=hmac.new(key.encode(),('pilot-band-T44-T11-v1|'+nonce).encode(),hashlib.sha256).digest()
    return np.unpackbits(np.frombuffer(v[:16],dtype=np.uint8))

def slots(key,nonce,cfg):
    rng=_rng(key,nonce)
    ll=np.concatenate([np.arange(c*4*NB,c*4*NB+NB,dtype=np.int32) for c in range(3)])
    detail=np.concatenate([np.arange(c*4*NB+band*NB,c*4*NB+(band+1)*NB,dtype=np.int32) for c in range(3) for band in (1,2,3)])
    rng.shuffle(ll);rng.shuffle(detail)
    pids=ll[:NPILOT*PILOT_REPS]
    lli=ll[len(pids):len(pids)+PAYLOAD*cfg.ll_copies]
    det=detail[:PAYLOAD*cfg.detail_copies]
    ids=np.concatenate([lli,det]);assign=np.tile(np.arange(PAYLOAD,dtype=np.int32),cfg.ll_copies+cfg.detail_copies)
    order=rng.permutation(len(ids))
    return ids[order],assign[order],pids

def _blocks(mat):
    return mat.reshape(BANDSIZE//BLOCK,BLOCK,BANDSIZE//BLOCK,BLOCK).transpose(0,2,1,3).reshape(NB,BLOCK,BLOCK).copy()

def _unblocks(blocks):
    return blocks.reshape(BANDSIZE//BLOCK,BANDSIZE//BLOCK,BLOCK,BLOCK).transpose(0,2,1,3).reshape(BANDSIZE,BANDSIZE)

def _schur_carrier(blocks,band,carrier_indices=(3,0,0,3)):
    symmetric=.5*(blocks+blocks.transpose(0,2,1))
    eigenvalues, vectors=np.linalg.eigh(symmetric)
    # eigh: eigenvalues in *algebraic* ascending order, the same order used
    # to survey the energy distribution. LL, HH -> t44; LH, HL -> t11.
    k=carrier_indices[band]
    return eigenvalues[:,k],vectors[:,:,k]

def _quant_target(mag,bits,q,center=False):
    # A quantized pair of disjoint decision intervals at q/4 and 3q/4.
    off=q*(.25+.5*bits)
    if center:
        k=np.floor((mag-off)/q+.5)
        return np.maximum(off,off+k*q)
    margin=q*.17
    lo=np.where(bits!=0,q*.5+margin,margin)
    hi=np.where(bits!=0,q-margin,q*.5-margin)
    k=np.floor(mag/q)
    cand=np.stack([np.maximum(0.,np.clip(mag,(k+s)*q+lo,(k+s)*q+hi)) for s in (-1,0,1)])
    return cand[np.argmin(np.abs(cand-mag),axis=0),np.arange(len(mag))]

def _embed_blocks(blocks,bits,q,band,center=False,carrier_indices=(3,0,0,3)):
    eig,u=_schur_carrier(blocks,band,carrier_indices)
    mag=np.abs(eig)
    target=_quant_target(mag,bits,q,center=center)
    delta=np.where(eig<0,-1.,1.)*(target-mag)
    return np.rint(blocks + delta[:,None,None]*u[:,:,None]*u[:,None,:])

def _soft_blocks(blocks,q,band,carrier_indices=(3,0,0,3)):
    eig,_=_schur_carrier(blocks,band,carrier_indices)
    return np.cos(2*np.pi*(np.mod(np.abs(eig),q)/q-.75))

def _groups(image):
    return [part for c in range(3) for part in iwt2(image[:,:,c])]

def _image(groups):
    out=np.stack([iiwt2(*groups[4*c:4*c+4]) for c in range(3)],axis=2)
    return np.uint8(np.clip(np.rint(out),0,255))

def _q(ids,cfg):
    return np.where((ids//NB)%4==0,cfg.q_ll,cfg.q_detail)

def _apply(host,ids,bits,qs,center=False,cfg=None):
    carrier_indices=(cfg.carrier_indices if cfg is not None else (3,0,0,3))
    parts=_groups(host);gids=ids//NB;loc=ids%NB
    for group_id in np.unique(gids):
        where=gids==group_id
        blocks=_blocks(parts[int(group_id)])
        locs=loc[where];bs=bits[where];steps=qs[where]
        for step in np.unique(steps):
            which=steps==step
            blocks[locs[which]]=_embed_blocks(blocks[locs[which]],bs[which],float(step),band=int(group_id)%4,center=center,carrier_indices=carrier_indices)
        parts[int(group_id)]=_unblocks(blocks)
    return _image(parts)

def _read_scores(host,ids,qs,cfg=None):
    carrier_indices=(cfg.carrier_indices if cfg is not None else (3,0,0,3))
    parts=_groups(host);gids=ids//NB;loc=ids%NB
    out=np.empty(len(ids),dtype=float)
    for group_id in np.unique(gids):
        where=gids==group_id
        blocks=_blocks(parts[int(group_id)])[loc[where]]
        steps=qs[where]
        positions=np.flatnonzero(where)
        for step in np.unique(steps):
            which=steps==step
            out[positions[which]]=_soft_blocks(blocks[which],float(step),band=int(group_id)%4,carrier_indices=carrier_indices)
    return out

def _decode_bit_scores(image,ids,assign,cfg):
    qs=_q(ids,cfg)
    soft=_read_scores(image,ids,qs,cfg=cfg)
    weights=np.where((ids//NB)%4==0,cfg.ll_vote_weight,1.)
    return np.bincount(assign,weights=soft*weights,minlength=PAYLOAD)

def _pilot_info(im,pids,key,nonce,cfg):
    raw=_read_scores(im,pids,np.full(len(pids),cfg.q_pilot),cfg=cfg).reshape(PILOT_REPS,NPILOT)
    val=np.uint8(raw.sum(axis=0)>0)
    ac=float(np.mean(val==_pilot(key,nonce)))
    return ac

def embed(host,watermark,*,key='DEMO_KEY',nonce='pepper_wm',cfg=Config()):
    if host.shape!=(SIZE,SIZE,3) or host.dtype!=np.uint8:raise ValueError('Host must be 512x512 uint8 color image')
    if watermark.shape!=(WM,WM) or watermark.dtype!=np.uint8 or not np.all((watermark==0)|(watermark==255)):
        raise ValueError('Watermark must be 64x64 binary uint8')
    bits=(watermark.reshape(-1)//255).astype(np.uint8)
    ids,assign,pids=slots(key,nonce,cfg)
    allids=np.concatenate([ids,pids]);allbits=np.concatenate([bits[assign],np.tile(_pilot(key,nonce),PILOT_REPS)])
    qs=np.concatenate([_q(ids,cfg),np.full(len(pids),cfg.q_pilot)])
    marked=_apply(host,allids,allbits,qs,cfg=cfg)
    for _ in range(cfg.iterations):
        scores=_decode_bit_scores(marked,ids,assign,cfg)
        pilot=_pilot_info(marked,pids,key,nonce,cfg)
        bad=(np.uint8(scores>0)!=bits)
        if not np.any(bad) and pilot>=1.:break
        mask=np.concatenate([bad[assign],np.tile(np.ones(NPILOT,dtype=bool) if pilot<1 else np.zeros(NPILOT,dtype=bool),PILOT_REPS)])
        if np.any(mask): marked=_apply(marked,allids[mask],allbits[mask],qs[mask],center=True,cfg=cfg)
    return marked

def extract(received,*,key='DEMO_KEY',nonce='pepper_wm',cfg=Config()):
    ids,assign,pids=slots(key,nonce,cfg)
    scores=_decode_bit_scores(received,ids,assign,cfg)
    raw=_pilot_info(received,pids,key,nonce,cfg)
    bits=np.uint8(scores>0)
    if raw<.5:bits=1-bits
    ac=max(raw,1.-raw)
    return Detection((bits.reshape(WM,WM)*255).astype(np.uint8),ac,ac>=cfg.pilot_threshold,bits)

def psnr(a,b):
    mse=np.mean((a.astype(float)-b.astype(float))**2)
    return 10*np.log10(255**2/mse) if mse else float('inf')

def nc(a,b):
    x=(a.astype(float)/255).ravel();y=(b.astype(float)/255).ravel()
    v=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/v) if v else float(np.array_equal(x,y))

def ber(a,b):return float(np.mean(a!=b))
