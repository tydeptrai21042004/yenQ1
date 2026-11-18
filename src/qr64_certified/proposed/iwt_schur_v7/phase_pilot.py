"""Second, payload-strength keyed pilot for blind photometric/phase calibration.

All its locations and bits are derived from the key/nonce, without sidecar.
The pilot is disjoint from both primary data sites and sync-pilot sites.
"""
from __future__ import annotations
import numpy as np
from . import fixed_t44 as m

N_PHASE=64
PHASE_REPS=2

def phase_slots(key,nonce,cfg):
    ids,_,pids=m.slots(key,nonce,cfg)
    used=np.zeros(m.NGROUPS*m.NB,dtype=bool)
    used[ids]=True; used[pids]=True
    candidates=np.flatnonzero(~used)
    candidates=candidates[((candidates//m.NB)%4)==0] # only LL
    rng=m._rng(key,nonce+'|independent-payload-phase-v7')
    rng.shuffle(candidates)
    if len(candidates)<N_PHASE*PHASE_REPS:
        raise ValueError('Not enough free LL sites for phase pilot')
    return candidates[:N_PHASE*PHASE_REPS]

def expected(key,nonce):
    return m._pilot(key,nonce)[:N_PHASE]

def accuracy(image,key,nonce,cfg,valid_n=None):
    from .robust_band import _keep
    inds=phase_slots(key,nonce,cfg)
    votes=m._read_scores(image,inds,np.full(len(inds),cfg.q_ll),cfg=cfg).reshape(PHASE_REPS,N_PHASE)
    valid=_keep(inds,valid_n).reshape(PHASE_REPS,N_PHASE)
    seen=valid.sum(axis=0)>0
    if not seen.any():return 0.,0.,0.
    predicted=(votes*valid).sum(axis=0)>0
    raw=float(np.mean(predicted[seen]==expected(key,nonce)[seen]))
    return max(raw,1.-raw),raw,float(seen.mean())

def embed(host,watermark,*,key,nonce,cfg):
    marked=m.embed(host,watermark,key=key,nonce=nonce,cfg=cfg)
    inds=phase_slots(key,nonce,cfg)
    desired=np.tile(expected(key,nonce),PHASE_REPS)
    steps=np.full(len(inds),cfg.q_ll)
    marked=m._apply(marked,inds,desired,steps,center=True,cfg=cfg)
    bits=(watermark.ravel()>0)
    data_ids,assignment,pilot_ids=m.slots(key,nonce,cfg)
    for _ in range(4):
        raw=m.extract(marked,key=key,nonce=nonce,cfg=cfg)
        ph,_,_=accuracy(marked,key,nonce,cfg)
        if np.array_equal(raw.watermark,watermark) and ph>=1.:
            break
        # Repair only sites corresponding to bits still incorrect and phase pilot.
        bad=raw.raw_bits!=bits
        if bad.any():
            subset=np.isin(assignment,np.flatnonzero(bad))
            marked=m._apply(marked,data_ids[subset],bits[assignment[subset]].astype(np.uint8),m._q(data_ids[subset],cfg),center=True,cfg=cfg)
        if ph<1.:
            votes=m._read_scores(marked,inds,steps,cfg=cfg).reshape(PHASE_REPS,N_PHASE)
            bad_phase=((votes.sum(axis=0)>0)!=expected(key,nonce))
            if bad_phase.any():
                sel=np.tile(bad_phase,PHASE_REPS)
                marked=m._apply(marked,inds[sel],desired[sel],steps[sel],center=True,cfg=cfg)
    return marked
