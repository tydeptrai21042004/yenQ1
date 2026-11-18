#!/usr/bin/env python3
"""Paired, host-clustered comparison of one proposal with unchanged baselines.

Joins exact host/watermark/attack/severity keys. Reports missing coverage rather
than silently comparing unmatched rows. Inference is omitted for one host-pair.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def load(path):
    with open(path,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write(path,rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def key(r):
    return (r.get('host'),r.get('watermark'),r.get('attack'),r.get('level'),
            r.get('condition_id'),r.get('protocol_sha256'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',required=True)
    p.add_argument('--baselines',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=451)
    a=p.parse_args()
    pr=[r for r in load(a.proposal) if r.get('status')=='ok']
    bl=[r for r in load(a.baselines) if r.get('status')=='ok']
    if any(not r.get('protocol_sha256') or not r.get('condition_id') for r in pr+bl):
        raise SystemExit('Paired statistics requires protocol IDs and condition IDs in both inputs')
    proj={key(r):r for r in pr} 
    baselines=sorted({r.get('method_id') for r in bl})
    rows=[];rng=np.random.default_rng(a.seed)
    for method in baselines:
        baseline={key(r):r for r in bl if r.get('method_id')==method}
        shared=sorted(set(proj)&set(baseline))
        all_pr=set(proj);all_bl=set(baseline)
        # Keep clean attacks separate from attacked comparisons.
        for metric in ('attacked_nc','attacked_ber'):
            deltas={}
            for k in shared:
                if k[2]=='none':continue
                try:delta=float(proj[k][metric])-float(baseline[k][metric])
                except (KeyError,ValueError,TypeError):continue
                # A host/watermark combination, not an attack row, is the
                # resampling unit. Avoid spurious precision from 60 attacks.
                deltas.setdefault(k[:2],[]).append(delta)
            values=np.array([np.mean(v) for v in deltas.values()],dtype=float)
            n=len(values)
            if n>1:
                sampled=rng.integers(0,n,size=(a.bootstrap,n))
                boot=values[sampled].mean(axis=1)
                ci_lo,ci_hi=map(float,np.quantile(boot,[.025,.975]))
                # Approximate two-sided sign-flip permutation test on host pairs.
                flips=rng.choice([-1,1],size=(a.bootstrap,n))
                null=np.mean(values[None,:]*flips,axis=1)
                obs=abs(float(np.mean(values)))
                pval=(np.sum(np.abs(null)>=obs)+1)/(len(null)+1)
            else:ci_lo=ci_hi=pval=None
            rows.append({'baseline':method,'metric':metric,'n_host_watermark_pairs':n,
                         'n_attacks_matched':sum(len(v) for v in deltas.values()),
                         'mean_proposal_minus_baseline':float(values.mean()) if n else None,
                         'ci95_low':ci_lo,'ci95_high':ci_hi,'sign_flip_p_approx':pval,
                         'proposal_rows_missing_from_baseline':len(all_pr-all_bl),
                         'baseline_rows_missing_from_proposal':len(all_bl-all_pr),
                         'inference_valid':bool(n>1 and len(all_pr-all_bl)==0 and len(all_bl-all_pr)==0)})
    # Familywise Holm adjustment only over eligible tests (without treating
    # incomplete matching as valid comparisons).
    indices=sorted([i for i,r in enumerate(rows) if r['inference_valid'] and r['sign_flip_p_approx'] is not None],
                   key=lambda i:rows[i]['sign_flip_p_approx'])
    prev=0
    for rank,i in enumerate(indices):
        pval=rows[i]['sign_flip_p_approx'];adj=min(1.,max(prev,(len(indices)-rank)*pval))
        rows[i]['p_holm_approx']=adj;prev=adj
    write(Path(a.out)/'paired_statistics.csv',rows)
    print('PAIRED comparisons:',len(rows),'pairs:',sorted({r['n_host_watermark_pairs'] for r in rows}))
if __name__=='__main__':main()
