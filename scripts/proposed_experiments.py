#!/usr/bin/env python3
"""One research proposal: blind IWT-Schur 4x4, LL/HH T44, LH/HL T11.

All result artifacts are generated only when a command is executed, and never
bundled with the distributed source. The original baseline registry, engines,
and implementations are imported but never patched or overwritten.

Research commands: attacks, energy, benchmark, ablation, sweep, payload,
security, baselines, analyze, all. The default protocol is 20 x 3 with WebP.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import cv2
import numpy as np
from qr64_certified.common.io import load_host_rgb, load_watermark_binary
from qr64_certified.common.metrics import psnr, ssim, watermark_metrics
from qr64_certified.baselines.registry import (
    ALL_BASELINE_IDS, get_baseline_spec, embed_baseline, extract_baseline,
    normalize_baseline_id,
)
from qr64_certified.proposed.iwt_schur_v7 import fixed_t44 as m
from qr64_certified.proposed.iwt_schur_v7 import robust_v7 as method
from qr64_certified.proposed.iwt_schur_v7 import robust_band as onepilot
from qr64_certified.proposed.iwt_schur_v7 import attacks

DEFAULT_CONFIG=ROOT/'configs/iwt_schur_v7.json'
RESULT_COLUMNS=(
    'experiment','method_id','variant','baseline_blindness','host','watermark',
    'key_fingerprint','nonce','seed','category','attack','level','parameter',
    'status','error','clean_psnr_db','clean_ssim','clean_nc','clean_ncc',
    'clean_ber','attacked_nc','attacked_ncc','attacked_ber','bit_accuracy',
    'detected','pilot_accuracy','phase_accuracy','extract_mode','embed_seconds',
    'extract_seconds','pass_nc_085','accepted_and_pass_nc_085',
)


def write_rows(path:Path,rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(list(RESULT_COLUMNS)+[field for row in rows for field in row]))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def cfg_from_json(obj):
    fields=set(m.Config.__dataclass_fields__)
    params={k:v for k,v in obj.items() if k in fields}
    if 'carrier_indices' in params:params['carrier_indices']=tuple(params['carrier_indices'])
    return m.Config(**params)


def load_setup(args):
    settings=read_json(args.config)
    cfg=cfg_from_json(settings)
    key=args.key or settings['key']
    if key=='IWT_SCHUR_RESEARCH_KEY_CHANGE_ME':
        print('NOTICE: change the default study key for a private deployment.',flush=True)
    return cfg,key


def select_hosts(text):
    paths=sorted((ROOT/'data/host').glob('*.bmp'))
    if text.lower()=='all': return paths
    allowed={p.stem.lower():p for p in paths}
    return [allowed[s.strip().removesuffix('.bmp').lower()] for s in text.split(',')]


def select_watermarks(text):
    paths=sorted((ROOT/'data/watermark').glob('*.png'))
    if text.lower()=='all':return paths
    allowed={p.stem.lower():p for p in paths}
    return [allowed[s.strip().removesuffix('.png').lower()] for s in text.split(',')]


def load_host(path):
    # The V7 algorithm uses BGR arrays. Published baseline API uses RGB arrays.
    return cv2.imread(str(path),cv2.IMREAD_COLOR)


def load_wm(path):return m.read_watermark(path)


def fingerprint(key):return hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]


def derive_nonce(host,watermark,seed):
    return f'{host.stem}|{watermark.stem}|seed={seed}|iwt-schur-v7'


def settings_identity(args,cfg):
    return {'proposal_id':'iwt_schur_4x4_dual_pilot',
            'parameters':asdict(cfg),'protocol':'webp20x3',
            'seed':args.seed,'python':sys.version.split()[0],
            'opencv':cv2.__version__,'numpy':np.__version__,
            'platform':platform.platform(),'baseline_source_modified':False}


def samples(args):
    for h in select_hosts(args.hosts):
        for w in select_watermarks(args.watermarks):yield h,w


def all_cases(args):
    cases=list(attacks.all_cases())
    if args.limit is not None:cases=cases[:args.limit]
    return cases


def make_common(experiment, name, variant,h,w,key,nonce,seed,marked,orig,reference,extracted,seconds,extract_s,detected=None,pilot=None,phase=None,mode=None,blindness='key-assisted blind'):
    met=watermark_metrics(reference,extracted)
    return {'experiment':experiment,'method_id':name,'variant':variant,
            'baseline_blindness':blindness,'host':h.name,'watermark':w.name,
            'key_fingerprint':fingerprint(key),'nonce':nonce,'seed':seed,
            'clean_psnr_db':float(psnr(orig,marked)),
            'clean_ssim':float(ssim(orig,marked)),
            'clean_nc':float(met['nc']),'clean_ncc':float(met['ncc']),
            'clean_ber':float(met['ber']),
            'embed_seconds':seconds,'extract_seconds':extract_s,
            'detected':detected,'pilot_accuracy':pilot,'phase_accuracy':phase,
            'extract_mode':mode}


def run_decode(img,kind,key,nonce,cfg):
    if kind=='full':
        return method.extract(img,key=key,nonce=nonce,cfg=cfg)
    if kind=='no_geometry':
        return method.extract(img,key=key,nonce=nonce,cfg=cfg,enable_geometry=False)
    if kind=='no_photo':
        return method.extract(img,key=key,nonce=nonce,cfg=cfg,enable_photo=False)
    if kind=='no_denoise':
        return method.extract(img,key=key,nonce=nonce,cfg=cfg,enable_denoise=False)
    if kind=='no_structure':
        return method.extract(img,key=key,nonce=nonce,cfg=cfg,enable_structure=False)
    if kind=='one_pilot':
        return onepilot.extract(img,key=key,nonce=nonce,cfg=cfg)
    if kind=='direct':
        return m.extract(img,key=key,nonce=nonce,cfg=cfg)
    raise ValueError(f'Unknown decoder variant: {kind}')


def result_values(z):
    return z.watermark,bool(z.detected),float(z.pilot_accuracy),getattr(z,'phase_accuracy',None),getattr(z,'method',getattr(z,'mode','direct'))


def measure_attack(base,marked,wm,name,level,param,key,nonce,cfg,decode='full',attack_seed=2026):
    a=attacks.attack(marked,name,level,seed=attack_seed)
    start=time.perf_counter()
    z=run_decode(a,decode,key,nonce,cfg)
    seconds=time.perf_counter()-start
    recovered,detected,pilot,phase,mode=result_values(z)
    met=watermark_metrics(wm,recovered)
    return {**base,'category':next(group for group,items in attacks.ATTACKS.items() if name in items),
            'attack':name,'level':level,'parameter':repr(param),
            'status':'ok','attacked_nc':met['nc'],'attacked_ncc':met['ncc'],
            'attacked_ber':met['ber'],'bit_accuracy':met['bit_accuracy'],
            'detected':detected,'pilot_accuracy':pilot,'phase_accuracy':phase,
            'extract_mode':mode,'extract_seconds':seconds,
            'pass_nc_085':bool(met['nc']>.85),
            'accepted_and_pass_nc_085':bool(met['nc']>.85 and detected)}


def eval_one_pair(hpath,wpath,args,cfg,key,variant='full',decode='full',experiment='benchmark'):
    host=load_host(hpath);wm=load_wm(wpath)
    nonce=derive_nonce(hpath,wpath,args.seed)
    start=time.perf_counter()
    marked=method.embed(host,wm,key=key,nonce=nonce,cfg=cfg)
    embed_s=time.perf_counter()-start
    start=time.perf_counter()
    clean=run_decode(marked,decode,key,nonce,cfg)
    clean_s=time.perf_counter()-start
    wclean,detected,pilot,phase,mode=result_values(clean)
    base=make_common(experiment,'iwt_schur_4x4_dual_pilot',variant,hpath,wpath,key,nonce,args.seed,
                     marked,host,wm,wclean,embed_s,clean_s,detected,pilot,phase,mode)
    met=watermark_metrics(wm,wclean)
    rows=[{**base,'category':'Clean','attack':'none','level':'Clean','parameter':'',
           'status':'ok','attacked_nc':met['nc'],'attacked_ncc':met['ncc'],'attacked_ber':met['ber'],
           'bit_accuracy':met['bit_accuracy'],'pass_nc_085':met['nc']>.85,
           'accepted_and_pass_nc_085':bool(met['nc']>.85 and detected)}]
    for cat,name,level,param in all_cases(args):
        try:rows.append(measure_attack(base,marked,wm,name,level,param,key,nonce,cfg,decode,args.seed))
        except Exception as e:
            rows.append({**base,'category':cat,'attack':name,'level':level,'parameter':repr(param),
                         'status':'error','error':f'{type(e).__name__}: {e}',
                         'pass_nc_085':False,'accepted_and_pass_nc_085':False})
            if args.fail_fast:raise
    return rows


def run_benchmark(args,cfg,key):
    allrows=[]
    for h,w in samples(args):
        print('PROPOSAL',h.name,w.name,flush=True)
        allrows.extend(eval_one_pair(h,w,args,cfg,key))
        write_rows(Path(args.out)/'per_attack.csv',allrows)
    return allrows


def run_energy(args):
    rows=[]
    for path in select_hosts(args.hosts):
        host=load_host(path)
        bands=m._groups(host)
        for k,mat in enumerate(bands):
            blocks=m._blocks(mat).astype(np.float64)
            symmetric=.5*(blocks+blocks.transpose(0,2,1))
            vals=np.linalg.eigvalsh(symmetric)
            energy=vals**2
            denom=energy.sum(axis=1)
            nonzero=denom>1e-12
            energy=energy[nonzero];denom=denom[nonzero]
            if not len(energy):continue
            share=energy/denom[:,None]
            dominant=np.argmax(energy,axis=1)
            band=k%4
            record={'host':path.name,'channel':('B','G','R')[k//4],
                    'band':('LL','LH','HL','HH')[band],
                    'blocks':len(energy),'chosen_position':('T44','T11','T11','T44')[band]}
            for j in range(4):
                record[f't{j+1}{j+1}_energy_percent']=float(share[:,j].mean()*100)
                record[f't{j+1}{j+1}_dominance_percent']=float(np.mean(dominant==j)*100)
            rows.append(record)
        print('ENERGY',path.name,flush=True)
    write_rows(Path(args.out)/'energy_by_host_band.csv',rows)
    return rows

DECODER_VARIANTS=('full','direct','one_pilot','no_geometry','no_photo','no_denoise','no_structure')
ENCODER_VARIANTS={
  'all_T44':dict(carrier_indices=(3,3,3,3)),
  'all_T11':dict(carrier_indices=(0,0,0,0)),
  'all_T22':dict(carrier_indices=(1,1,1,1)),
  'LL_only':dict(detail_copies=0),
  'one_copy_per_group':dict(ll_copies=1,detail_copies=1),
  'reduced_refinement':dict(iterations=1),
}

def run_ablation(args,cfg,key):
    rows=[]
    variants=args.variants.split(',') if args.variants else list(DECODER_VARIANTS)+list(ENCODER_VARIANTS)
    for h,w in samples(args):
        host=load_host(h);wm=load_wm(w);nonce=derive_nonce(h,w,args.seed)
        shared=None
        if any(v in DECODER_VARIANTS for v in variants):
            t=time.perf_counter();shared=method.embed(host,wm,key=key,nonce=nonce,cfg=cfg)
            shared_seconds=time.perf_counter()-t
        cases=all_cases(args)
        # Store a single attacked image per condition and reuse it across all
        # decoder-only ablations. Each encoder ablation gets a fresh embed.
        attacked_shared={}
        if shared is not None:
            for _,name,level,param in cases:
                attacked_shared[(name,level)]=attacks.attack(shared,name,level,seed=args.seed)
        for variant in variants:
            if variant in DECODER_VARIANTS:
                marked=shared;used_cfg=cfg;dec=variant;seconds=shared_seconds
            elif variant in ENCODER_VARIANTS:
                used_cfg=replace(cfg,**ENCODER_VARIANTS[variant]);dec='full'
                t=time.perf_counter()
                marked=method.embed(host,wm,key=key,nonce=nonce,cfg=used_cfg)
                seconds=time.perf_counter()-t
            else:raise ValueError(f'Unrecognized ablation {variant}')
            clean=run_decode(marked,dec,key,nonce,used_cfg)
            rec,det,p,phase,mode=result_values(clean)
            base=make_common('ablation','iwt_schur_4x4_dual_pilot',variant,h,w,key,nonce,args.seed,
                             marked,host,wm,rec,seconds,0.,det,p,phase,mode)
            met=watermark_metrics(wm,rec)
            rows.append({**base,'status':'ok','category':'Clean','attack':'none','level':'Clean',
                         'attacked_nc':met['nc'],'attacked_ber':met['ber'],
                         'attacked_ncc':met['ncc'],'bit_accuracy':met['bit_accuracy'],
                         'pass_nc_085':met['nc']>.85,'accepted_and_pass_nc_085':bool(met['nc']>.85 and det)})
            for cat,name,level,param in cases:
                try:
                    if variant in DECODER_VARIANTS:
                        attacked=attacked_shared[(name,level)]
                    else:attacked=attacks.attack(marked,name,level,seed=args.seed)
                    t=time.perf_counter();z=run_decode(attacked,dec,key,nonce,used_cfg)
                    sec=time.perf_counter()-t
                    recovered,det,p,phase,mode=result_values(z);met=watermark_metrics(wm,recovered)
                    rows.append({**base,'category':cat,'attack':name,'level':level,
                                 'parameter':repr(param),'status':'ok',
                                 'attacked_nc':met['nc'],'attacked_ncc':met['ncc'],
                                 'attacked_ber':met['ber'],'bit_accuracy':met['bit_accuracy'],
                                 'detected':det,'pilot_accuracy':p,'phase_accuracy':phase,
                                 'extract_mode':mode,'extract_seconds':sec,
                                 'pass_nc_085':met['nc']>.85,
                                 'accepted_and_pass_nc_085':bool(met['nc']>.85 and det)})
                except Exception as e:
                    rows.append({**base,'status':'error','error':repr(e),'category':cat,
                                 'attack':name,'level':level,'pass_nc_085':False})
                    if args.fail_fast:raise
            write_rows(Path(args.out)/'per_attack.csv',rows)
            print('ABLATION',h.stem,w.stem,variant,flush=True)
    return rows


def run_sweep(args,cfg,key):
    # DEVELOPMENT-ONLY parameter sweep. Never optimize using a held-out test
    # host/watermark. Report every attempted setting and failed extraction.
    rows=[]
    qll=[float(x) for x in args.q_ll_values.split(',')]
    qdt=[float(x) for x in args.q_detail_values.split(',')]
    for h,w in samples(args):
        for a in qll:
            for b in qdt:
                trial=replace(cfg,q_ll=a,q_detail=b)
                try:
                    tmp=eval_one_pair(h,w,args,trial,key,variant=f'q_ll={a};q_detail={b}',experiment='sweep')
                    rows.extend(tmp)
                except Exception as exc:
                    rows.append({'experiment':'sweep','host':h.name,'watermark':w.name,
                                 'variant':f'q_ll={a};q_detail={b}','status':'error','error':repr(exc)})
                    if args.fail_fast:raise
                write_rows(Path(args.out)/'per_attack.csv',rows)
    return rows


def run_payload(args,cfg,key):
    # Sweep copies/strength with one fixed 64x64 payload. Larger payloads require
    # a separate implementation and are not claimed as supported by V7.
    rows=[]
    wanted=[(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    if args.payload_variants:
        wanted=[]
        for part in args.payload_variants.split(','):
            a,b=part.split(':')
            wanted.append((int(a),int(b)))
    for ll,detail in wanted:
            trial=replace(cfg,ll_copies=ll,detail_copies=detail)
            for h,w in samples(args):
                variant=f'LLcopies={ll};detailcopies={detail}'
                try:rows.extend(eval_one_pair(h,w,args,trial,key,variant,experiment='payload'))
                except Exception as exc:
                    rows.append({'experiment':'payload','variant':variant,'host':h.name,'watermark':w.name,
                                 'status':'error','error':repr(exc)})
                    if args.fail_fast:raise
                write_rows(Path(args.out)/'per_attack.csv',rows)
    return rows


def run_security(args,cfg,key):
    rows=[]
    hosts=select_hosts(args.hosts);watermarks=select_watermarks(args.watermarks)
    for h in hosts:
        for w in watermarks:
            host=load_host(h);wm=load_wm(w)
            nonce=derive_nonce(h,w,args.seed)
            marked=method.embed(host,wm,key=key,nonce=nonce,cfg=cfg)
            cases=[('positive',marked,key,nonce),('unmarked',host,key,nonce),
                   ('wrong_nonce',marked,key,nonce+'|wrong')]
            for k in range(args.negative_trials):
                cases.append((f'wrong_key_{k}',marked,f'{key}|wrong-{k}',nonce))
            for other in hosts:
                if other!=h:cases.append((f'unrelated_host_{other.stem}',load_host(other),key,nonce))
            for name,img,k,n in cases:
                try:
                    z=method.extract(img,key=k,nonce=n,cfg=cfg)
                    rows.append({'experiment':'security','host':h.name,'watermark':w.name,
                                 'control':name,'status':'ok','detected':z.detected,
                                 'pilot_accuracy':z.pilot_accuracy,'phase_accuracy':z.phase_accuracy,
                                 # NC is a diagnostic ONLY; not an accept criterion.
                                 'diagnostic_nc':watermark_metrics(wm,z.watermark)['nc']})
                except Exception as exc:
                    rows.append({'experiment':'security','host':h.name,'watermark':w.name,
                                 'control':name,'status':'error','error':repr(exc)})
            write_rows(Path(args.out)/'security_controls.csv',rows)
    return rows


def run_baselines(args):
    selected=ALL_BASELINE_IDS if args.baselines=='all' else [normalize_baseline_id(s.strip()) for s in args.baselines.split(',')]
    rows=[]
    for h,w in samples(args):
        orig_rgb=load_host_rgb(h);wm=load_watermark_binary(w)
        for baseline_id in selected:
            spec=get_baseline_spec(baseline_id)
            t=time.perf_counter()
            try:
                marked,key=embed_baseline(baseline_id,orig_rgb,wm,seed=args.seed)
                embed_seconds=time.perf_counter()-t
                t=time.perf_counter()
                clean=extract_baseline(marked,key,original_host=orig_rgb if spec.requires_original_host else None)
                ext_s=time.perf_counter()-t
                met=watermark_metrics(wm,clean)
                common={'experiment':'baselines','method_id':baseline_id,'variant':'original_unmodified',
                        'baseline_blindness':spec.blindness_tier,'host':h.name,'watermark':w.name,
                        'seed':args.seed,'clean_psnr_db':psnr(orig_rgb,marked),
                        'clean_ssim':ssim(orig_rgb,marked),'clean_nc':met['nc'],
                        'clean_ncc':met['ncc'],'clean_ber':met['ber'],
                        'embed_seconds':embed_seconds,'extract_seconds':ext_s}
                rows.append({**common,'category':'Clean','attack':'none','level':'Clean',
                             'status':'ok','attacked_nc':met['nc'],'attacked_ncc':met['ncc'],
                             'attacked_ber':met['ber'],'bit_accuracy':met['bit_accuracy']})
                for cat,name,level,param in all_cases(args):
                    try:
                        attacked_rgb=attacks.attack(cv2.cvtColor(marked,cv2.COLOR_RGB2BGR),name,level,seed=args.seed)
                        attacked_rgb=cv2.cvtColor(attacked_rgb,cv2.COLOR_BGR2RGB)
                        t=time.perf_counter()
                        recovered=extract_baseline(attacked_rgb,key,original_host=orig_rgb if spec.requires_original_host else None)
                        ex_s=time.perf_counter()-t
                        met=watermark_metrics(wm,recovered)
                        rows.append({**common,'category':cat,'attack':name,'level':level,
                                     'parameter':repr(param),'status':'ok',
                                     'attacked_nc':met['nc'],'attacked_ncc':met['ncc'],
                                     'attacked_ber':met['ber'],'bit_accuracy':met['bit_accuracy'],
                                     'extract_seconds':ex_s,'pass_nc_085':met['nc']>.85})
                    except Exception as exc:
                        rows.append({**common,'category':cat,'attack':name,'level':level,
                                     'status':'error','error':repr(exc)})
                        if args.fail_fast:raise
            except Exception as exc:
                rows.append({'experiment':'baselines','method_id':baseline_id,'host':h.name,
                             'watermark':w.name,'status':'error','error':repr(exc)})
                if args.fail_fast:raise
            write_rows(Path(args.out)/'per_attack.csv',rows)
            print('BASELINE',baseline_id,h.stem,w.stem,flush=True)
    return rows


def run_analyze(args):
    source=Path(args.input)
    if not source.exists():raise FileNotFoundError(source)
    with source.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    groups={}
    for row in rows:
        if row.get('attack') in ('none','') or row.get('status')!='ok':continue
        try:n=float(row['attacked_nc'])
        except (KeyError,ValueError,TypeError):continue
        key=(row.get('method_id',''),row.get('variant',''),row.get('category',''))
        groups.setdefault(key,[]).append(n)
    out=[]
    for (method_id,variant,cat),nums in sorted(groups.items()):
        a=np.array(nums)
        out.append({'method_id':method_id,'variant':variant,'category':cat,
                    'n':len(a),'min_nc':float(np.min(a)),'mean_nc':float(np.mean(a)),
                    'median_nc':float(np.median(a)),'pass_085':int(np.sum(a>.85)),
                    'fraction_pass_085':float(np.mean(a>.85))})
    write_rows(Path(args.out)/'summary.csv',out)
    return out


def run_attacks(args):
    cases=all_cases(args)
    if len(list(attacks.all_cases()))!=60:raise RuntimeError('Not exactly 60 attacks')
    rows=[]
    for cat,name,level,param in cases:
        rows.append({'category':cat,'attack':name,'level':level,'parameter':repr(param),
                     'status':'defined'})
    # The manifest lists definitions, NOT experimental results.
    write_rows(Path(args.out)/'attack_definitions.csv',rows)
    return rows


def run_all(args,cfg,key):
    # This orchestrator is intentionally separate from the original MSDP
    # run_all_experiments.py. No baseline implementation is changed.
    out=Path(args.out)
    stages=['attacks','energy','benchmark','ablation','sweep','payload','security']
    if args.include_baselines:stages.append('baselines')
    status=[]
    for stage in stages:
        print('RUN ALL:',stage,flush=True)
        sub=argparse.Namespace(**vars(args));sub.out=str(out/stage)
        fn={'attacks':lambda:run_attacks(sub),'energy':lambda:run_energy(sub),
            'benchmark':lambda:run_benchmark(sub,cfg,key),
            'ablation':lambda:run_ablation(sub,cfg,key),
            'sweep':lambda:run_sweep(sub,cfg,key),
            'payload':lambda:run_payload(sub,cfg,key),
            'security':lambda:run_security(sub,cfg,key),
            'baselines':lambda:run_baselines(sub)}[stage]
        try:
            rows=fn()
            status.append({'stage':stage,'status':'ok','rows':len(rows)})
        except Exception as exc:
            status.append({'stage':stage,'status':'error','error':repr(exc)})
            write_rows(out/'execution_status.csv',status)
            if args.fail_fast:raise
        write_rows(out/'execution_status.csv',status)
    return status


def make_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('experiment',choices=('attacks','energy','benchmark','ablation','sweep','payload','security','baselines','analyze','all'))
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--hosts',default='pepper')
    p.add_argument('--watermarks',default='wm')
    p.add_argument('--out',default=None)
    p.add_argument('--key',default=None)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--limit',type=int,default=None,help='Number of attack conditions; omit for all 60')
    p.add_argument('--variants',default=None,help='Comma-separated ablation variants')
    p.add_argument('--baselines',default='all')
    p.add_argument('--q-ll-values',default='24,28,32')
    p.add_argument('--q-detail-values',default='12,14,16')
    p.add_argument('--negative-trials',type=int,default=8)
    p.add_argument('--payload-variants',default=None,
                   help='Optional subset of LLcopies:detailcopies, e.g. 2:2,1:1')
    p.add_argument('--input',default=None,help='Existing per_attack.csv for analyze only')
    p.add_argument('--include-baselines',action='store_true',help='Also run unchanged original literature baselines in all command')
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--strict',action='store_true',help='Nonzero exit if clean PSNR/NC or post-attack threshold fails')
    p.add_argument('--psnr-min',type=float,default=45.0)
    p.add_argument('--nc-min',type=float,default=0.85)
    return p


def main(argv=None):
    p=make_parser();args=p.parse_args(argv)
    if args.limit is not None and not (0<=args.limit<=60):p.error('--limit must be 0..60')
    if args.experiment=='analyze' and not args.input:p.error('--input is required for analyze')
    if args.out is None:args.out=str(ROOT/'outputs'/'proposed'/args.experiment)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg,key=load_setup(args)
    (out/'run_settings.json').write_text(json.dumps({**settings_identity(args,cfg),
       'hosts':args.hosts,'watermarks':args.watermarks,'key_fingerprint':fingerprint(key),
       'attack_limit':args.limit},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    methods={'attacks':lambda:run_attacks(args),'energy':lambda:run_energy(args),
             'benchmark':lambda:run_benchmark(args,cfg,key),
             'ablation':lambda:run_ablation(args,cfg,key),
             'sweep':lambda:run_sweep(args,cfg,key),
             'payload':lambda:run_payload(args,cfg,key),
             'security':lambda:run_security(args,cfg,key),
             'baselines':lambda:run_baselines(args),
             'analyze':lambda:run_analyze(args),
             'all':lambda:run_all(args,cfg,key)}
    start=time.perf_counter();rows=methods[args.experiment]()
    print(f'{args.experiment}: {len(rows)} rows; {time.perf_counter()-start:.1f}s; output={out}',flush=True)
    errors=sum(x.get('status')=='error' for x in rows)
    if errors:
        print(f'WARNING: {errors} rows failed and were retained, not silently dropped.',flush=True)
        if args.fail_fast:raise SystemExit(2)
    if args.strict and args.experiment in ('benchmark','ablation','sweep','payload'):
        errors_out=[]
        for row in rows:
            if row.get('status')!='ok':
                errors_out.append((row.get('host'),row.get('attack'),'error'))
                continue
            try:
                if float(row['clean_psnr_db'])<=args.psnr_min:
                    errors_out.append((row.get('host'),row.get('attack'),'PSNR'))
                if float(row['clean_ber'])!=0.0:
                    errors_out.append((row.get('host'),row.get('attack'),'clean bit errors'))
                if row.get('attack')!='none' and float(row['attacked_nc'])<=args.nc_min:
                    errors_out.append((row.get('host'),row.get('attack'),'NC'))
                if row.get('attack')!='none' and row.get('detected') is False:
                    errors_out.append((row.get('host'),row.get('attack'),'not detected'))
            except (ValueError,TypeError,KeyError):
                errors_out.append((row.get('host'),row.get('attack'),'missing metric'))
        print(f'STRICT: {len(errors_out)} violations of PSNR>{args.psnr_min}, clean BER=0, NC>{args.nc_min}, detection.',flush=True)
        if errors_out:
            print('First violations:',errors_out[:12],flush=True)
            raise SystemExit(3)

if __name__=='__main__':main()
