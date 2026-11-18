#!/usr/bin/env python3
"""Reproduce the ORIGINAL frozen V7 60-attack Pepper/wm WebP protocol.

This deliberately uses the exact V7 key and nonce. It invokes the integrated
proposal implementation, not an additional proposed method. Reference results
are NOT bundled: optionally pass --reference <original V7 60-attack CSV> to
compare NC, BER, pilot, phase, detection, extraction mode and clean PSNR.
"""
from __future__ import annotations
import argparse,csv,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def compare_csv(generated:Path,reference:Path,output:Path):
    with generated.open(newline='',encoding='utf-8') as f:
        actual=list(csv.DictReader(f))
    with reference.open(newline='',encoding='utf-8') as f:
        original=list(csv.DictReader(f))
    by_case={(r['attack'],r['level']):r for r in actual if r['attack']!='none' and r.get('status')=='ok'}
    if len(by_case)!=60 or len(original)!=60:
        raise ValueError(f'Expected 60 attacked cases, got {len(by_case)} and {len(original)}')
    rows=[]
    for ref in original:
        k=(ref['attack'],ref['severity'])
        if k not in by_case:raise ValueError(f'Missing case: {k}')
        a=by_case[k]
        numeric={
            'nc':('attacked_nc','nc_after',5e-8),
            'ber':('attacked_ber','ber_after',5e-8),
            'pilot':('pilot_accuracy','pilot_accuracy',6e-5),
            'phase':('phase_accuracy','phase_accuracy',6e-5),
            'clean_psnr':('clean_psnr_db','psnr_clean_db',5e-7),
        }
        errors={name:float(a[x])-float(ref[y]) for name,(x,y,_) in numeric.items()}
        all_numeric=all(abs(errors[name])<=tol for name,(_,_,tol) in numeric.items())
        detected_match=int(bool(a['detected']=='True')==bool(int(ref['detected'])))
        mode_match=int(a['extract_mode']==ref['extract_mode'])
        matched=bool(all_numeric and detected_match and mode_match)
        rows.append({'case':ref['case'],'attack':ref['attack'],'severity':ref['severity'],
                     'nc_original':ref['nc_after'],'nc_reproduced':a['attacked_nc'],
                     'nc_delta':errors['nc'],'ber_delta':errors['ber'],
                     'pilot_delta':errors['pilot'],'phase_delta':errors['phase'],
                     'psnr_delta':errors['clean_psnr'],
                     'detected_match':detected_match,'extract_mode_match':mode_match,
                     'match':int(matched)})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    good=sum(r['match'] for r in rows)
    print(f'V7 REFERENCE MATCH: {good}/60 cases; audit={output}',flush=True)
    if good!=60:raise SystemExit(4)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/v7_exact_reference')
    ap.add_argument('--reference',type=Path,help='Original V7 60-attack CSV for exact audit')
    ap.add_argument('--limit',type=int,default=None,help='Quick smoke test; omit for full 60')
    a=ap.parse_args()
    cmd=[sys.executable,str(ROOT/'scripts/proposed_experiments.py'),'benchmark',
        '--hosts','pepper','--watermarks','wm','--key','FIXED_T44_DEMO_KEY',
        '--nonce','pepper_wm_fixed_t44','--seed','2026',
        '--out',str(a.out),'--strict','--fail-fast']
    if a.limit is not None:cmd+=['--limit',str(a.limit)]
    print('Running original V7 settings with integrated proposal',flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    if a.reference:
        if a.limit is not None:
            raise ValueError('Do not use --limit with full reference comparison')
        compare_csv(a.out/'per_attack.csv',a.reference,a.out/'reference_audit.csv')
if __name__=='__main__':main()
