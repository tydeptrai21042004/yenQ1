#!/usr/bin/env python3
"""Paper-style reproducible protocol with disjoint parameter tuning/test hosts.

No outputs ship with this repository. 'full' runs all 60 conditions on holdout
hosts and both original 64x64 watermarks. The original baseline engines stay
unchanged. Only the new IWT-Schur pipeline is treated as the proposal.
"""
from __future__ import annotations
import argparse,sys,subprocess,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['quick','full'],default='quick')
    p.add_argument('--out',default='outputs/paper_protocol')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--include-baselines',action='store_true')
    p.add_argument('--all-baselines',action='store_true')
    p.add_argument('--fail-fast',action='store_true')
    a=p.parse_args()
    output=(ROOT/a.out).resolve();output.mkdir(parents=True,exist_ok=True)
    # Explicit disjoint host split. Development data never overlap with evaluation hosts.
    develop='Girl,house,pepper,sailboat'
    heldout='airplane,athens,baboon,butfly,manhatan,milkdrop,safari,tiffany'
    if a.profile=='quick':develop='Girl';heldout='airplane'
    limit=['--limit','3'] if a.profile=='quick' else []
    cmds=[]
    def task(name,script,cmd):
        cmds.append((name,[sys.executable,str(ROOT/'scripts'/script),*cmd]))
    def exp(name,typ,hosts,marks='wm',extra=()):
        task(name,'proposed_experiments.py',[
            typ,'--hosts',hosts,'--watermarks',marks,'--seed',str(a.seed),
            '--out',str(output/name),*limit,*extra])
    exp('attack_definitions','attacks',develop)
    exp('energy_survey','energy','all')
    exp('development_sweep','sweep',develop)
    if a.profile=='full':
        exp('development_payload','payload',develop)
    exp('heldout_benchmark','benchmark',heldout,'all')
    exp('heldout_ablation','ablation',heldout,'all',
        extra=['--limit','0','--variants','full,no_geometry,no_photo,no_denoise,one_pilot'] if a.profile=='quick' else [])
    if a.profile=='full':
        exp('heldout_security','security',heldout,'all',extra=['--negative-trials','24'])
    if a.include_baselines:
        selected='all' if a.all_baselines or a.profile=='full' else 'dct_dm_qim_chen2001_blind'
        exp('heldout_baselines','baselines',heldout,'all',extra=['--baselines',selected])
        task('paired_statistics','proposed_paired_statistics.py',[
            '--proposal',str(output/'heldout_benchmark/per_attack.csv'),
            '--baselines',str(output/'heldout_baselines/per_attack.csv'),
            '--out',str(output/'paired_statistics')])
    task('baseline_audit','audit_paper_baselines.py',['--output',str(output/'baseline_audit')])
    state_path=output/'execution_status.json'
    old=json.loads(state_path.read_text()) if state_path.exists() else {}
    spec={'profile':a.profile,'seed':a.seed,'development_hosts':develop,
          'heldout_hosts':heldout,'test_watermarks':'all','condition_count':len(limit) and 3 or 60,
          'commands':{name:command for name,command in cmds}}
    (output/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    for index,(name,command) in enumerate(cmds,1):
        if old.get(name,{}).get('exit_code')==0:
            print(f'{index}/{len(cmds)} {name} already complete',flush=True);continue
        start=time.perf_counter();print(f'{index}/{len(cmds)} RUN {name}',flush=True)
        log=output/(name+'.log')
        with log.open('a',encoding='utf-8') as fh:
            fh.write('\n=== EXECUTION ===\n')
            proc=subprocess.run(command,cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT)
        old[name]={'exit_code':proc.returncode,'seconds':time.perf_counter()-start,'log':str(log)}
        state_path.write_text(json.dumps(old,indent=2)+'\n')
        print(f'  exit {proc.returncode}',flush=True)
        if proc.returncode!=0:
            print(f'Check {log}. Incomplete results are not declared successful.',flush=True)
            if a.fail_fast:raise SystemExit(proc.returncode)
            break
    if all(old.get(n,{}).get('exit_code')==0 for n,_ in cmds):
        print('ALL PROTOCOL STAGES COMPLETED',flush=True)
    else:print('PROTOCOL INCOMPLETE: check execution_status.json',flush=True)
if __name__=='__main__':main()
