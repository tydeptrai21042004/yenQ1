#!/usr/bin/env python3
"""Single entry point from a fresh GitHub clone: run, audit and report.

full: every comparable method/variant/host/watermark receives 1 clean + 60
canonical attacks. quick: 1 clean + first 3 attacks; NEVER a full evaluation.
Algorithms and baseline source modules are not modified by this runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def fingerprint_inputs():
    """Invalidate resume status when tracked algorithm, configs, or inputs change."""
    h=hashlib.sha256()
    for folder in ('scripts','src','configs','data'):
        for f in sorted((ROOT/folder).rglob('*')):
            if not f.is_file() or f.suffix=='.pyc' or '__pycache__' in f.parts:continue
            h.update(f.relative_to(ROOT).as_posix().encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--profile',choices=['quick','full'],default='quick')
    p.add_argument('--out',default='outputs/paper_protocol')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--include-baselines',dest='include_baselines',action='store_true',default=True)
    p.add_argument('--no-baselines',dest='include_baselines',action='store_false')
    p.add_argument('--all-baselines',action='store_true',help='Compatibility alias; all 16 are checked by default')
    p.add_argument('--fail-fast',action='store_true',help='Stop immediately if a child stage fails')
    p.add_argument('--force',action='store_true',help='Rerun successful stages, even if input fingerprint matches')
    p.add_argument('--dry-run',action='store_true',help='Print stage commands without running the experiments')
    a=p.parse_args(argv)
    output=(ROOT/a.out).resolve();output.mkdir(parents=True,exist_ok=True)
    full=a.profile=='full'
    develop='Girl,house,pepper,sailboat' if full else 'Girl'
    heldout='airplane,athens,baboon,butfly,manhatan,milkdrop,safari,tiffany' if full else 'airplane'
    limit=[] if full else ['--limit','3']
    heldout_marks='all' if full else 'wm'
    commands=[]
    def task(name,script,options):
        commands.append((name,[sys.executable,str(ROOT/'scripts'/script),*options]))
    def exp(name,kind,hosts,marks='wm',extra=()):
        task(name,'proposed_experiments.py',[
            kind,'--hosts',hosts,'--watermarks',marks,'--seed',str(a.seed),
            '--out',str(output/name),*limit,*extra,'--fail-fast'])
    task('validation','validate_project.py',[])
    exp('attack_definitions','attacks',develop)
    exp('energy_survey','energy','all')
    exp('development_sweep','sweep',develop,
        extra=['--q-ll-values','24,28,32','--q-detail-values','12,14,16'])
    if full:
        exp('development_payload','payload',develop)
    exp('heldout_benchmark','benchmark',heldout,heldout_marks)
    exp('heldout_ablation','ablation',heldout,heldout_marks,extra=[] if full else
        ['--variants','full,direct'])
    if full:
        exp('heldout_security','security',heldout,'all',extra=['--negative-trials','24'])
    if a.include_baselines:
        exp('heldout_baselines','baselines',heldout,heldout_marks,extra=['--baselines','all'])
        task('paired_statistics','proposed_paired_statistics.py',[
            '--proposal',str(output/'heldout_benchmark/per_attack.csv'),
            '--baselines',str(output/'heldout_baselines/per_attack.csv'),
            '--out',str(output/'paired_statistics')])
    task('baseline_audit','audit_paper_baselines.py',['--output',str(output/'baseline_audit')])
    task('protocol_reports','protocol_reports.py',[
        '--root',str(output),'--profile',a.profile,
        *(['--include-baselines'] if a.include_baselines else [])])

    code_digest=fingerprint_inputs()
    spec={'profile':a.profile,'seed':a.seed,'development_hosts':develop,
          'heldout_hosts':heldout,'development_watermark':'wm',
          'test_watermarks':heldout_marks,'attack_count':60 if full else 3,
          'clean_count':1,'include_baselines':a.include_baselines,
          'source_input_sha256':code_digest,'commands':{name:cmd for name,cmd in commands}}
    if a.dry_run:
        print(json.dumps(spec,indent=2),flush=True)
        return 0
    (output/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    frozen=subprocess.run([sys.executable,'-m','pip','freeze'],
                          text=True,capture_output=True)
    (output/'environment_pip_freeze.txt').write_text(
        frozen.stdout if frozen.returncode==0 else
        'pip freeze unavailable: '+frozen.stderr, encoding='utf-8')
    state_path=output/'execution_status.json'
    old=json.loads(state_path.read_text()) if state_path.exists() else {}
    env=os.environ.copy()
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    env['MKL_NUM_THREADS']='1'
    for index,(name,command) in enumerate(commands,1):
        signature=hashlib.sha256(json.dumps([command,code_digest],sort_keys=True).encode()).hexdigest()
        done=old.get(name,{})
        if not a.force and done.get('exit_code')==0 and done.get('signature')==signature:
            print(f'{index}/{len(commands)} SKIP verified prior stage {name}',flush=True)
            continue
        print(f'{index}/{len(commands)} RUN {name}',flush=True)
        log=output/(name+'.log')
        start=time.perf_counter()
        with log.open('w',encoding='utf-8') as handle:
            handle.write('COMMAND: '+ ' '.join(command)+'\n')
            handle.flush()
            process=subprocess.run(command,cwd=ROOT,env=env,
                                   stdout=handle,stderr=subprocess.STDOUT)
        old[name]={'exit_code':process.returncode,'signature':signature,
                   'seconds':round(time.perf_counter()-start,3),
                   'log':str(log.relative_to(output))}
        state_path.write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
        print(f'  exit={process.returncode}; log={log.relative_to(output)}',flush=True)
        if process.returncode!=0:
            print('INCOMPLETE: an experiment failed. See the stage log; do not claim all 61 conditions passed.',flush=True)
            return process.returncode if process.returncode>0 else 1
    print('COMPLETE: all stages succeeded and experiment coverage was audited.',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
