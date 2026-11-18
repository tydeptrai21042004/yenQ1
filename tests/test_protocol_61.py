"""Fast invariants: shared manifest, unique IDs, same inputs, auditable capacity."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

from qr64_certified.baselines.registry import ALL_BASELINE_IDS, get_baseline_spec
from qr64_certified.proposed.iwt_schur_v7 import attacks

ROOT=Path(__file__).resolve().parents[1]


def test_canonical_61_includes_clean_and_exactly_60_attack_definitions():
    conditions=attacks.canonical_conditions()
    assert len(conditions)==61
    assert conditions[0]['condition_id']=='C000'
    assert conditions[0]['attack']=='none'
    assert [row['condition_id'] for row in conditions[1:]]==[f'A{i:03d}' for i in range(1,61)]
    assert len({(row['attack'],row['level']) for row in conditions})==61
    assert Counter(row['category'] for row in conditions[1:])=={
        'Compression':12,'Noise':12,'Filtering':12,'Geometry':12,'Photometric':12}
    assert len(attacks.protocol_sha256())==64
    for row in conditions:
        assert attacks.condition_for(row['attack'],row['level'])==row


def test_registered_baselines_and_capacity_are_explicit():
    assert len(ALL_BASELINE_IDS)==16
    ineligible=[id for id in ALL_BASELINE_IDS
                if not get_baseline_spec(id).common_4096_payload]
    assert ineligible==['dct_dew_langelaar2001_blind']


def test_full_orchestration_has_no_attack_limit(tmp_path):
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_paper_protocol.py'),
        '--profile','full','--dry-run','--out',str(tmp_path/'paper')],
        cwd=ROOT,text=True,capture_output=True,check=True)
    plan=json.loads(proc.stdout)
    assert plan['attack_count']==60 and plan['clean_count']==1
    assert plan['test_watermarks']=='all'
    assert len(plan['heldout_hosts'].split(','))==8
    for name in ('development_sweep','development_payload','heldout_benchmark',
                 'heldout_ablation','heldout_baselines'):
        assert '--limit' not in plan['commands'][name]
    assert '--include-baselines' in plan['commands']['protocol_reports']
