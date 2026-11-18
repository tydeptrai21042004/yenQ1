#!/usr/bin/env python3
"""Source-only project validation; does not write experimental results."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
from qr64_certified.baselines import ALL_BASELINE_IDS
from qr64_certified.proposed.iwt_schur_v7 import fixed_t44 as m,attacks

def main():
    checks={}
    hashes=json.loads((ROOT/'configs/baseline_integrity_sha256.json').read_text())
    bad=[path for path,digest in hashes.items()
         if not (ROOT/path).is_file() or hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]
    checks['baseline_files_byte_identical']=not bad
    checks['baseline_count_16']=len(ALL_BASELINE_IDS)==16
    checks['one_active_proposal']=not (ROOT/'src/qr64_certified/msdp.py').exists()
    checks['carrier4x4_LL_T44_LH_T11_HL_T11_HH_T44']=m.BLOCK==4 and m.Config().carrier_indices==(3,0,0,3)
    cases=list(attacks.all_cases())
    checks['attack_count_60']=len(cases)==60
    names={name for _,name,_,_ in cases}
    checks['webp_in_color_quantization_out']='WebP lossy quality' in names and not any('Color quantization' in s for s in names)
    checks['all_original_host_files']=len(list((ROOT/'data/host').glob('*.bmp')))==12
    checks['two_original_watermark_files']=len(list((ROOT/'data/watermark').glob('*.png')))==2
    for f in sorted((ROOT/'data/watermark').glob('*.png')):
        wm=m.read_watermark(f)
        checks['binary_64_'+f.name]=wm.shape==(64,64) and set(wm.flat).issubset({0,255})
    host=cv2.imread(str(ROOT/'data/host/pepper.bmp'))
    for name in ('WebP lossy quality','JPEG2000 quality-layer parameter'):
        try:
            a=attacks.attack(host,name,'Mild')
            checks[name+'_codec']=a.shape==host.shape and a.dtype==host.dtype
        except Exception as exc:
            checks[name+'_codec']=False
            print('CODEC FAILED',name,repr(exc))
    for name,ok in checks.items():print(('PASS' if ok else 'FAIL'),name)
    if not all(checks.values()):raise SystemExit(1)
    print('PASS: proposal source, 16 baseline implementations, original data and 60-attack definitions verified.')
if __name__=='__main__':main()
