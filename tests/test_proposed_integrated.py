"""Regression checks of the single proposal and the unchanged literature baselines."""
from __future__ import annotations
from dataclasses import replace
import hashlib,json
from pathlib import Path
import cv2,numpy as np,pytest
from qr64_certified.baselines import ALL_BASELINE_IDS,embed_baseline,extract_baseline
from qr64_certified.common.io import load_host_rgb,load_watermark_binary
from qr64_certified.proposed.iwt_schur_v7 import fixed_t44 as m,robust_v7 as v7,phase_pilot as phase,attacks
ROOT=Path(__file__).resolve().parents[1]
CFG=m.Config(q_ll=28.,q_detail=14.,q_pilot=32.,iterations=8)
KEY='FIXED_T44_DEMO_KEY';NONCE='pepper_wm_fixed_t44'

@pytest.fixture(scope='module')
def pair():
    image=cv2.imread(str(ROOT/'data/host/pepper.bmp'))
    wm=m.read_watermark(ROOT/'data/watermark/wm.png')
    marked=v7.embed(image,wm,key=KEY,nonce=NONCE,cfg=CFG)
    return image,wm,marked

def test_baseline_implementations_unchanged():
    assert len(ALL_BASELINE_IDS)==16
    hashes=json.loads((ROOT/'configs/baseline_integrity_sha256.json').read_text())
    assert len(hashes)>=50
    for rel,digest in hashes.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest

def test_four_band_fixed_carrier_position(pair):
    image,_,_=pair
    assert m.BLOCK==4 and CFG.carrier_indices==(3,0,0,3)
    for band in range(4):
        A=m._blocks(m._groups(image)[band])[:30]
        e=np.linalg.eigvalsh(.5*(A+A.transpose(0,2,1)))
        chosen,_=m._schur_carrier(A,band,CFG.carrier_indices)
        assert np.allclose(chosen,e[:,CFG.carrier_indices[band]])

def test_pilot_sites_disjoint_and_unique():
    ids,ass,pids=m.slots(KEY,NONCE,CFG)
    extra=phase.phase_slots(KEY,NONCE,CFG)
    assert len(np.unique(ids))==len(ids)
    assert not (set(ids)&set(pids) or set(ids)&set(extra) or set(extra)&set(pids))
    assert len(ids)==4*4096

def test_exact_clean_detection_and_psnr(pair):
    image,wm,marked=pair
    z=v7.extract(marked,key=KEY,nonce=NONCE,cfg=CFG)
    assert m.psnr(image,marked)>45.
    assert np.array_equal(z.watermark,wm)
    assert z.detected
    assert z.pilot_accuracy==1 and z.phase_accuracy==1

def test_false_positive_controls(pair):
    image,wm,marked=pair
    assert not v7.extract(image,key=KEY,nonce=NONCE,cfg=CFG).detected
    assert not v7.extract(marked,key=KEY+'WRONG',nonce=NONCE,cfg=CFG).detected

def test_two_original_watermarks_binary():
    for f in (ROOT/'data/watermark').glob('*.png'):
        wm=m.read_watermark(f)
        assert wm.shape==(64,64) and wm.dtype==np.uint8
        assert set(np.unique(wm)).issubset({0,255})

def test_attack_suite_matches_modified_protocol():
    cases=list(attacks.all_cases())
    assert len(cases)==60
    names={name for _,name,_,_ in cases}
    assert 'WebP lossy quality' in names
    assert not any('Color quantization' in n for n in names)
    assert len({(name,level) for _,name,level,_ in cases})==60

def test_encoder_ablation_changes_actual_coefficient(pair):
    image,wm,_=pair
    bl=m._blocks(m._groups(image)[1])[:6]
    default,_=m._schur_carrier(bl,1,CFG.carrier_indices)
    alt,_=m._schur_carrier(bl,1,(3,3,3,3))
    assert not np.allclose(default,alt)
    trial=replace(CFG,carrier_indices=(3,3,3,3))
    out=v7.embed(image,wm,key=KEY,nonce=NONCE,cfg=trial)
    assert out.shape==image.shape

def test_blind_extract_requires_only_received_key_nonce(pair):
    _,_,marked=pair
    recovered=v7.extract(marked,key=KEY,nonce=NONCE,cfg=CFG,
                         enable_geometry=False,enable_photo=False)
    assert recovered.watermark.shape==(64,64)

def test_baseline_api_unmodified():
    host=load_host_rgb(ROOT/'data/host/pepper.bmp')
    wm=load_watermark_binary(ROOT/'data/watermark/wm.png')
    watermarked,key=embed_baseline('dct_dm_qim_chen2001_blind',host,wm,seed=2026)
    ext=extract_baseline(watermarked,key)
    assert ext.shape==wm.shape

@pytest.mark.parametrize('attack,level',[
 ('JPEG quality','Mild'),('Gaussian variance [0,1]','Strong'),('WebP lossy quality','Moderate')
])
def test_attack_decoder_produces_valid_shape(pair,attack,level):
    _,wm,marked=pair
    received=attacks.attack(marked,attack,level,seed=2026)
    result=v7.extract(received,key=KEY,nonce=NONCE,cfg=CFG)
    assert result.watermark.shape==wm.shape
    assert np.isfinite(m.nc(wm,result.watermark))
