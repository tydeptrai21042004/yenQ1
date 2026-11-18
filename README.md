# Blind IWT–Schur 4×4 watermarking — source code only

This project contains **one proposed method**: blind, keyed IWT–Schur watermarking for a **512×512 color host** and a **64×64 binary watermark**. It is adapted from the tested V7 implementation, integrated into the uploaded Quynh repository. The former MSDP proposal and drivers are **not included**. All **16 literature-baseline implementations** are preserved byte-for-byte from the uploaded archive.

**No experimental results, figures, generated images, or DOCX files are included.** Scripts create outputs only when you run them. Do not infer experimental success from the presence of scripts.

## Proposed algorithm

- Apply integer Haar IWT separately to B, G, R channels, using LL, LH, HL, HH.
- Split all subbands into **4×4** blocks, perform eigen/real-Schur decomposition on the block's symmetric part.
- Embed one bit into exactly **one fixed Schur diagonal position per block**: **LL/HH → T44; LH/HL → T11**. Fixed positions correspond to the original algebraic eigenvalue order; no adaptive per-block coordinate switching.
- Key-dependent dispersed repetitions; two disjoint pilots (geometry and QIM-phase), iterative bit-consistency repair.
- Blind extraction with automatic geometric alignment, photometric correction, pilot-selected denoising, structural recovery. No host/watermark template is used by the decoder.
- Direct 64×64 binary watermark; **no DWT/IWT of the watermark**.

This is an **experimental research implementation**, not a certified robustness/security guarantee. A high NC score is not a valid detection when `detected=False`. Always report clean/attacked BER, NC, NCC, PSNR, and pilot detection separately.

## Ubuntu / WSL installation

```bash
cd Quynh_watermarking-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q
python scripts/validate_project.py
```

## Exact reproducibility of the previously reported V7 Pepper + wm.png results

The generic paper-protocol script intentionally uses a **different key and a per-pair
nonce**, and therefore does **not** recreate the frozen V7 demonstration image.
This difference affects PSNR, NC and the pilot detector. To reproduce the **same**
V7 experiment with the integrated proposal, use:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce_v7_reference.py \
  --out outputs/v7_exact_reference
```

To verify all 60 rows against the previously published V7 WebP CSV, provide its
file path; the source-only archive deliberately does not bundle results:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce_v7_reference.py \
  --out outputs/v7_exact_reference \
  --reference /path/to/FINAL_IWT_Schur_4x4_V7_WebP_60_Attacks.csv
```

The comparison checks NC, BER, pilot scores, `detected`, decoder mode and clean
PSNR. It refuses to claim a match if any row differs beyond numeric formatting
tolerance. Exact-reference values: key `FIXED_T44_DEMO_KEY`, nonce
`pepper_wm_fixed_t44`, attack seed `2026`, iterations `8`, LL/detail/pilot
QIM steps `28/14/32`. This reproducibility key is PUBLIC and should not be used
as a secret operational key. All baseline source files stay byte-identical.

## Run experiments (outputs created at runtime, not in this ZIP)

```bash
# Single pair, complete WebP-based 20×3 protocol = 60 attacks
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py benchmark \
  --hosts pepper --watermarks wm --out outputs/onepair --strict

# Raw energy survey of all 4x4 Schur diagonal elements on 12 original hosts
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py energy \
  --hosts all --out outputs/energy

# Decoder-only + embedder ablations (see docs/EXPERIMENT_PLAN.md)
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py ablation \
  --hosts pepper --watermarks wm --out outputs/ablation

# Quantization / repetition sweeps on DEVELOPMENT images only
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py sweep \
  --hosts Girl,house,pepper,sailboat --watermarks wm --out outputs/tuning
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py payload \
  --hosts Girl --watermarks wm --out outputs/payload

# False-positive controls, bad keys and unmarked hosts
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py security \
  --hosts pepper,airplane --watermarks all --out outputs/security

# All 16 ORIGINAL baseline implementations under exactly the same 60 attacks
OPENBLAS_NUM_THREADS=1 python scripts/proposed_experiments.py baselines \
  --baselines all --hosts pepper --watermarks wm --out outputs/baselines

# Summary over any CSV, including worst case
python scripts/proposed_experiments.py analyze \
  --input outputs/onepair/per_attack.csv --out outputs/onepair/summary

# Full train/test experiment protocol on held-out hosts and both watermarks
OPENBLAS_NUM_THREADS=1 python scripts/run_paper_protocol.py \
  --profile full --include-baselines --all-baselines --fail-fast
```

Use `--limit 3` for quick smoke tests; **omit** the limit for the complete 60-condition suite. The full protocol, especially the 16 baselines × all hosts × 60 attacks and geometric synchronization search, can be computationally expensive. Stage progress and failures are recorded; rerunning the paper runner skips successful stages.

The default study key in `configs/iwt_schur_v7.json` is not a secure secret. Replace it before deployment. `nonce` is derived from the host, watermark and seed for reproducible laboratory experiments. Metadata with original watermark bits is never stored for extraction.

## Package directories

| Path | Role |
| --- | --- |
| `src/qr64_certified/proposed/iwt_schur_v7/` | The **only proposal**, including embed/extract, pilot alignment, denoising, IWT, attacks |
| `src/qr64_certified/baselines/` | **All 16 original baselines**, source bytes unchanged |
| `scripts/proposed_experiments.py` | Energy, 60 attacks, ablations, sweep, payload, controls, baseline comparison and analysis |
| `scripts/run_paper_protocol.py` | Development/held-out workflow |
| `scripts/proposed_paired_statistics.py` | Paired host-level statistical comparison |
| `scripts/validate_project.py` | Baseline checksums and input/codecs smoke checks |
| `configs/baseline_integrity_sha256.json` | SHA-256 proof for every original baseline source file |
| `data/` | Original 12 colored hosts and 2 watermarks |

The original source archive was the Quynh repository; all non-baseline MSDP proposal code was omitted so that the present research package has one active proposal. Baseline algorithm definitions, implementations, metadata and paper evidence were not changed.
