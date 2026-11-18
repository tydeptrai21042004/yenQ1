# Experimental plan (scripts only; no precomputed results)

## Protocol

- 12 original 512×512 color images, two original 64×64 binary watermark files.
- 20 distinct attack types × three strengths, **WebP compression 90/70/50 in place of color quantization**. This is a modified protocol and must never be presented as identical to a prior color-quantization suite.
- NC is positive-pixel cosine NC. Also output mean-subtracted NCC and raw BER; report minimum per setting, failure rows and pilot `detected` independently.
- For fair baseline comparison, same host, watermark, attack implementation, attack seed and applied attack parameter. Note baseline extraction classes (blind / key-assisted / semi-blind / nonblind) separately.

## Data split

- Development hosts: Girl, house, pepper, sailboat. Development watermark: wm.
- Held-out test hosts: airplane, athens, baboon, butfly, manhatan, milkdrop, safari, tiffany. Watermarks: both original files.
- Choose parameters on development only, **freeze settings**, then run held-out validation. Don't tune to the reported 60 attacked test rows.

## Required experiments

| Experiment | Command | Key outputs created when run |
| --- | --- | --- |
| Attack-definition check | `proposed_experiments.py attacks` | 60 input attack definitions |
| Four-band Schur energy | `... energy --hosts all` | coefficient energy fractions and dominant-position frequencies by band/channel/image |
| Clean quality, PSNR/SSIM/NC/NCC/BER | `... benchmark` | clean row per pair |
| Full robustness 20×3 | `... benchmark` | 60 attack rows + clean per pair; report minima |
| Pilot detection & false positives | `... security` | unmarked, wrong key/nonce, substituted host |
| Encoder coefficient ablation | `... ablation` | band-guided vs T44/T11/T22, LL-only, repetition variants |
| Decoder component ablation | `... ablation` | geometric, photometric, denoise, structural, dual pilot, direct decoder |
| Strength sweep | `... sweep` | LL/detail QIM q grid, development only |
| Capacity/repetition sweep | `... payload` | number of copies LL/detail; fixed payload 4096 bits |
| Published baselines | `... baselines --baselines all` | same images/attacks under original baseline engines |
| Statistical analysis | `proposed_paired_statistics.py` | paired host-clustered bootstrap and approximate sign-flip inference |
| Implementation regression | `pytest`, `validate_project.py` | execution pass/fail only |

## Ablation controls

**Decoder-only ablations** use precisely the same marked image per attack: `full`, `direct`, `one_pilot`, `no_geometry`, `no_photo`, `no_denoise`, `no_structure`. Only the extraction side changes.

**Encoder ablations** require fresh watermark embedding because the image changes: `all_T44`, `all_T11`, `all_T22`, `LL_only`, `one_copy_per_group`, `reduced_refinement`. Do not claim they isolate only decoding, and never compare PSNR across different attacked images without disclosing it.

## Interpretation constraints

- PSNR >45 dB and clean NC=1 are **targets**, not guaranteed on all host/WM pairs, keys or ablations. Use `--strict` to fail on violations.
- A `detected=False` row is a detection failure even if a postprocessed watermark has high NC.
- Additional learned threshold selection should use development data only.
- For one host pair, mean differences are descriptive; do not claim statistical significance or generalization.
- Schur of symmetric 4×4 component has four real eigenvalues sorted algebraically: the fixed diagonal indices are exact and reproducible. Energy dominance does not by itself prove robustness.
- Different attack sets must be labeled distinctly: this suite replaces color quantization with WebP.
