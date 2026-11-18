"""Single proposed algorithm: blind IWT–Schur 4x4 dual-pilot watermarking.

All methods are packaged separately from the original MSDP and published baselines.
The repository's original code and baseline registrations are intentionally unchanged.
"""
from .fixed_t44 import Config, read_watermark, psnr, nc, ber
from .robust_v7 import embed, extract, Result
__all__ = ["Config", "embed", "extract", "Result", "read_watermark", "psnr", "nc", "ber"]
