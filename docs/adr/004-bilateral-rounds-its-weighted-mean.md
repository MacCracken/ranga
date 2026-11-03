# 004 — `bilateral` rounds its weighted mean

**Status**: Accepted
**Date**: 2026-11-02

## Context

`bilateral` replaces each channel with a weighted mean of its neighbourhood, where
the weight is a spatial Gaussian times a colour Gaussian. Rust 1.0.1 converted that
mean to a byte with `(sum / sum_w).clamp(0.0, 255.0) as u8`, which truncates. Up to
2.3.1 the port did the same. The oracle confirmed it matched Rust bit for bit
(group `bilateral`), and `fuzz_blur` allowed Rust's [v − 1, v] on a uniform image.

Truncation drops the fraction, so every output sits up to one count low:

- **A uniform image did not survive.** The f32 mean of a flat neighbourhood lands a
  few ULPs either side of v, and truncation turns the low side into v − 1. Over every
  value 0..255, radii 1..6 and two sigma pairs, 1,423 of 3,072 cases came back
  changed.
- **Every image was biased dark.** The measurement used 40,824 channel values from 80
  random images (radii 1..4, sigma_space 0.5..9.9, sigma_color 5..64), compared
  against the exact weighted mean computed in f64 from the same weights:

| | Mean error | Worst error | Values off by more than ½ count |
|---|---:|---:|---:|
| Rust 1.0.1 | −0.466 | 1.000 | 19,017 (46.6%) |
| ranga 2.3.2 | −0.000 | 0.500 | 0 |

This is the same defect class as the separable blurs
([ADR 003](003-blur-keeps-an-f32-intermediate-and-rounds-once.md)), with one
quantisation instead of two.

## Decision

`bilateral` rounds its weighted mean half up: `(sum / sum_w + 0.5)` through the
existing clamp (`_fk_f32_to_u8`).

Nothing else changes:
- the weights are still single precision, with f32 `exp`, as in Rust;
- the spatial table is the same;
- sigma ≤ 0 still reports `RG_ERR_OTHER`;
- a NaN sigma still copies every pixel, because `sum_w > 0` fails;
- radius 0 is still a clone;
- alpha is still copied.

## Consequences

- **Positive**
  - A uniform image survives exactly. `fuzz_blur` asserts this on random sizes, and
    `tests/filter_kernel.tcyr` checks every value 0..255 at radii 1..6 with two
    sigma pairs.
  - The output is within half a count of the exact weighted mean, with no bias.
  - The speed is unchanged: `bilateral_r2_256x256` measured 141 ms against 140 ms
    back to back. The division is now inline instead of a helper call, which pays
    for the add.
- **Negative**
  - It no longer matches Rust 1.0.1 byte for byte. Values move by 0 or +1, never
    down: 46.6% of them in the measurement above.
- **Neutral**
  - The oracle group `bilateral` is in `scripts/oracle/compare.py`'s `ACCEPTED` with
    a bound of 1 count. A difference larger than 1 still fails the run.
    `bilateral.code` (the error codes) stays exact.
  - `tests/oracle_golden.tcyr` no longer digests `bilateral`, because
    `make_golden.py` skips accepted groups. The CI oracle step still compares it
    against Rust within the bound.

## Alternatives considered

- **Keep Rust's output.** Rejected. Matching a defect is not correctness (ADR 003).
- **Compute the mean in f64.** This would move the result a few ULPs, not half a
  count, so it fixes neither the bias nor the uniform case without the rounding.
  Once there is rounding, it buys nothing measurable. It would also change the
  weights' precision, which the oracle had just pinned to f32 in 2.2.0.
