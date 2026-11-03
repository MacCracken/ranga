# 003 — The separable blurs keep an f32 intermediate and round once

**Status**: Accepted
**Date**: 2026-11-02

## Context

`gaussian_blur` and `box_blur` are separable: a horizontal pass and then a vertical
pass over the same kernel. `unsharp_mask` is built on `gaussian_blur`.

Rust 1.0.1 wrote the horizontal pass back to a `u8` buffer with `as u8`, then ran the
vertical pass on those bytes and truncated again. Each pixel was quantised twice and
rounded down both times. Its f32 kernels also sum to slightly less than 1: the r=1
gaussian taps sum to 0.999999881. Up to 2.3.0 the port reproduced this exactly, and
`fuzz_blur` asserted Rust's measured bounds instead of the obvious invariant.

The results were measured on rust-old and on the 2.3.0 port, which matched it byte
for byte:

- **A uniform image did not survive.** A flat image came back up to 2 counts darker:
  gaussian in 979 of 2,816 cases and box in 4,278 of 13,824 (every value, radii
  0..10). For example, a uniform 100 at r=1 went to 99 after the horizontal pass and
  98 after the vertical.
- **Every image was biased dark.** The measurement used 65,328 channel values from 60
  random images at radii 1..10, compared against an exact f64 separable convolution
  rounded once:

| | Mean error | Worst error | Values off by more than ½ count |
|---|---:|---:|---:|
| Rust 1.0.1, gaussian | −1.004 | 1.938 | 61,430 (94.0%) |
| Rust 1.0.1, box | −0.884 | 1.852 | 56,633 (86.7%) |
| ranga 2.3.0, gaussian | −0.000 | 0.500 | 1 (a tie) |
| ranga 2.3.0, box | −0.001 | 0.499 | 0 |

So the defect is not limited to flat images. Rust's blur darkened almost every value
it produced by about one count.

## Decision

ranga's separable blurs keep the intermediate in f32 and round once, half up, at the
end.

- `_fk_blur_pass_h` writes three f32 per pixel (R, G, B) to a scratch buffer of
  `12 * w * h` bytes.
- `_fk_blur_pass_v` reads that scratch, accumulates, adds 0.5 and converts with the
  existing clamp (`_fk_f32_to_u8`).
- Alpha is still not blurred: the vertical pass copies it from the source.

The kernel itself is unchanged: the same gaussian construction (sigma floored at 0.5),
the same normalisation and the same edge clamping. Within a few f32 ULPs the weights
still sum to 1, which is far inside the half count that rounding absorbs. So v in
gives v out for every v at every radius.

This covers `gaussian_blur`, `box_blur`, and through it `unsharp_mask`. `bilateral`
and `median` are separate code and are not changed by this ADR.

## Consequences

- **Positive**
  - A uniform image survives both blurs exactly. `tests/filter_kernel.tcyr` checks
    every value 0..255 at radii 1..10, and `fuzz_blur` checks exactness on random
    sizes.
  - The output is within half a count of the exact convolution, with no bias.
  - It is faster. On 1080p, measured back to back: gaussian r3 went from 408 ms to
    358 ms and box r3 from 408 ms to 358 ms (both −12%); unsharp r2 went from 475 ms
    to 430 ms. The vertical pass now loads floats instead of converting a byte to
    f32 at every tap.
- **Negative**
  - The blurs no longer match Rust 1.0.1 byte for byte. Against Rust, values move
    by 0 to +2 counts, never down: 94% of gaussian values and 87% of box values
    change in the measurement above.
  - The scratch buffer is three times larger: 12 bytes per pixel instead of 4, or
    24.9 MB at 1080p instead of 8.3 MB.
- **Neutral**
  - The oracle (`scripts/oracle`) has no blur group, so no `ACCEPTED` entry is
    needed and the golden digests are unaffected.
  - The two unit tests that pinned Rust's 98 now assert 100. A point-source test now
    pins the rounding: centre 158 and diagonal 3, where Rust gave 157 and 2.
  - The GPU blur is not implemented yet (roadmap 2.5.x). The WGSL in
    `gpu_shaders.cyr` writes its horizontal pass to packed u8, so when a native blur
    kernel is written it must follow this ADR, not that shader.

## Alternatives considered

- **Keep Rust's output.** Rejected. Matching a defect is not correctness. The
  oracle's job is to catch porting mistakes, not to preserve Rust's.
- **Round half up after each pass, keeping the u8 intermediate.** This fixes the
  uniform case, because the weights sum to 1 within ULPs. But it still quantises
  twice, so a non-uniform image picks up up to a count of extra error. It is no
  faster, since it keeps the byte-to-float conversion in the vertical pass.
- **Renormalise the kernel in f64 so it sums to exactly 1.** This fixes neither
  problem: the truncation between passes stays, and it is the larger error.
