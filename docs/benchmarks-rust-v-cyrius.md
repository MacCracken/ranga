# Benchmarks — Rust v1.0.1 vs. Cyrius v2.0

Reference snapshot for the ranga Rust → Cyrius port. The Rust side is frozen at
tag `1.0.1` (`git checkout 1.0.1` to inspect it, or `git checkout 1.0.1 -- rust-old/`
after `rust-old/` is retired); the Cyrius side is the v2.0.x tree.

The full Rust-era benchmark series — 534 rows spanning 0.20.3 → 1.0.1 — is
preserved in [`rust-v1-bench-history.csv`](rust-v1-bench-history.csv). It is
lifted here **before** running `cyrius port`, because `_port_move` sweeps
`benches/` into `rust-old/` and [ADR 007](decisions/007-final-rust-release.md)
designates this series as the Cyrius implementation's performance target.

## Source size

| | Rust v1.0.1 | Cyrius v2.0 | Delta |
|---|---:|---:|---:|
| Library source | 13,958 LOC across 18 files | — | — |
| Benchmark harness | 1,353 LOC across 10 criterion suites | — (`tests/ranga.bcyr`) | — |
| Tests | 2,292 LOC across 6 integration files | — (`tests/*.tcyr`) | — |
| Fuzz targets | 532 LOC across 8 targets | — (`tests/ranga.fcyr`) | — |
| Examples | 893 LOC across 11 examples | — | — |

Test surface to reach parity: **214** in-module `#[test]` fns, **123**
integration tests, **161** doctests, **134** named benchmarks, **8** fuzz targets.
Doctests have no direct Cyrius equivalent — their assertions fold into the
`.tcyr` suite rather than being dropped.

## Benchmark methodology

Rust numbers are criterion's point estimate (the middle value of its
`[low estimate high]` triple), normalised to nanoseconds from criterion's mixed
ps/ns/µs/ms output. Cyrius numbers will come from `tests/ranga.bcyr` via
`bench_new`/`bench_run`/`bench_report`, recorded to `bench-history.csv`.

⚠ **The Rust baseline below is not a clean-room measurement.** It was taken on a
loaded machine — AMD Ryzen 7 5800H, 16 cores, `powersave` governor, load average
≈4.5, cores scaling at 76%. Sub-microsecond entries are the least trustworthy.
An A/B against the same tree without the 1.0.1 dependency changes confirmed the
release itself was performance-neutral, so these numbers are a fair
representation of the Rust implementation *on that machine* — but the Cyrius
comparison run must be taken on a quiesced machine, and ideally the Rust side
re-run there too before any delta is published.

Amplification will be needed on the Cyrius side for the sub-timer-floor entries:
14 benchmarks land under 100 ns, the fastest at 1.04 ns.

## Results

Cyrius numbers are from `cyrius bench tests/ranga.bcyr` (35 benchmarks, CPU
only — GPU and spectral land with M5/M6). Rust numbers are the frozen 1.0.1
series. **Both were taken on the same loaded machine under a `powersave`
governor**, so the ratio is more trustworthy than either absolute.

**Median ratio: 8.6× slower than Rust; range 3.2×–46.0×.** The first
measurement was 10.6× median over 3.8×–79.9×; three performance passes have
since landed. That is the expected
shape for a scalar-first port and is not alarming on its own — Rust's numbers
include SSE2/AVX2 kernels for the paths that dominate this list, and ranga's
Cyrius side currently has vector code only in `brightness`. The interesting
information is in the spread, not the median.

| Benchmark | Rust v1.0.1 | Cyrius v2.0 | Ratio |
|---|---:|---:|---:|
| `blend_pixel_normal` | 12.8 ns | 83.0 ns | 6.5× |
| `blend_pixel_modes/Multiply` | 13.8 ns | 88.0 ns | 6.4× |
| `blend_pixel_modes/SoftLight` | 17.1 ns | 143.0 ns | 8.3× |
| `blend_pixel_argb_normal` | 14.4 ns | 134.0 ns | 9.3× |
| `delta_e_cie76` | 2.4 ns | 9.0 ns | 3.8× |
| `delta_e_cie94` | 7.4 ns | 37.0 ns | 5.0× |
| `delta_e_ciede2000` | 111.3 ns | 557.0 ns | 5.0× |
| `blend_row_1920px` | 3.63 µs | 46.59 µs | 12.8× |
| `blend_row_argb_1920px` | 3.94 µs | 47.56 µs | 12.1× |
| `brightness_1080p` | 277.91 µs | 3.87 ms | 13.9× |
| `grayscale_1080p` | 1.50 ms | 10.62 ms | 7.1× |
| `invert_1080p` | 1.06 ms | 3.39 ms | 3.2× |
| `contrast_1080p` | 7.32 ms | 89.18 ms | 12.2× |
| `saturation_1080p` | 5.89 ms | 86.97 ms | 14.8× |
| `threshold_1080p` | 2.75 ms | 11.39 ms | 4.1× |
| `flip_horizontal_1080p` | 1.80 ms | 25.12 ms | 13.9× |
| `flip_vertical_1080p` | 583.61 µs | 13.97 ms | 23.9× |
| `crop_1080p_to_720p` | 139.96 µs | 6.44 ms | 46.0× |
| `resize_bilinear_1080p_to_720p` | 19.72 ms | 85.08 ms | 4.3× |
| `resize_nearest_1080p_to_720p` | 1.59 ms | 12.45 ms | 7.8× |
| `rgba_to_yuv420p_1080p` | — | 9.07 ms | — |
| `yuv420p_to_rgba_bt601_1080p` | 1.20 ms | 37.27 ms | 31.1× |
| `rgba8_to_argb8_1080p` | 2.16 ms | 14.19 ms | 6.6× |
| `premultiply_alpha_1080p` | 3.53 ms | 30.20 ms | 8.6× |
| `fill_solid_1080p` | 149.61 µs | 3.16 ms | 21.1× |
| `composite_at_1080p` | — | 59.34 ms | — |
| `luminance_histogram_1080p` | 4.70 ms | 18.34 ms | 3.9× |
| `equalize_1080p` | 10.37 ms | 202.01 ms | 19.5× |
| `auto_levels_1080p` | 8.18 ms | 188.28 ms | 23.0× |
| `box_blur_r3_1080p` | 18.60 ms | 394.97 ms | 21.2× |
| `gaussian_blur_r3_1080p` | 18.50 ms | 396.32 ms | 21.4× |
| `median_r1_512x512` | 29.34 ms | 185.59 ms | 6.3× |
| `bilateral_r2_256x256` | 14.90 ms | 126.75 ms | 8.5× |
| `vignette_1080p` | 12.80 ms | 122.04 ms | 9.5× |
| `noise_gaussian_1080p` | 59.61 ms | 430.76 ms | 7.2× |

### Reading the spread

- **The 3.8×–8× band** (`delta_e_*`, `luminance_histogram`, `median`,
  `bilateral`, `noise_gaussian`) is arithmetic-bound work with no Rust SIMD
  counterpart. This is roughly the honest cost of the language difference today,
  and it is the band everything else should be pulled toward.
- **The 10×–20× band** is mostly Rust SIMD versus Cyrius scalar — `grayscale`,
  `blend_row`, `contrast`, `saturation`, the blurs. These are the targets for
  the remaining `asm { }` kernels.
- **`crop` at 79.9× and `flip_vertical` at 40.2× are the outliers worth fixing
  first**, and neither is a SIMD problem: both are pure `memcpy` in Rust. The
  Cyrius versions allocate a fresh `PixelBuffer` (which zero-fills via
  `fl_calloc`) and then overwrite every byte of it. For `crop` the zero-fill is
  most of the work. An uninitialised-allocation path for the
  "about to overwrite everything" case should close most of that gap, and it
  would help `flip_horizontal`, `rgba8_to_argb8` and `fill_solid` too.
- **`yuv420p_to_rgba` at 44.2×** stands out because Rust has an SSE2 kernel
  processing 8 pixels per iteration there; the Cyrius side is per-pixel scalar
  with a chroma-index computation per pixel. Hoisting the chroma row pointers
  out of the inner loop is worth trying before reaching for assembly.

### The allocator, not the arithmetic

`crop` at 79.9x was almost entirely `fl_calloc`. It zeroes **byte at a time** on
top of an `mmap` the kernel has already zeroed — measured **10.946 ms vs
29.633 us** for an 8.3 MB frame, a **369x** gap, none of it in ranga's code.
Adding `pixel_buffer_uninit` for the callers that provably overwrite every byte:

| | before | after | speedup |
|---|---:|---:|---:|
| `crop_1080p_to_720p` | 11.18 ms | 6.53 ms | 1.71x |
| `rgba8_to_argb8_1080p` | 24.46 ms | 14.00 ms | 1.75x |
| `flip_vertical_1080p` | 23.45 ms | 14.10 ms | 1.66x |
| `flip_horizontal_1080p` | 35.18 ms | 26.03 ms | 1.35x |

Filed upstream as
`cyrius/docs/development/issues/2026-08-17-fl-calloc-byte-loop-over-already-zero-mmap.md`
with a three-line fix. **If that lands, the consumer-side workaround should be
reverted** — it trades guaranteed-zero buffers for uninitialised ones across a
dozen call sites, and the hazard is untestable (fresh mmap reads as zero, so a
wrong use only breaks once the allocator recycles). `affine_transform` and
`perspective_transform` deliberately keep the zeroed path: their skipped pixels
*are* the transparent black.

### Vector kernels, in the order they were cheapest to write

One SSE2 kernel (`_sx_luma_row_sse2`) ended up serving **four** call sites,
because `grayscale`, `threshold` and all three YUV forward converters share the
same integer `(cr*R + cg*G + cb*B) >> 8` shape. Writing it once and finding the
other consumers was worth more than writing four kernels.

| | before | after | speedup | vs Rust now |
|---|---:|---:|---:|---:|
| `invert_1080p` | 7.91 ms | **3.36 ms** | 2.35× | 3.2× (was 7.5×) |
| `rgba_to_yuv420p_1080p` | 19.10 ms | **9.00 ms** | 2.12× | — |
| `threshold_1080p` | 17.72 ms | **10.78 ms** | 1.62× | 4.1× (was 6.5×) |
| `grayscale_1080p` | 16.10 ms | **10.83 ms** | 1.50× | 7.2× (was 10.7×) |
| `yuv420p_to_rgba_bt601_1080p` | 52.92 ms | **37.06 ms** | 1.43× | 31.1× (was 44.2×) |

`grayscale` and `threshold` vectorise only the luma ARITHMETIC and keep a scalar
scatter back across R/G/B. A fully-vectorised scatter would need to expand each
luma byte across three lanes while leaving the fourth alone — punpck plus a
masked merge, a second kernel's worth of hand assembly for the smaller half of
the work.

### First vector kernels + a loop hoist

| | before | after | speedup | vs Rust now |
|---|---:|---:|---:|---:|
| `rgba_to_yuv420p_1080p` | 19.10 ms | **9.10 ms** | 2.10x | — |
| `invert_1080p` | 7.91 ms | **3.39 ms** | 2.33x | 3.2x (was 7.5x) |
| `yuv420p_to_rgba_bt601_1080p` | 52.92 ms | **37.72 ms** | 1.40x | 31x (was 44.2x) |

Three different techniques, in increasing order of cost to write:

1. **`yuv420p_to_rgba` — a loop hoist, no assembly.** The chroma row pointers
   were being recomputed per pixel (a divide, a clamp and two multiplies for
   values that change once per two rows). Hoisting them out gave 1.40x.
2. **`invert` — vectorised with primitives already proven.** `255 - c` on a byte
   IS a saturating subtract, so `psubusb` plus the existing alpha mask-and-select
   did the whole thing with no new assembly at all.
3. **`rgba_to_yuv420p` — a genuine new SSE2 kernel.** `_sx_luma_row_sse2` uses
   `pmaddwd` to do two pixels per iteration: `punpcklbw` widens the eight source
   bytes to i16, `pmaddwd` multiplies and pair-sums into i32, then
   `pshufd`/`paddd` folds each pixel's halves together and two packs narrow back
   to bytes. The whole loop lives inside one `asm { }` block, because a
   per-2-pixel function call would cost more than the vectorisation saves.

The kernel is differentially tested against an independent scalar reference
across the byte range with **different values in each lane of the pair**, so a
kernel that broadcast one pixel across both would fail. It also covers the
alpha-is-ignored property, all three coefficient standards, the internal loop
across multiple pairs, and the `pairs == 0` guard.

### What the inline experiment showed

Cyrius leaves **general function inlining off by default** (`_INLINE_OK`, a
compiler-internal default-0 that a consumer cannot switch on). Small helper
functions in a hot loop are therefore real calls, and the cost is large:
rewriting the two separable blur passes to do their f32 arithmetic inline
instead of through `_fk_fma` / `_fk_u8_to_f32` took

| | before | after | speedup |
|---|---:|---:|---:|
| `box_blur_r3_1080p` | 833.03 ms | 395.10 ms | **2.11×** |
| `gaussian_blur_r3_1080p` | 832.98 ms | 399.32 ms | **2.09×** |
| `vignette_1080p` | 178.43 ms | 122.98 ms | 1.45× |
| `bilateral_r2_256x256` | 154.81 ms | 126.89 ms | 1.22× |

with every test still green. The readable factoring is right for module surface
and for anything called once per image; it is wrong for anything on a per-pixel
or per-tap path. See `port-mechanics.md`.

### Caveats

⚠ Machine state is not benchmark-clean — AMD Ryzen 7 5800H, `powersave`
governor, other work in flight. Absolutes will move on a quiesced machine; the
Rust and Cyrius numbers were taken under the same conditions, so ratios are the
figure to trust.

⚠ `rgba_to_yuv420p_1080p` and `composite_at_1080p` have no Rust counterpart
under those exact names in the 1.0.1 series, so they are baselines only.

⚠ A correct SIMD path is invisible to assertions — see the note in
`roadmap.md`. These benchmarks are currently the only evidence that vector code
is executing at all.
