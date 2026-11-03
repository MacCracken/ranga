# Benchmarks — Rust 1.0.1 vs. Cyrius 2.3.0

The Rust side is the crate frozen at tag `1.0.1` and kept at `rust-old/`. The Cyrius
side is 2.3.0. Since 2.3.0 every one of Rust's 108 CPU benchmarks has a Cyrius
counterpart **under the same name and on the same workload**: `tests/ranga.bcyr` (100)
and `tests/spectral.bcyr` (8, which needs prakash). Rust's 26 GPU benchmarks wait on a
decision about GPU hardware (roadmap 2.3.x).

The full Rust-era series, 534 rows from 0.20.3 to 1.0.1, is preserved in
[`rust-v1-bench-history.csv`](rust-v1-bench-history.csv).
[ADR 007](decisions/007-final-rust-release.md) makes it the Cyrius implementation's
performance target. Cyrius runs are recorded in `bench-history.csv` by
`scripts/bench-history.sh`.

## Source size

| | Rust 1.0.1 | Cyrius 2.3.0 |
|---|---:|---:|
| Library source | 13,958 lines across 18 files | 11,961 lines across 22 files |
| Benchmark harness | 1,353 lines across 10 criterion suites | 755 lines across 2 suites |
| Tests | 2,292 lines across 6 integration files, plus in-module tests | 9,782 lines across 23 suites |
| Fuzz targets | 532 lines across 8 targets | 1,292 lines across 8 harnesses |
| Examples | 893 lines across 11 examples | none (roadmap 2.7.x) |

## How these numbers were taken

Both sides ran on the same machine in the same session on 2026-11-02, one after the
other: an AMD Ryzen 7 5800H (8 cores, 16 threads) under the `powersave` governor, in a
desktop session with a load average of 2.3–2.8. That is **not a quiesced machine**, so
read the ratios, which share the conditions, rather than the absolutes.

- **Rust:** `rust-old/` at 1.0.1, built by the current stable rustc (1.99.1) from a
  scratch copy, so `rust-old/` itself was not touched. criterion 0.8 with
  `--warm-up-time 1 --measurement-time 2`; the figure is criterion's point estimate.
  ```sh
  cargo bench --features spectral --bench blend --bench color_convert --bench color_science \
    --bench filter --bench transform --bench composite --bench histogram --bench icc \
    --bench spectral -- --warm-up-time 1 --measurement-time 2 --noplot
  ```
- **Cyrius:** cyrius 6.7.18, `./scripts/bench-history.sh`. The figure is the mean over
  a fixed batch, sized to take roughly 50–300 ms, with the timer floor subtracted.

The "frozen series" column is the 1.0.1 row of `rust-v1-bench-history.csv`, recorded
in August 2026 with that era's toolchain. This run's Rust numbers agree with it to a
median of 0.98×. The exceptions are the separable blurs: `gaussian_blur_r3` and
`box_blur_r3` take 61 ms today against 18.5 ms then, and `unsharp_mask_r2` 59 ms
against 28 ms, with tight intervals on both runs. The most likely cause is codegen in
the newer rustc, which was not investigated. `resize_bicubic` went the other way,
55 ms against 105 ms. Against the frozen series the blurs would be about 22× rather
than about 7×.

**What "same workload" means.** The inputs and parameters are Rust's: uniform 128
frames where Rust uses `make_buf`, `i % 256` where it uses `make_varied_buf`, opacity
200 for blend, a 640×480 source at (100, 100) for `composite_at`, the 17-point
identity cube, and so on. In-place operations run repeatedly on one buffer, as Rust's
do. Two differences remain:

- Rust's colour structs live on the stack and `.into()` allocates nothing. The port
  returns heap handles, so `srgb_to_lab`, `cmyk_roundtrip`, `hsl_roundtrip`,
  `pixel_view_create`, `spd_to_xyz`, `wavelength_to_xyz` and
  `xyz_conversion_roundtrip` each include an allocation and a free.
- `curves_1080p` builds its LUT in f64 rather than f32. The table only has to be a
  plausible curve, since `curves` is a lookup either way.

Up to 2.2.0, 35 benchmarks existed, and several shared a Rust name but not its
workload. Two were also renamed to Rust's names in 2.3.0 (`composite_at_1080p` and
`rgba_to_yuv420p_1080p`). So `bench-history.csv` rows from before 2.3.0 do not compare
with later ones.

## Results

**Median 8.0× slower than Rust, from 0.5× to 367×.**

| Ratio | Benchmarks |
|---|---:|
| faster than Rust (< 1×) | 2 |
| 1–2× | 2 |
| 2–4× | 6 |
| 4–8× | 43 |
| 8–16× | 24 |
| 16× and over | 31 |

### Reading the spread

- **Faster or level:** `blend_row` (0.5×, both widths), an SSE2 kernel, and
  `checkerboard` (1.0×). `gradient_linear` (1.3×) computes one row and copies it down.
- **The 4–8× band,** where most benchmarks sit, is per-pixel arithmetic with no Rust
  SIMD counterpart: the blend modes, the delta-E formulas, `median`, the YUV forward
  converters and the blurs as measured today. This is roughly the cost of the language
  difference, and the band the rest should be pulled toward.
- **Byte loops in the standard library.** At cyrius 6.7.18, `memcpy` and `memset` in
  `lib/string.cyr` move one byte per iteration. The x86 `rep movsb` path was removed
  because the aarch64 backend emitted it too and lost instruction alignment. Rust's
  are vectorised. This accounts for:
  - `buffer_pool_acquire_release` (127×): every acquire re-zeroes an 8.3 MB buffer,
    as Rust's `resize(size, 0)` does;
  - `crop` (53×) and `flip_vertical` (32×), which copy whole rows with `memcpy`;
  - `flip_horizontal` (18×) and `fill_solid` (18.5×), which store per pixel.

  An architecture-guarded fast path belongs upstream. A word-at-a-time copy in ranga
  is the local alternative (roadmap 2.6.x).
- **Per-byte f32 through helper calls:** `fade` (57×), `dissolve` (50×), the angled
  and radial gradients (37–42×), `vibrance` (27×), `color_balance` (21×),
  `channel_mixer` and `contrast` (16–17×). `fade` makes four helper calls per
  channel, and Cyrius does not inline (see "What the inline experiment showed"
  below). Inlining these by hand, as was done for the blurs in 2.0.x, is the first
  thing to try (roadmap 2.6.x).
- **`equalize` (31×) and `auto_levels` (41×)** make two passes over the frame, the
  second converting each channel through f32 helper calls.
- **The YUV inverses (37×) and `nv12_to_rgba` (52×)** are SSE2 in Rust, 8 pixels per
  iteration, and scalar here.
- **Nanosecond scalars:** `xyz_conversion_roundtrip` (53×), `wavelength_to_xyz` (25×),
  `cie_cmf_at`, `p3_to_linear_srgb` and the tone curves (18×). Rust inlines these to a
  few instructions; here each is a call, often with a heap handle.
- **`color_rendering_index` (367×) is upstream:** 310 µs per call in prakash 2.15.1
  against 0.84 µs in Rust. ranga only forwards to it.
- **`srgb_v2_profile_generation` (40×) and `icc_from_bytes_srgb_v2` (18×)** build and
  parse the profile field by field with byte stores and loads, plus an allocation
  each.

The full table follows, grouped by Rust bench file in Rust's order.

### `blend.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `blend_pixel_normal` | 12.8 ns | 13.0 ns | 88.0 ns | 6.8× |
| `blend_row_1920px` | 3.63 µs | 3.67 µs | 1.77 µs | 0.5× |
| `blend_row_1080p_width` | 3.64 µs | 3.67 µs | 1.78 µs | 0.5× |
| `blend_pixel_modes/Normal` | 12.8 ns | 12.9 ns | 87.0 ns | 6.7× |
| `blend_pixel_modes/Multiply` | 13.8 ns | 13.6 ns | 94.0 ns | 6.9× |
| `blend_pixel_modes/Screen` | 15.2 ns | 15.4 ns | 102.0 ns | 6.6× |
| `blend_pixel_modes/Overlay` | 15.6 ns | 15.6 ns | 104.0 ns | 6.7× |
| `blend_pixel_modes/Darken` | 15.1 ns | 13.5 ns | 132.0 ns | 9.8× |
| `blend_pixel_modes/Lighten` | 14.9 ns | 13.5 ns | 134.0 ns | 9.9× |
| `blend_pixel_modes/ColorDodge` | 16.2 ns | 15.9 ns | 146.0 ns | 9.2× |
| `blend_pixel_modes/ColorBurn` | 17.0 ns | 16.7 ns | 155.0 ns | 9.3× |
| `blend_pixel_modes/SoftLight` | 17.1 ns | 17.1 ns | 131.0 ns | 7.7× |
| `blend_pixel_modes/HardLight` | 15.0 ns | 14.7 ns | 110.0 ns | 7.5× |
| `blend_pixel_modes/Difference` | 14.1 ns | 14.2 ns | 104.0 ns | 7.3× |
| `blend_pixel_modes/Exclusion` | 14.9 ns | 14.9 ns | 116.0 ns | 7.8× |
| `blend_pixel_argb_normal` | 14.4 ns | 14.3 ns | 150.0 ns | 10.5× |
| `blend_row_argb_1920px` | 3.94 µs | 3.90 µs | 52.44 µs | 13.4× |

### `color_convert.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `rgba_to_yuv420p_bt601_1080p` | 2.19 ms | 2.38 ms | 9.97 ms | 4.2× |
| `yuv420p_to_rgba_bt601_1080p` | 1.20 ms | 1.11 ms | 41.29 ms | 37.0× |
| `rgba_to_yuv420p_bt709_1080p` | 2.22 ms | 2.18 ms | 9.93 ms | 4.6× |
| `yuv420p_to_rgba_bt709_1080p` | 1.21 ms | 1.10 ms | 41.31 ms | 37.6× |
| `nv12_to_rgba_1080p` | 1.02 ms | 806.93 µs | 41.78 ms | 51.8× |
| `rgb8_to_rgba8_1080p` | 2.54 ms | 2.50 ms | 14.25 ms | 5.7× |
| `argb8_to_rgba8_1080p` | 2.11 ms | 2.02 ms | 13.81 ms | 6.8× |
| `rgbaf32_to_rgba8_1080p` | 15.02 ms | 14.63 ms | 99.38 ms | 6.8× |
| `rgba_to_yuv420p_bt2020_1080p` | 2.48 ms | 2.63 ms | 9.97 ms | 3.8× |
| `yuv420p_to_rgba_bt2020_1080p` | 1.26 ms | 1.13 ms | 41.28 ms | 36.6× |
| `argb_to_nv12_1080p` | 4.68 ms | 4.79 ms | 20.77 ms | 4.3× |
| `rgba8_to_argb8_1080p` | 2.16 ms | 1.86 ms | 14.26 ms | 7.6× |
| `rgba8_to_rgb8_1080p` | 2.37 ms | 2.25 ms | 12.07 ms | 5.4× |
| `rgba8_to_rgbaf32_1080p` | 7.62 ms | 7.27 ms | 31.22 ms | 4.3× |

### `color_science.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `srgb_to_lab` | 87.5 ns | 83.3 ns | 1.33 µs | 16.0× |
| `delta_e_cie76` | 2.4 ns | 2.3 ns | 11.0 ns | 4.8× |
| `delta_e_cie94` | 7.4 ns | 6.9 ns | 37.0 ns | 5.4× |
| `delta_e_ciede2000` | 111.3 ns | 120.2 ns | 530.0 ns | 4.4× |
| `p3_to_linear_srgb` | 1.5 ns | 1.5 ns | 27.0 ns | 17.8× |
| `cmyk_roundtrip` | 13.1 ns | 13.0 ns | 109.0 ns | 8.4× |
| `color_temperature_6600K` | 7.8 ns | 8.0 ns | 64.0 ns | 8.0× |
| `hsl_roundtrip` | 20.2 ns | 19.1 ns | 144.0 ns | 7.5× |
| `icc_tone_curve_apply` | 11.4 ns | 11.4 ns | 207.0 ns | 18.2× |
| `buffer_pool_acquire_release` | 107.81 µs | 92.66 µs | 11.80 ms | 127.4× |
| `pixel_view_create` | 8.6 ns | 8.9 ns | 43.0 ns | 4.8× |

### `composite.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `premultiply_alpha_1080p` | 3.53 ms | 3.54 ms | 33.80 ms | 9.5× |
| `composite_at_640x480_on_1080p` | 1.73 ms | 1.75 ms | 9.63 ms | 5.5× |
| `dissolve_1080p` | 9.45 ms | 9.43 ms | 475.47 ms | 50.4× |
| `fade_1080p` | 7.00 ms | 5.52 ms | 315.92 ms | 57.3× |
| `wipe_1080p` | 2.23 ms | 2.16 ms | 14.54 ms | 6.7× |
| `gradient_linear_1080p` | 10.37 ms | 10.50 ms | 13.55 ms | 1.3× |
| `checkerboard_1080p` | 6.32 ms | 6.38 ms | 6.68 ms | 1.0× |
| `unpremultiply_alpha_1080p` | 6.35 ms | 6.37 ms | 17.79 ms | 2.8× |
| `apply_mask_1080p` | 1.47 ms | 1.56 ms | 12.84 ms | 8.2× |
| `gradient_radial_1080p` | 16.08 ms | 16.16 ms | 600.02 ms | 37.1× |
| `gradient_linear_angled_1080p` | 13.80 ms | 13.75 ms | 580.25 ms | 42.2× |
| `fill_solid_1080p` | 149.61 µs | 148.77 µs | 2.75 ms | 18.5× |
| `composite_at_argb_640x480_on_1080p` | 1.82 ms | 1.81 ms | 9.55 ms | 5.3× |

### `filter.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `brightness_1080p` | 277.91 µs | 282.41 µs | 3.65 ms | 12.9× |
| `contrast_1080p` | 7.32 ms | 5.83 ms | 93.76 ms | 16.1× |
| `saturation_1080p` | 5.89 ms | 7.07 ms | 84.72 ms | 12.0× |
| `grayscale_1080p` | 1.50 ms | 1.48 ms | 8.86 ms | 6.0× |
| `invert_1080p` | 1.06 ms | 1.03 ms | 3.38 ms | 3.3× |
| `levels_1080p` | 2.01 ms | 2.00 ms | 8.48 ms | 4.2× |
| `curves_1080p` | 2.02 ms | 1.99 ms | 9.08 ms | 4.6× |
| `gaussian_blur_r3_1080p` | 18.50 ms | 61.40 ms | 407.10 ms | 6.6× |
| `box_blur_r3_1080p` | 18.60 ms | 60.49 ms | 406.92 ms | 6.7× |
| `unsharp_mask_r2_1080p` | 28.38 ms | 58.73 ms | 471.02 ms | 8.0× |
| `hue_shift_1080p` | 48.81 ms | 47.08 ms | 382.88 ms | 8.1× |
| `color_balance_1080p` | 10.04 ms | 10.42 ms | 213.29 ms | 20.5× |
| `vignette_1080p` | 12.80 ms | 12.54 ms | 133.68 ms | 10.7× |
| `lut3d_17cube_1080p` | 57.74 ms | 56.31 ms | 800.55 ms | 14.2× |
| `noise_gaussian_1080p` | 59.61 ms | 58.39 ms | 467.63 ms | 8.0× |
| `noise_salt_pepper_1080p` | 4.82 ms | 4.82 ms | 30.39 ms | 6.3× |
| `median_r1_512x512` | 29.34 ms | 28.65 ms | 198.54 ms | 6.9× |
| `median_r5_512x512` | 40.32 ms | 37.21 ms | 262.14 ms | 7.0× |
| `bilateral_r2_256x256` | 14.90 ms | 12.82 ms | 134.12 ms | 10.5× |
| `vibrance_1080p` | 8.42 ms | 8.60 ms | 227.62 ms | 26.5× |
| `channel_mixer_1080p` | 5.23 ms | 6.40 ms | 107.83 ms | 16.8× |
| `threshold_1080p` | 2.75 ms | 2.66 ms | 9.52 ms | 3.6× |
| `flood_fill_1080p_uniform` | 18.00 ms | 18.10 ms | 158.79 ms | 8.8× |
| `auto_white_balance_1080p` | 6.48 ms | 5.85 ms | 62.31 ms | 10.7× |

### `histogram.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `luminance_histogram_1080p` | 4.70 ms | 4.61 ms | 19.19 ms | 4.2× |
| `rgb_histograms_1080p` | 1.80 ms | 1.68 ms | 11.57 ms | 6.9× |
| `equalize_1080p` | 10.37 ms | 10.23 ms | 321.12 ms | 31.4× |
| `auto_levels_1080p` | 8.18 ms | 7.44 ms | 301.15 ms | 40.5× |
| `chi_squared_256bins` | 308.2 ns | 290.0 ns | 1.88 µs | 6.5× |

### `icc.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `srgb_v2_profile_generation` | 24.5 ns | 23.1 ns | 927.0 ns | 40.2× |
| `icc_from_bytes_srgb_v2` | 76.2 ns | 68.7 ns | 1.23 µs | 17.8× |
| `icc_apply_mid_gray` | 48.0 ns | 37.1 ns | 645.0 ns | 17.4× |
| `icc_apply_white` | 30.6 ns | 26.0 ns | 62.0 ns | 2.4× |
| `icc_apply_black` | 20.1 ns | 18.9 ns | 113.0 ns | 6.0× |
| `icc_apply_saturated_red` | 22.0 ns | 19.9 ns | 98.0 ns | 4.9× |
| `tone_curve_gamma_2_2` | 11.6 ns | 11.2 ns | 204.0 ns | 18.3× |
| `tone_curve_table_256` | 4.5 ns | 4.4 ns | 19.0 ns | 4.3× |

### `transform.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `crop_1080p_to_720p` | 139.96 µs | 125.22 µs | 6.64 ms | 53.0× |
| `resize_nearest_1080p_to_720p` | 1.59 ms | 1.47 ms | 4.16 ms | 2.8× |
| `resize_bilinear_1080p_to_720p` | 19.72 ms | 18.13 ms | 92.66 ms | 5.1× |
| `flip_horizontal_1080p` | 1.80 ms | 1.51 ms | 26.94 ms | 17.9× |
| `flip_vertical_1080p` | 583.61 µs | 453.05 µs | 14.34 ms | 31.6× |
| `affine_rotate_512x512` | 7.27 ms | 6.26 ms | 30.62 ms | 4.9× |
| `resize_bicubic_1080p_to_720p` | 104.83 ms | 54.81 ms | 1.42 s | 25.9× |
| `perspective_identity_512x512` | 8.80 ms | 7.96 ms | 34.72 ms | 4.4× |

### `spectral.rs`

| Benchmark | Rust 1.0.1 (frozen series) | Rust 1.0.1 (this run) | Cyrius 2.3.0 | Ratio |
|---|---:|---:|---:|---:|
| `spd_to_xyz` | 66.8 ns | 63.5 ns | 1.53 µs | 24.1× |
| `xyz_to_cct` | 4.6 ns | 3.6 ns | 42.0 ns | 11.7× |
| `wavelength_to_xyz` | 3.6 ns | 3.4 ns | 87.0 ns | 25.3× |
| `blackbody_spd` | 985.0 ns | 941.2 ns | 9.85 µs | 10.5× |
| `color_temperature_to_rgb` | 14.5 ns | 14.3 ns | 122.0 ns | 8.6× |
| `cie_cmf_at` | 3.6 ns | 3.5 ns | 62.0 ns | 17.9× |
| `color_rendering_index` | 885.4 ns | 844.0 ns | 309.64 µs | 366.9× |
| `xyz_conversion_roundtrip` | 1.0 ns | 1.0 ns | 53.0 ns | 53.0× |

## Earlier performance passes (2.0.x)

These sections record how the first gaps were closed. Their numbers come from the
2.0.x benchmarks, measured against the frozen series, and several of those benchmarks
had workloads that differ from Rust's (see above).

#### The allocator, not the arithmetic

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

#### Vector kernels, in the order they were cheapest to write

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

#### First vector kernels + a loop hoist

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

#### What the inline experiment showed

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

## Caveats

- Machine state is not benchmark-clean (see "How these numbers were taken"). Absolutes
  will move on a quiesced machine; the ratios share the conditions.
- A correct SIMD path is invisible to assertions. These benchmarks are the only
  evidence that vector code is executing at all.
- Rust's GPU benchmarks (`gpu.rs`, 26 of them, half of which are CPU baselines run
  beside the GPU ones) are not ported. See roadmap 2.3.x.
