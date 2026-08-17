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

Fill the Cyrius columns as modules land. A blank row is an unported module, not
a regression.

| Benchmark | Rust v1.0.1 | Rust (ns) | Cyrius v2.0 | Delta |
|---|---:|---:|---:|---:|
| `affine_rotate_512x512` | 7.270 ms | 7,269,900.0 | — | — |
| `apply_mask_1080p` | 1.469 ms | 1,469,100.0 | — | — |
| `argb8_to_rgba8_1080p` | 2.108 ms | 2,107,600.0 | — | — |
| `argb_to_nv12_1080p` | 4.679 ms | 4,678,900.0 | — | — |
| `auto_levels_1080p` | 8.177 ms | 8,177,300.0 | — | — |
| `auto_white_balance_1080p` | 6.480 ms | 6,479,800.0 | — | — |
| `bilateral_r2_256x256` | 14.901 ms | 14,901,000.0 | — | — |
| `blackbody_spd` | 984.990 ns | 985.0 | — | — |
| `blend_pixel_argb_normal` | 14.411 ns | 14.4 | — | — |
| `blend_pixel_modes/ColorBurn` | 17.015 ns | 17.0 | — | — |
| `blend_pixel_modes/ColorDodge` | 16.189 ns | 16.2 | — | — |
| `blend_pixel_modes/Darken` | 15.051 ns | 15.1 | — | — |
| `blend_pixel_modes/Difference` | 14.062 ns | 14.1 | — | — |
| `blend_pixel_modes/Exclusion` | 14.871 ns | 14.9 | — | — |
| `blend_pixel_modes/HardLight` | 15.006 ns | 15.0 | — | — |
| `blend_pixel_modes/Lighten` | 14.853 ns | 14.9 | — | — |
| `blend_pixel_modes/Multiply` | 13.825 ns | 13.8 | — | — |
| `blend_pixel_modes/Normal` | 12.793 ns | 12.8 | — | — |
| `blend_pixel_modes/Overlay` | 15.611 ns | 15.6 | — | — |
| `blend_pixel_modes/Screen` | 15.248 ns | 15.2 | — | — |
| `blend_pixel_modes/SoftLight` | 17.139 ns | 17.1 | — | — |
| `blend_pixel_normal` | 12.791 ns | 12.8 | — | — |
| `blend_row_1080p_width` | 3.640 µs | 3,640.1 | — | — |
| `blend_row_1920px` | 3.634 µs | 3,633.8 | — | — |
| `blend_row_argb_1920px` | 3.936 µs | 3,935.5 | — | — |
| `box_blur_r3_1080p` | 18.598 ms | 18,598,000.0 | — | — |
| `brightness_1080p` | 277.910 µs | 277,910.0 | — | — |
| `buffer_pool_acquire_release` | 107.810 µs | 107,810.0 | — | — |
| `channel_mixer_1080p` | 5.230 ms | 5,230,200.0 | — | — |
| `checkerboard_1080p` | 6.315 ms | 6,315,200.0 | — | — |
| `chi_squared_256bins` | 308.240 ns | 308.2 | — | — |
| `cie_cmf_at` | 3.588 ns | 3.6 | — | — |
| `cmyk_roundtrip` | 13.076 ns | 13.1 | — | — |
| `color_balance_1080p` | 10.037 ms | 10,037,000.0 | — | — |
| `color_rendering_index` | 885.430 ns | 885.4 | — | — |
| `color_temperature_6600K` | 7.788 ns | 7.8 | — | — |
| `color_temperature_to_rgb` | 14.540 ns | 14.5 | — | — |
| `composite_at_640x480_on_1080p` | 1.727 ms | 1,727,000.0 | — | — |
| `composite_at_argb_640x480_on_1080p` | 1.823 ms | 1,822,500.0 | — | — |
| `contrast_1080p` | 7.325 ms | 7,324,900.0 | — | — |
| `cpu_blend_normal_1080p` | 4.686 ms | 4,686,200.0 | — | — |
| `cpu_brightness_1080p` | 296.100 µs | 296,100.0 | — | — |
| `cpu_crop_1080p_to_720p` | 156.250 µs | 156,250.0 | — | — |
| `cpu_dissolve_1080p` | 10.237 ms | 10,237,000.0 | — | — |
| `cpu_fade_1080p` | 7.108 ms | 7,108,000.0 | — | — |
| `cpu_flip_horizontal_1080p` | 1.985 ms | 1,984,900.0 | — | — |
| `cpu_gaussian_blur_r3_1080p` | 16.077 ms | 16,077,000.0 | — | — |
| `cpu_grayscale_1080p` | 1.564 ms | 1,564,200.0 | — | — |
| `cpu_invert_1080p` | 1.073 ms | 1,072,700.0 | — | — |
| `cpu_noise_gaussian_1080p` | 60.604 ms | 60,604,000.0 | — | — |
| `cpu_resize_bilinear_1080p_to_720p` | 19.376 ms | 19,376,000.0 | — | — |
| `cpu_saturation_1080p` | 5.951 ms | 5,951,100.0 | — | — |
| `crop_1080p_to_720p` | 139.960 µs | 139,960.0 | — | — |
| `curves_1080p` | 2.016 ms | 2,015,600.0 | — | — |
| `delta_e_cie76` | 2.364 ns | 2.4 | — | — |
| `delta_e_cie94` | 7.401 ns | 7.4 | — | — |
| `delta_e_ciede2000` | 111.280 ns | 111.3 | — | — |
| `dissolve_1080p` | 9.453 ms | 9,453,200.0 | — | — |
| `equalize_1080p` | 10.365 ms | 10,365,000.0 | — | — |
| `fade_1080p` | 7.005 ms | 7,004,800.0 | — | — |
| `fill_solid_1080p` | 149.610 µs | 149,610.0 | — | — |
| `flip_horizontal_1080p` | 1.804 ms | 1,804,500.0 | — | — |
| `flip_vertical_1080p` | 583.610 µs | 583,610.0 | — | — |
| `flood_fill_1080p_uniform` | 17.999 ms | 17,999,000.0 | — | — |
| `gaussian_blur_r3_1080p` | 18.503 ms | 18,503,000.0 | — | — |
| `gpu_blend_normal_1080p` | 3.832 ms | 3,831,500.0 | — | — |
| `gpu_brightness_contrast_1080p` | 3.153 ms | 3,153,200.0 | — | — |
| `gpu_chain_invert_brightness_saturation_1080p` | 4.945 ms | 4,945,400.0 | — | — |
| `gpu_crop_1080p_to_720p` | 2.540 ms | 2,539,800.0 | — | — |
| `gpu_dissolve_1080p` | 4.716 ms | 4,716,200.0 | — | — |
| `gpu_fade_1080p` | 3.656 ms | 3,655,700.0 | — | — |
| `gpu_flip_horizontal_1080p` | 4.700 ms | 4,699,900.0 | — | — |
| `gpu_gaussian_blur_r3_1080p` | 7.441 ms | 7,441,500.0 | — | — |
| `gpu_grayscale_1080p` | 3.003 ms | 3,003,500.0 | — | — |
| `gpu_invert_1080p` | 2.993 ms | 2,993,400.0 | — | — |
| `gpu_noise_gaussian_1080p` | 2.993 ms | 2,993,300.0 | — | — |
| `gpu_resize_bilinear_1080p_to_720p` | 2.238 ms | 2,237,800.0 | — | — |
| `gpu_saturation_1080p` | 3.177 ms | 3,177,100.0 | — | — |
| `gpu_sequential_invert_brightness_saturation_1080p` | 8.659 ms | 8,658,800.0 | — | — |
| `gradient_linear_1080p` | 10.373 ms | 10,373,000.0 | — | — |
| `gradient_linear_angled_1080p` | 13.798 ms | 13,798,000.0 | — | — |
| `gradient_radial_1080p` | 16.079 ms | 16,079,000.0 | — | — |
| `grayscale_1080p` | 1.504 ms | 1,503,600.0 | — | — |
| `hsl_roundtrip` | 20.176 ns | 20.2 | — | — |
| `hue_shift_1080p` | 48.808 ms | 48,808,000.0 | — | — |
| `icc_apply_black` | 20.138 ns | 20.1 | — | — |
| `icc_apply_mid_gray` | 48.006 ns | 48.0 | — | — |
| `icc_apply_saturated_red` | 21.951 ns | 22.0 | — | — |
| `icc_apply_white` | 30.638 ns | 30.6 | — | — |
| `icc_from_bytes_srgb_v2` | 76.191 ns | 76.2 | — | — |
| `icc_tone_curve_apply` | 11.386 ns | 11.4 | — | — |
| `invert_1080p` | 1.060 ms | 1,060,000.0 | — | — |
| `levels_1080p` | 2.011 ms | 2,010,600.0 | — | — |
| `luminance_histogram_1080p` | 4.704 ms | 4,704,100.0 | — | — |
| `lut3d_17cube_1080p` | 57.743 ms | 57,743,000.0 | — | — |
| `median_r1_512x512` | 29.341 ms | 29,341,000.0 | — | — |
| `median_r5_512x512` | 40.317 ms | 40,317,000.0 | — | — |
| `noise_gaussian_1080p` | 59.608 ms | 59,608,000.0 | — | — |
| `noise_salt_pepper_1080p` | 4.823 ms | 4,823,200.0 | — | — |
| `nv12_to_rgba_1080p` | 1.018 ms | 1,017,700.0 | — | — |
| `p3_to_linear_srgb` | 1.542 ns | 1.5 | — | — |
| `perspective_identity_512x512` | 8.796 ms | 8,796,200.0 | — | — |
| `pixel_view_create` | 8.615 ns | 8.6 | — | — |
| `premultiply_alpha_1080p` | 3.525 ms | 3,525,300.0 | — | — |
| `resize_bicubic_1080p_to_720p` | 104.830 ms | 104,830,000.0 | — | — |
| `resize_bilinear_1080p_to_720p` | 19.722 ms | 19,722,000.0 | — | — |
| `resize_nearest_1080p_to_720p` | 1.592 ms | 1,592,100.0 | — | — |
| `rgb8_to_rgba8_1080p` | 2.538 ms | 2,537,700.0 | — | — |
| `rgb_histograms_1080p` | 1.798 ms | 1,798,000.0 | — | — |
| `rgba8_to_argb8_1080p` | 2.162 ms | 2,161,800.0 | — | — |
| `rgba8_to_rgb8_1080p` | 2.374 ms | 2,374,000.0 | — | — |
| `rgba8_to_rgbaf32_1080p` | 7.625 ms | 7,624,500.0 | — | — |
| `rgba_to_yuv420p_bt2020_1080p` | 2.481 ms | 2,481,100.0 | — | — |
| `rgba_to_yuv420p_bt601_1080p` | 2.195 ms | 2,194,600.0 | — | — |
| `rgba_to_yuv420p_bt709_1080p` | 2.223 ms | 2,222,700.0 | — | — |
| `rgbaf32_to_rgba8_1080p` | 15.016 ms | 15,016,000.0 | — | — |
| `saturation_1080p` | 5.893 ms | 5,893,400.0 | — | — |
| `spd_to_xyz` | 66.753 ns | 66.8 | — | — |
| `srgb_to_lab` | 87.461 ns | 87.5 | — | — |
| `srgb_v2_profile_generation` | 24.534 ns | 24.5 | — | — |
| `threshold_1080p` | 2.746 ms | 2,746,400.0 | — | — |
| `tone_curve_gamma_2_2` | 11.624 ns | 11.6 | — | — |
| `tone_curve_table_256` | 4.489 ns | 4.5 | — | — |
| `unpremultiply_alpha_1080p` | 6.354 ms | 6,354,100.0 | — | — |
| `unsharp_mask_r2_1080p` | 28.381 ms | 28,381,000.0 | — | — |
| `vibrance_1080p` | 8.416 ms | 8,416,500.0 | — | — |
| `vignette_1080p` | 12.805 ms | 12,805,000.0 | — | — |
| `wavelength_to_xyz` | 3.628 ns | 3.6 | — | — |
| `wipe_1080p` | 2.232 ms | 2,232,100.0 | — | — |
| `xyz_conversion_roundtrip` | 1.040 ns | 1.0 | — | — |
| `xyz_to_cct` | 4.558 ns | 4.6 | — | — |
| `yuv420p_to_rgba_bt2020_1080p` | 1.264 ms | 1,264,300.0 | — | — |
| `yuv420p_to_rgba_bt601_1080p` | 1.197 ms | 1,197,400.0 | — | — |
| `yuv420p_to_rgba_bt709_1080p` | 1.207 ms | 1,206,600.0 | — | — |

## Notes on specific benchmarks

- **`median_r1_512x512` (29.341 ms)** — this is the 1.0.0 Huang rewrite's
  small-radius tradeoff, not a regression to fix. At r=1 the 9-sample window is
  cheaper to sort than to build 256-bin histograms for. `median_r5_512x512`
  (40.317 ms) exists to show the radius scaling the rewrite was actually for;
  port both or neither.
- **GPU benchmarks** (`gpu_*`) are measured against wgpu on the Rust side. The
  Cyrius port targets mabda's **native** AMD/NVIDIA backends, so these are not a
  like-for-like comparison — record the backend alongside the number.
- **`cpu_*` / `gpu_*` pairs** exist to locate the GPU offload crossover. Keep
  them paired; a Cyrius run that ports only one half of a pair is not comparable.
