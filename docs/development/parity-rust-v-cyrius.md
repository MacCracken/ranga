# Parity: ranga Rust 1.0.1 (`rust-old/`) vs the Cyrius port

> **Review of 2026-11-02, at ranga 2.1.1 / cyrius 6.7.18.** This replaces the M7 audit
> written before 2.0.0, whose totals and several "byte-identical" claims no longer hold
> (see [What the old audit got wrong](#what-the-old-audit-got-wrong)). Every public item
> in `rust-old/src` was matched against `src/*.cyr`, both sides were read, and the
> behavioural claims were **measured**. The crate at `rust-old/` was built with
> `cargo --release` (default features, which include `simd`), the port was built under
> cyrius 6.7.18, and the outputs were diffed value by value. Six of the findings were
> then reproduced independently. The open work this produces is sequenced in
> [`roadmap.md`](roadmap.md).
>
> **Updated for 2.1.2.** The P(-1) closeout
> ([`../audit/2026-11-02-audit.md`](../audit/2026-11-02-audit.md)) fixed two of the
> ICC items below and one undocumented trig divergence. A cross-toolchain diff also
> confirmed that every remaining SUSPECT item predates the 6.6.9 → 6.7.18 bumps.
>
> **Updated for 2.2.0.** All 18 CPU-side SUSPECT items are resolved:
> - 16 now match Rust bit for bit;
> - 2 are accepted by [ADR 002](../adr/002-f32-transcendentals-follow-correct-rounding-not-glibc.md)
>   (f32 transcendentals; glibc's are not correctly rounded).
>
> [`scripts/oracle`](../../scripts/oracle/README.md) now measures this on every run,
> and the six remaining SUSPECT items are all GPU (roadmap 2.5.x). Rows below that
> 2.2.0 changed are marked "(2.2.0)".
>
> **Updated for 2.3.1.** `gaussian_blur`, `box_blur` and `unsharp_mask` move from
> PARITY to IMPROVED. Rust's blur darkening is fixed in Cyrius, not reproduced
> ([ADR 003](../adr/003-blur-keeps-an-f32-intermediate-and-rounds-once.md)), and the
> 26 GPU benchmarks are ported.
>
> **Updated for 2.3.2.** `bilateral` moves from PARITY to IMPROVED: it rounds its
> weighted mean, where Rust truncated
> ([ADR 004](../adr/004-bilateral-rounds-its-weighted-mean.md)).
>
> **Updated for 2.4.0** (milestone 2.4.x, "surface completion"). Outside the GPU layer
> every MISSING and PARTIAL item is now ported: 19 MISSING and 10 PARTIAL items move,
> and `Lut3d::from_cube` moves from DIV-DOC to IMPROVED. The serde surface (11 items)
> is JSON in serde_json's form; it matches serde_json 1.0.151 byte for byte on encode
> and decision for decision on decode, with numbers correctly rounded where serde_json's
> best-effort parser is not (ADR 008), so it counts as IMPROVED. The 14 MISSING items
> left are all GPU (`GpuChain` and `new_with_hwaccel`).
>
> **Updated for 2.3.3.** No status changes. GPU compute is refused on every backend but
> native AMD ([ADR 005](../adr/005-gpu-compute-runs-only-on-native-amd.md)), and the
> GPU table's statuses describe the AMD backend.

## How each item is classified

| Status | Meaning |
| --- | --- |
| **PARITY** | Counterpart exists and behaves the same: argument meaning and order, clamping, rounding and error case. |
| **IMPROVED** | Counterpart exists and deliberately does better than Rust, through hardening, a Rust bug left unreproduced, or a safer bound. |
| **DIVERGENT-DOCUMENTED** | Absent or different on purpose, with the reason written down. |
| **PARTIAL** | Counterpart exists but lacks part of the Rust surface or behaviour. |
| **MISSING** | No counterpart. |
| **SUSPECT** | Counterpart exists and **computes something different from Rust without a documented reason**. Every SUSPECT below was confirmed by measurement. |

The project rule is "diverge only with an ADR". So each SUSPECT item is either fixed
to match Rust or written up as an ADR and becomes DIVERGENT-DOCUMENTED. Each MISSING
item is either ported or given an ADR. Nothing stays SUSPECT.

## Totals

| Area (Rust files) | Items | PARITY | IMPROVED | DIV-DOC | PARTIAL | MISSING | SUSPECT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| error, lib, pixel, color, convert | 88 | 66 | 6 | 16 | 0 | 0 | 0 |
| filter, transform, histogram | 73 | 60 | 10 | 3 | 0 | 0 | 0 |
| composite, blend, icc, spectral, hwaccel | 70 | 60 | 5 | 5 | 0 | 0 | 0 |
| gpu (context, buffer, shaders, pipeline) | 72 | 39 | 0 | 4 | 9 | 14 | 6 |
| **Total** | **303** | **225** | **21** | **28** | **9** | **14** | **6** |

(At 2.3.3: 207 parity, 9 improved, 29 divergent, 19 partial, 33 missing, 6 suspect.)

An item is a pub fn (impl methods included), struct, enum, const, or trait impl/derive.
Each derive counts once per type. The M7 audit counted 318 under a looser
convention, so the two totals are not comparable line for line.

## Findings that matter most

1. **Several byte outputs were not bit-exact against Rust (fixed in 2.2.0).** The old audit said "every u8
   output path is byte-exact", and that is false. The common cause is computing in f64
   and narrowing once at the end, where Rust works in f32 throughout.
   `src/color.cyr:195-200` names this exact pattern as wrong, for `srgb_to_linear`,
   which is bit-exact. Affected:
   - **Colour conversions:** `linear_to_srgb`, `lin_rgba_to_srgba`, `hsl_to_srgba`,
     `cmyk_to_srgba`, `rgbaf32_to_rgba8`.
   - **Filters:** `levels`, `bilateral`, and `hue_shift`, mostly through `srgba_to_hsl`.

   Most of the values involved are half-step inputs that sampled sweeps never hit. For
   example, f32 `0x3f010101` through `rgbaf32_to_rgba8` gives Rust 129 and the port
   gave 128.

   2.2.0 rewrote every one of them in f32, in Rust's order. The oracle's transition
   search confirms each matches Rust on every f32 in [0, 1].
2. **`affine_inverse` reassociated the translation (fixed in 2.2.0).** Rust computes
   `tx = (c*ty - d*tx) * inv`, and the port computes `-(ia*tx + ic*ty)`. For a plain
   scale+translate (`a=0.6, d=2.5, tx=ty=-1.9`), tx is `0x4009555555555555` in Rust
   and `…554` in the port, and an entire column of Nearest samples in
   `affine_transform` then comes from the wrong source column.
3. **Two ICC parse regressions came from the toolchain, not from ranga (fixed in
   2.1.2).** Since cyrius 6.6.7, `fl_alloc(0)` and `fl_calloc(0)` return 0. From
   2.0.3 to 2.1.1, a profile with zero tags, and an `mft2` LUT with zero input
   entries, failed with `RG_ERR_ALLOCATION`. Rust returns InvalidFormat for the
   first and parses the second, using identity curves. Both now match Rust. Grid
   size 1, which Rust handles, is accepted since 2.2.0.
4. **The GPU "wgpu fallback" never runs.** ranga carries all 21 WGSL sources but
   dispatches none of them, and mabda 4.2.0's context-aware wgpu dispatch is a stub
   (see [architecture note 001](../architecture/001-mabda-native-gpu-constraints.md)).
   So on wgpu every `gpu_*` operation fails. On native backends:
   - 11 of the 14 operations run;
   - `gpu_blend`, `gpu_noise_gaussian` and `gpu_gaussian_blur` are stubs on every
     backend, and so is bilinear/bicubic `gpu_resize`;
   - `GpuChain` (13 items) is not ported at all, although `hwaccel.cyr`'s cost model
     assumes it.
5. **The serde surface is 14 types (resolved in 2.4.0, ADR 008).** The old audit said
   "four enums, hand-written codecs". In fact Rust derives Serialize/Deserialize on:
   - the 8 colour structs;
   - `PixelFormat`, `ColorSpace`, `BlendMode`, `ToneCurve`, `Affine` and `Perspective`.

   Up to 2.3.3 `src/` had no codec for any of them. Since 2.4.0 each has
   `<type>_to_json` / `<type>_from_json` in `src/serde.cyr`, checked against serde_json
   1.0.151 by the oracle's serde group. Rust has **no `FromStr` anywhere**, so the old
   "missing FromStr" gaps were never gaps.

## Improved in Cyrius (keep)

- **`pixel_buffer_zeroed`**: fallible instead of panicking.
- **`pixel_buffer_row_offset` / `row_bytes`** (Rust `rows`/`rows_mut`): reject planar
  formats, and width 0 does not panic as Rust's `chunks_exact(0)` does.
- **`pixel_view_new`**: a size overflow returns an error code instead of aborting.
- **`brightness`**: a huge offset saturates. Rust's i16 wrap, which turns
  `(255, 0, 0)` at offset 1000, is not reproduced; byte-exact on 1,042 ordinary offsets.
- **`blend_row`**: an invalid mode returns `RG_ERR_INVALID_PARAMETER`.
- **`gaussian_blur`, `box_blur`, `unsharp_mask`** (2.3.1, ADR 003): the blurs keep an
  f32 intermediate and round once. Rust truncated after each pass, which biased
  output about one count dark (mean −1.00, worst −1.94) and darkened a uniform image
  by up to 2. Now the mean error is 0.00, the worst 0.50, a uniform image is exact,
  and the blurs are 12% faster.
- **`Lut3d::from_cube`** (2.4.0, ADR 007): honours the file's `DOMAIN_MIN`/`DOMAIN_MAX`,
  which Rust skipped and so mis-applied every non-[0, 1] LUT, and reads Resolve's
  `LUT_3D_INPUT_RANGE`, which Rust rejected.
- **serde** (2.4.0, ADR 008): JSON in serde_json 1.0.151's exact form, with numbers
  decoded correctly rounded; serde_json's default parser is best effort (2,782 of 4,000
  long f64 decimals right, against ranga's 4,000).
- **`bilateral`** (2.3.2, ADR 004): rounds its weighted mean. Rust truncated it, which
  biased output half a count dark (mean −0.47, worst −1.00) and darkened a uniform
  image by 1 in about half of all cases. Now the mean error is 0.00, the worst 0.50,
  and a uniform image is exact.

## Deliberate divergences (documented; not to be re-litigated)

- **No payloads:** error payloads are dropped. `ranga_error_message` returns the format
  string with its placeholders removed, and `Result` becomes the `err_out` convention.
- **One read function per pair:** `PixelView` and `PixelViewMut` share one
  representation, and `data`/`data_mut` and `rows`/`rows_mut` merge, because Cyrius
  pointers carry no mutability.
- **Checked only:** panicking variants (`buffer_size`) collapse into the checked ones.
  `pixel_format_checked_buffer_size` is bounded by i64, not usize.
- **No i16 wrap in the YUV inverses** ([ADR 001](../adr/001-yuv-inverse-does-not-reproduce-rusts-i16-wrap.md)).
  This covers all four inverses: `yuv420p_to_rgba`, `yuv420p_to_rgba_bt709`,
  `yuv420p_to_rgba_bt2020` and `nv12_to_rgba`. The ADR names them all since 2.2.0, and
  its wrong "all five failed to reproduce" note is corrected.
- **f32 transcendentals follow correct rounding**
  ([ADR 002](../adr/002-f32-transcendentals-follow-correct-rounding-not-glibc.md)).
  `oklch_to_oklab` and `color_temperature` differ from Rust-on-glibc by ≤ 2 ULP. The
  differences come from glibc's `sinf`/`cosf`/`powf`/`logf`, which are not correctly
  rounded. The oracle enforces the bound.
- **Grayscale coefficients:** GPU grayscale is BT.709 and CPU grayscale is BT.601,
  inherited from Rust.
- **Derives:** `Debug` has no Cyrius equivalent. ⚠ The old rationale, "an i64 is already
  copyable and comparable", covers the enums but **not** the heap-allocated records:
  - `Affine`, `Perspective` and `Lut3d` have no `*_clone` or `*_eq`;
  - neither do the colour structs, ICC profiles or `HwReport`.

  Aliasing a pointer and freeing one copy leaves the other dangling.
- **GPU omissions:**
  - `GpuBuffer::download_async`: mabda has no async readback.
  - `GpuContext::new` becomes `gpu_context_new_amd` / `_nvidia` / `_from_launcher`; the
    caller picks the backend.
  - `backend_name` reports mabda's backends.
- **`Lut3d::from_cube`** is listed as "not ported", with no reason. ⚠ The user guides
  still advertise it. It needs an ADR or a port.

## Per-area detail (everything that is not PARITY)

### error, lib, pixel, color, convert

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `linear_to_srgb`, `From<LinRgba> for Srgba`, `From<Srgba> for Hsl`, `From<Hsl> for Srgba`, `cmyk_to_srgb`, `srgb_to_cmyk`, `From<Oklab> for Oklch`, `rgbaf32_to_rgba8` | color.cyr, convert.cyr | PARITY (2.2.0) | Were SUSPECT: f64 where Rust is f32. Rewritten in f32 in Rust's order; bit-exact over the oracle's grids and, for the byte paths, every f32 in [0, 1]. |
| `From<Oklch> for Oklab`, `color_temperature` | color.cyr | DIV-DOC (2.2.0) | Were SUSPECT. Now f32 as in Rust; the residual ≤ 2 ULP is glibc's libm rounding, accepted by ADR 002. |
| `BufferPool::new` (and acquire/release) | pixel.cyr | PARITY (2.2.0) | Was SUSPECT: eager `max_buffers` slots. Now a growable array with swap_remove, like Rust's Vec. |
| `LinRgba`, `Hsl`, `CieXyz`, `CieLab`, `Cmyk`, `Oklab`, `Oklch` | color.cyr | PARITY ×7 (2.4.0) | Were PARTIAL (getters only). Each field has a setter, every struct (Srgba too) has `_clone` and `_eq` (IEEE `==`, as derived PartialEq), Srgba has `srgba_hash`, and all eight layouts are in the module header. |
| `PixelBuffer::into_data` | `pixel_buffer_into_bv` | PARITY (2.4.0) | Was MISSING. Frees the header, hands back the bytevec. |
| serde on `PixelFormat`, `ColorSpace` and the 8 colour structs | serde.cyr | IMPROVED ×3 (2.4.0) | Were MISSING. JSON in serde_json 1.0.151's form; decoded numbers correctly rounded (ADR 008). ColorSpace's CieLab and Oklab are `"CieLab"`/`"Oklab"`. |
| `PixelBuffer::new` overflow | pixel.cyr | PARITY (2.2.0) | Was DIV-DOC without an ADR. Now BUFFER_TOO_SMALL, as Rust's. |
| `yuv420p_to_rgba_bt709` / `_bt2020`, `nv12_to_rgba` | convert.cyr | DIV-DOC | ADR 001's i16 wrap; named in the ADR since 2.2.0. |

**Also found (no Rust counterpart):**
- **Fixed in 2.2.0:**
  - `convert.cyr` ignored `_cv_y_row`'s return value, so a failed 16-byte allocation
    handed back an uninitialised Y plane with `RG_ERR_NONE`.
  - `buffer_pool_acquire` did not guard against a negative size.
- **Resolved in 2.4.0 (ADR 006):** `bytevec.cyr`'s 19 `bv_*` functions sat in the
  unprefixed global namespace. They are `rg_bv_*` now.

### filter, transform, histogram

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `levels`, `hue_shift`, `Affine::inverse`, `affine_transform`, `Affine::is_identity` | filter_point.cyr, filter_kernel.cyr, transform.cyr | PARITY (2.2.0) | Were SUSPECT: f64 where Rust is f32 (levels, hue_shift, and bilateral's weights), a reassociated inverse, a NaN-passing identity test. Also bilateral's σ ≤ 0 → OTHER and NaN σ proceeding. Bit-exact on every oracle case. |
| `bilateral` | filter_kernel.cyr | IMPROVED (2.3.2) | PARITY in 2.2.0 (f32 weights, σ guards). 2.3.2 rounds the weighted mean where Rust truncated it (ADR 004). The weights are still bit-exact; outputs move 0 or +1 against Rust, and the oracle bounds them at 1. |
| `gaussian_blur`, `box_blur`, `unsharp_mask` | filter_kernel.cyr | IMPROVED (2.3.1) | Were PARITY with a Rust defect: truncation after each separable pass darkened a uniform image by up to 2 and biased every image about one count dark. Fixed in Cyrius by ADR 003; values move 0 to +2 against Rust, never down. |
| `Affine::rotate` | transform.cyr:100 | PARITY (2.1.2) | The review counted this as parity with a documented NaN for \|θ\| ≥ 2^63. That guard had outlived the toolchain fault it worked around, so `rotate(1e20)` was NaN where Rust is finite. 2.1.2 delegates every finite angle; the item is not counted twice. |
| `Lut3d` Clone | `lut3d_clone` | PARITY (2.4.0) | Was MISSING. A deep copy, domain included. |
| `Affine`: Clone, Copy, PartialEq, Serialize, Deserialize | `affine_clone`, `affine_eq`, `affine_to/from_json` | PARITY ×3, IMPROVED ×2 (2.4.0) | Were MISSING. Setters for the six pub fields too. serde per ADR 008. |
| `Perspective`: Clone, Copy, PartialEq, Serialize, Deserialize | `perspective_clone`, `_eq`, `_set_m`, `_to/from_json` | PARITY ×3, IMPROVED ×2 (2.4.0) | Same. |
| `Lut3d::from_cube` | `lut3d_from_cube` (cube.cyr) | IMPROVED (2.4.0) | Was DIV-DOC ("not ported"). Rust's grammar and errors, and its correctly rounded f32 parse, exactly (124 exact-oracle cases, 46 of them double-rounding traps). Honours DOMAIN_MIN/MAX and reads LUT_3D_INPUT_RANGE, which Rust skipped or rejected (ADR 007). |

**Also found:**
- **Fixed in 2.2.0:**
  - `lut3d_new` had no overflow guard on `size³·12`.
  - `flood_fill`'s fixed `w·h` stack had no capacity check; it now grows.
- **Did not reproduce:** the reported bilinear over-read on a 1-pixel source with a
  non-finite inverse. Cyrius's `f64_to` saturates NaN to 0, as Rust's `as usize`
  does, so NaN coordinates read pixel 0 in both. The oracle's `affine_nonfinite` group
  (NaN and inf matrices, 1×1 and 20×20 sources) is byte-identical to Rust. The comment
  claiming the inverse "cannot produce NaN" was wrong, and is corrected.
- **Resolved in 2.4.0 (ADR 006):** `xorshift64_*` was public in the port but private
  in Rust. It is `_fk_xorshift64_*` now.

### composite, blend, icc, spectral, hwaccel

composite and blend are **byte-identical to Rust**, exhaustively for blend (all 12 modes
× all 65,536 channel pairs) and across every composite function, NaN and inf included.
ICC LUT outputs, table curves and the `srgb_v2_profile` bytes are bit-exact. Gamma,
para and matrix results differ by 1–2 ULP because `ganita_f64_pow` is not libm's `pow`.
That is counted as PARITY.

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `IccProfile::from_bytes` | icc.cyr:451 | PARITY (2.1.2) | Zero tags gave `RG_ERR_ALLOCATION` from 2.0.3 to 2.1.1; it is now InvalidFormat, as in Rust. |
| `IccLutProfile::from_bytes` | icc.cyr | PARITY (2.2.0) | Grid 1 is accepted (a constant LUT), as in Rust; grid 0 stays rejected. mft2 with 0 entries was fixed in 2.1.2. |
| `IccProfile`, `IccLutProfile` structs | `icc_profile_new`, `icc_lut_profile_new`, setters, `_clone` | PARITY ×2 (2.4.0) | Were PARTIAL (read-only). A constructor is Rust's struct literal, every pub field has a setter, and Clone is a deep copy. What Rust would panic on later (a null curve, a LUT whose length disagrees with its grid) is refused. The LUT profile keeps a length per input curve, as Rust's three Vecs do. |
| `ToneCurve` | `TC_*`, `tone_curve_table_at`, setters, `tone_curve_clone` | PARITY (2.4.0) | Was PARTIAL. Table entries and the gamma are readable and writable; Clone is deep. |
| serde on `BlendMode`, `ToneCurve` | serde.cyr | IMPROVED ×4 (2.4.0) | Were MISSING. BlendMode is the variant name ("ColorDodge"), ToneCurve externally tagged; ADR 008. |

**Also found:**
- `src/blend.cyr:143` (`one - ganita_f32_min(..)`, whose callee is declared `: i64`) was
  correct but fragile. It became `_bl_sub(one, …)` in 2.1.2, and all 65,536 ColorBurn
  inputs still match Rust.
- ICC returned `RG_ERR_PARSE` for some malformed input and `RG_ERR_INVALID_FORMAT` for
  the rest, while Rust uses InvalidFormat throughout. Since 2.2.0 the public parsers
  report INVALID_FORMAT for all of it, and the split survives only in the internal
  `_icc_*_parse` functions, for tests.
- prakash renamed three of the 20 spectral re-exports: `CIE_1931_2DEG` →
  `cie_1931_table()` and `VISIBLE_MIN/MAX_NM` →
  `spec_visible_min/max_nm()`. Since 2.4.0 `spectral.cyr` also provides the Rust names:
  `VISIBLE_MIN_NM` and `VISIBLE_MAX_NM` as values, and `CIE_1931_2DEG()` (a function,
  because prakash builds the table on first use) with `CIE_1931_2DEG_LEN`.
- ai-hwaccel 2.5.6 picks the first of equal-ranked devices, where Rust's `max_by_key`
  picks the last, and counts shared RAM once. So `hw_probe` can disagree with Rust on
  multi-GPU or unified-memory machines. This is upstream behaviour.

### gpu

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `GpuChain` + 12 methods | — | MISSING ×13 | Not ported, and missing from the CHANGELOG's "not ported" list. |
| `GpuContext::new_with_hwaccel` | — | MISSING | There is no gpu+hwaccel bundle profile. |
| `gpu_blend`, `gpu_noise_gaussian`, `gpu_gaussian_blur` | gpu_pipeline.cyr | PARTIAL ×3 | Stubs on every backend. Blur returns UNSUPPORTED even for radius 0, where Rust returns a clone. |
| `gpu_resize` | :449 | PARTIAL | Nearest only. A 0 dimension returns null + FORMAT; Rust returns Ok(0×0). |
| `adapter_name`, `device`, `queue`, `get_or_create_pipeline_1buf/_3buf` | gpu_context.cyr | PARTIAL ×5 | Pinned stub, or wgpu-only. The pipeline builders have no caller. |
| `gpu_fade`, `gpu_dissolve` | :314, :351 | SUSPECT | Packing rounds half UP (`floor(x+0.5)`), where WGSL `round` rounds half to even. fade(0.5) of 1/5/9/13 gives 1/3/5/7 against 0/2/4/6. |
| `gpu_flip_horizontal`, `gpu_flip_vertical`, `gpu_crop` | :388, :394, :410 | SUSPECT | 1D dispatch is capped at 65,535 groups, so 4096² fails with BUFFER_OP; Rust's 2D dispatch works. A degenerate crop returns 0×h, not 0×0. |
| `GpuError` Display | `gpu_error_message` | SUSPECT | FORMAT and UNSUPPORTED both render "no error". |

**Also found:**
- **Since 2.3.3, all fourteen operations are refused on every backend but native AMD**
  ([ADR 005](../adr/005-gpu-compute-runs-only-on-native-amd.md)). Rust ran them on any
  adapter wgpu found. mabda 4.2.0's NVIDIA backend executed ranga's SPIR-V as SASS
  and faulted the channel. The statuses in this table describe the AMD backend.
  `gpu_context_supports_compute` is a ranga addition with no Rust counterpart.
- Guards run in a different order from Rust: buffers are read before the null checks.
- `err_out` is set inconsistently on dimension mismatch.
- The cache vecs are never freed.
- `gpu_buffer_upload` leaks the GPU handle if a later allocation fails.

## Verification parity (tests, benchmarks, fuzz, examples)

| Rust 1.0.1 | Port today |
| --- | --- |
| 214 unit tests | 127 covered, 56 partial, 27 missing, 4 N/A |
| 123 integration tests (integration, edge_cases, proptest, photoshop_reference, visual_regression, spectral) | **All 123 since 2.3.0.** `integration.tcyr` has integration (23), photoshop_reference (12) and visual_regression (10) under Rust's names; `edge_cases.tcyr` has the 30 edge cases; `properties.tcyr` has the 31 properties; `spectral.tcyr` covers spectral's 17 (the 8 re-export tests added in 2.3.0). |
| 161 doctests (118 runnable, 110 with asserts, 42 `no_run`, 1 `ignore`) | ~85 covered, 16 partial, 10 missing (of the 110) |
| Property tests (31, proptest) | **All 31 since 2.3.0** (`properties.tcyr`): an edge grid plus 4,096 seeded random inputs each. |
| Photoshop golden vectors (7 modes on one fixed pixel pair) | **Pinned since 2.3.0,** within Rust's ±2. |
| 134 benchmarks (108 CPU, 26 GPU) | **All 134:** the 108 CPU benchmarks since 2.3.0 (`ranga.bcyr`, `spectral.bcyr`) and the 26 GPU ones since 2.3.1 (`gpu.bcyr`, run on the AMD dev host), under Rust's names and on its workloads. Five GPU benchmarks print `MISSING` until their operations are ported (2.5.x). |
| 8 fuzz targets | **All 8 since 2.3.0** (`fuzz/`), with invariants, run by CI under `--poison`. |
| Oracle outputs (no Rust counterpart) | **Captured since 2.3.0:** `oracle_golden.tcyr` checks 52 exact groups (53 until `bilateral` left the exact set in 2.3.2) (~2.1M values) against Rust 1.0.1's digests. |
| 11 examples | **0.** `docs/examples/` is empty, and the guides are Rust-era. |

Further detail on test coverage:
- **GPU coverage depends on the host.** 163 GPU assertions (23 + 49 + 83 + 8 across
  the four AMD-gated suites) run only when an AMD context opens. Since 2.3.0 a host
  without one prints `SKIPPED: N hardware assertions` per suite, and the checks that
  need no context (stubs, guards, null context, workgroup arithmetic) run
  everywhere. Assertion totals in `state.md` are **AMD-host** counts.
  - **NVIDIA has its own gated suite since 2.3.3.** `gpu_nvidia.tcyr` has 53
    assertions, and it passes on a GTX 1660 SUPER (`chew`). It checks:
    - the context and a buffer round trip;
    - that all eleven native operations are refused (ADR 005);
    - that no shader module was created.
  - Compute on NVIDIA waits on mabda's SPIR-V→SASS compiler (N9).
- **Some tests pin behaviour that contradicts Rust,** with no ADR:
  - the error strings;
  - bilateral's error codes;
  - the GPU stubs returning UNSUPPORTED where Rust returns a clone, InvalidFormat or Ok;
  - the BT.709 and NV12 inverses (see ADR 001 above).

## What the old audit got wrong

Listed so the corrections are not lost:

- **"Every u8 output path is byte-exact":** false. See findings 1 and 2.
- **Rows it called "did not reproduce":** `linear_to_srgb`, `rgbaf32_to_rgba8` and
  `cmyk_to_srgba`/`srgba_to_cmyk` were reproduced against the real crate. So was
  "levels byte-identical", and "affine_inverse identical" was also wrong.
- **"`Hsl.h` — 15 values, port more accurate":** h, s and l differ widely, and the
  differences reach bytes through `hue_shift`.
- **"serde surface is exactly four enums … codecs are hand-written":** it is 14 types,
  and no codec exists.
- **"Missing `FromStr`" and "`ColorSpace::Bt601` missing":** Rust has no `FromStr`, and
  `CS_BT601` exists.
- **"Entirety of `pipeline.rs` unported":** 11 of the 14 ops are ported. `GpuChain` is
  not, and 3 ops are stubs.
- **"ICC grid 0 makes Rust index wildly in release":** Rust panics, because Vec
  indexing is bounds-checked. Rejecting `< 2` also drops grid 1, which is valid.

## Re-running the oracle

`./scripts/oracle/run.sh` (since 2.2.0) builds a Rust driver against `rust-old/` and a
Cyrius probe against `src/`, runs both on identical inputs, and diffs the results:
2.1 million values in 41 groups. It exits non-zero on any difference that is not
accepted by an ADR within a bound. See
[`scripts/oracle/README.md`](../../scripts/oracle/README.md).

`rust-old/` is due to be deleted eventually (roadmap 2.x). Before that, the outputs
this oracle produces have to be captured as golden vectors in `tests/`, so that the
evidence outlives the crate it came from.
