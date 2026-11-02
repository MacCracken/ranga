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
| error, lib, pixel, color, convert | 88 | 48 | 3 | 15 | 7 | 4 | 11 |
| filter, transform, histogram | 73 | 51 | 1 | 4 | 0 | 11 | 6 |
| composite, blend, icc, spectral, hwaccel | 70 | 55 | 1 | 5 | 3 | 4 | 2 |
| gpu (context, buffer, shaders, pipeline) | 72 | 39 | 0 | 4 | 9 | 14 | 6 |
| **Total** | **303** | **193** | **5** | **28** | **19** | **33** | **25** |

An item is a pub fn (impl methods included), struct, enum, const, or trait impl/derive.
Each derive counts once per type. The M7 audit counted 318 under a looser
convention, so the two totals are not comparable line for line.

## Findings that matter most

1. **Several byte outputs are not bit-exact against Rust.** The old audit said "every u8
   output path is byte-exact", and that is false. The common cause is computing in f64
   and narrowing once at the end, where Rust works in f32 throughout.
   `src/color.cyr:195-200` names this exact pattern as wrong, for `srgb_to_linear`,
   which is bit-exact. Affected:
   - **Colour conversions:** `linear_to_srgb`, `lin_rgba_to_srgba`, `hsl_to_srgba`,
     `cmyk_to_srgba`, `rgbaf32_to_rgba8`.
   - **Filters:** `levels`, `bilateral`, and `hue_shift`, mostly through `srgba_to_hsl`.

   Most of the values involved are half-step inputs that sampled sweeps never hit. For
   example, f32 `0x3f010101` through `rgbaf32_to_rgba8` gives Rust 129 and the port
   128.
2. **`affine_inverse` reassociates the translation.** Rust computes
   `tx = (c*ty - d*tx) * inv`, and the port computes `-(ia*tx + ic*ty)`. For a plain
   scale+translate (`a=0.6, d=2.5, tx=ty=-1.9`), tx is `0x4009555555555555` in Rust
   and `…554` in the port, and an entire column of Nearest samples in
   `affine_transform` then comes from the wrong source column.
3. **Two ICC parse regressions came from the toolchain, not from ranga.** Since cyrius
   6.6.7, `fl_alloc(0)` and `fl_calloc(0)` return 0. As a result, a profile with zero
   tags, and an `mft2` LUT with zero input entries, now fail with `RG_ERR_ALLOCATION`.
   Rust returns InvalidFormat for the first and parses the second, using identity
   curves. The parser also rejects grid size 1, which Rust handles. ICC is the only
   module that parses untrusted input.
4. **The GPU "wgpu fallback" never runs.** ranga carries all 21 WGSL sources but
   dispatches none of them, and mabda 4.2.0's context-aware wgpu dispatch is a stub
   (see [architecture note 001](../architecture/001-mabda-native-gpu-constraints.md)).
   So on wgpu every `gpu_*` operation fails. On native backends:
   - 11 of the 14 operations run;
   - `gpu_blend`, `gpu_noise_gaussian` and `gpu_gaussian_blur` are stubs on every
     backend, and so is bilinear/bicubic `gpu_resize`;
   - `GpuChain` (13 items) is not ported at all, although `hwaccel.cyr`'s cost model
     assumes it.
5. **The serde surface is 14 types, and no codec exists.** The old audit said
   "four enums, hand-written codecs". In fact Rust derives Serialize/Deserialize on:
   - the 8 colour structs;
   - `PixelFormat`, `ColorSpace`, `BlendMode`, `ToneCurve`, `Affine` and `Perspective`.

   `src/` has no codec for any of them. Rust has **no `FromStr` anywhere**, so the old
   "missing FromStr" gaps were never gaps.

## Improved in Cyrius (keep)

- **`pixel_buffer_zeroed`**: fallible instead of panicking.
- **`pixel_buffer_row_offset` / `row_bytes`** (Rust `rows`/`rows_mut`): reject planar
  formats, and width 0 does not panic as Rust's `chunks_exact(0)` does.
- **`pixel_view_new`**: a size overflow returns an error code instead of aborting.
- **`brightness`**: a huge offset saturates. Rust's i16 wrap, which turns
  `(255, 0, 0)` at offset 1000, is not reproduced; byte-exact on 1,042 ordinary offsets.
- **`blend_row`**: an invalid mode returns `RG_ERR_INVALID_PARAMETER`.

## Deliberate divergences (documented; not to be re-litigated)

- **No payloads:** error payloads are dropped. `ranga_error_message` returns the format
  string with its placeholders removed, and `Result` becomes the `err_out` convention.
- **One read function per pair:** `PixelView` and `PixelViewMut` share one
  representation, and `data`/`data_mut` and `rows`/`rows_mut` merge, because Cyrius
  pointers carry no mutability.
- **Checked only:** panicking variants (`buffer_size`) collapse into the checked ones.
  `pixel_format_checked_buffer_size` is bounded by i64, not usize.
- **Overflow error code:** `pixel_buffer_new` reports a size overflow as
  `RG_ERR_ALLOCATION`, where Rust reports `BufferTooSmall`. ⚠ This has needed an ADR
  since M7, and none exists yet.
- **No i16 wrap in the YUV inverses** ([ADR 001](../adr/001-yuv-inverse-does-not-reproduce-rusts-i16-wrap.md)).
  ⚠ The ADR names only the BT.601 inverse, but `yuv420p_to_rgba_bt709`,
  `yuv420p_to_rgba_bt2020` and `nv12_to_rgba` diverge the same way, and their tests pin
  the diverged values. The ADR's "all five failed to reproduce" line is also wrong:
  three of them reproduce.
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
| `linear_to_srgb` | color.cyr:224 | SUSPECT | Computed in f64 with an f64 exponent of 1/2.4. 7 of 7 probed half-step inputs differ by 1. An emulation estimates ~219 affected f32 inputs in [0,1]. |
| `From<LinRgba> for Srgba` | `lin_rgba_to_srgba` :270 | SUSPECT | Alpha rounded in f64, and RGB goes through `linear_to_srgb`. |
| `From<Srgba> for Hsl` | `srgba_to_hsl` :398 | SUSPECT | On a 52³ grid, h differs by up to 22 ULP and s by up to 15 ULP. `hue_shift` inherits this. |
| `From<Hsl> for Srgba` | `hsl_to_srgba` :467 | SUSPECT | f64 path, and its `s < 1e-6` compare is f64 where Rust's is f32. Bytes differ. |
| `cmyk_to_srgb` | `cmyk_to_srgba` :519 | SUSPECT | Byte output computed in f64. 1,086 k values differ even with c=m=y=0. |
| `srgb_to_cmyk` | `srgba_to_cmyk` :529 | SUSPECT | f32 fields differ by up to 248 ULP. |
| `From<Oklab> for Oklch`, reverse | `oklab_to_oklch` :744, `oklch_to_oklab` :758 | SUSPECT | f64 trig. The header at :564 claims f32 throughout. |
| `color_temperature` | :781 | SUSPECT | f64 path. 4,205 of 5,572 kelvin values differ by up to 233 ULP. |
| `rgbaf32_to_rgba8` | convert.cyr:427 | SUSPECT | `v*255+0.5` computed in f64. 128 half-step f32 values in [0,1] round the other way. |
| `BufferPool::new` | pixel.cyr:398 | SUSPECT | Allocates `max_buffers*8` slots up front. `buffer_pool_new(1<<40)` returns 0, and acquire/release scan O(max). |
| `LinRgba`, `Hsl`, `CieXyz`, `CieLab`, `Cmyk`, `Oklab`, `Oklch` | color.cyr | PARTIAL ×7 | Getters only; Rust's pub fields are writable. The Hsl and Cmyk layouts are not in the module header. |
| `PixelBuffer::into_data` | — | MISSING | No way to take the bytevec and free only the header. |
| serde on `PixelFormat`, `ColorSpace` and the 8 colour structs | — | MISSING ×3 | No codec. |
| `PixelBuffer::new` overflow | pixel.cyr:154 | DIV-DOC | Needs the ADR flagged since M7. |
| `yuv420p_to_rgba_bt709` / `_bt2020`, `nv12_to_rgba` | convert.cyr | DIV-DOC | Diverge by ADR 001's i16 wrap, but the ADR does not name them. |

**Also found (no Rust counterpart, worth fixing):**
- `convert.cyr:103` ignores `_cv_y_row`'s return value. A failed 16-byte allocation
  hands back an uninitialised Y plane with `RG_ERR_NONE`.
- `buffer_pool_acquire` does not guard against a negative size.
- `bytevec.cyr`'s 19 `bv_*` functions sit in the unprefixed global namespace.

### filter, transform, histogram

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `levels` | filter_point.cyr:222 | SUSPECT | pow and tail computed in f64. `(black 0, white 0.8, γ1)` maps 106 to 132; Rust gives 133 (the f32 product is exactly 132.5). 17 of 9,900 settings differ. |
| `hue_shift` | filter_point.cyr:413 | SUSPECT | ±1 on 7–23% of pixels at 10°, 45° and −90°, mostly from `srgba_to_hsl`. At ±1e9° results diverge outright, because Rust's f32 add swallows the hue. |
| `bilateral` | filter_kernel.cyr:705 | SUSPECT | f64 coefficients and exp give ±1 on up to 67 of 1,564 bytes. σ ≤ 0 returns INVALID_PARAMETER where Rust returns `Other`; a NaN σ is rejected where Rust returns Ok. |
| `Affine::inverse` | transform.cyr:151 | SUSPECT | tx/ty reassociated: 8 of 15 inverses differ in bits, and samples flip. |
| `affine_transform` | transform.cyr:452 | SUSPECT | Inherits the inverse error. A model flips at least one Nearest sample in 561 of 20,000 random transforms. |
| `Affine::is_identity` | transform.cyr:171 | SUSPECT | A NaN field passes (returns 1); Rust returns false. |
| `Lut3d` Clone | — | MISSING | Heap-owning, and there is no `lut3d_clone`. |
| `Affine`: Clone, Copy, PartialEq, Serialize, Deserialize | — | MISSING ×5 | No `affine_clone` / `affine_eq`, and no codec. |
| `Perspective`: Clone, Copy, PartialEq, Serialize, Deserialize | — | MISSING ×5 | Same. |
| `Lut3d::from_cube` | — | DIV-DOC | Listed "not ported" with no reason. |

**Also found:**
- `lut3d_new` has no overflow guard on `size³·12`.
- `flood_fill`'s fixed w·h stack has no capacity check.
- Bilinear sampling of a 1-pixel-wide or 1-pixel-tall source can read up to 12 bytes past
  the buffer when the inverse is non-finite. `transform.cyr:446-449` claims it cannot
  produce NaN.
- `xorshift64_*` is public in the port but was private in Rust, which makes it a new API
  commitment.

### composite, blend, icc, spectral, hwaccel

composite and blend are **byte-identical to Rust**, exhaustively for blend (all 12 modes
× all 65,536 channel pairs) and across every composite function, NaN and inf included.
ICC LUT outputs, table curves and the `srgb_v2_profile` bytes are bit-exact. Gamma,
para and matrix results differ by 1–2 ULP because `ganita_f64_pow` is not libm's `pow`.
That is counted as PARITY.

| Rust item | Port | Status | What is wrong or missing |
| --- | --- | --- | --- |
| `IccProfile::from_bytes` | icc.cyr:451 | SUSPECT | Zero tags gives `RG_ERR_ALLOCATION` (`fl_calloc(0)`, :177); Rust gives InvalidFormat. |
| `IccLutProfile::from_bytes` | icc.cyr:920 | SUSPECT | mft2 with 0 input entries fails at `fl_alloc(0)` (:796), where Rust uses identity curves. Grid 1 is rejected by `grid < 2` (:862), where Rust handles it. |
| `IccProfile`, `IccLutProfile` structs | getters | PARTIAL ×2 | Read-only; there is no way to build a profile. |
| `ToneCurve` | `TC_*` | PARTIAL | Table entries have no accessor (raw `load64(t+16)` only). |
| serde on `BlendMode`, `ToneCurve` | — | MISSING ×4 | No codec. BlendMode's wire form is the variant name ("ColorDodge"), not `blend_mode_name`. |

**Also found:**
- `src/blend.cyr:143` (`one - ganita_f32_min(..)`, whose callee is declared `: i64`) is
  correct at 6.7.18; all 65,536 ColorBurn inputs match. It is fragile, though, and
  `_bl_sub(one, …)` removes both the warning and the risk.
- ICC returns `RG_ERR_PARSE` for some malformed input and `RG_ERR_INVALID_FORMAT` for
  the rest; Rust uses InvalidFormat throughout.
- prakash renamed three of the 20 spectral re-exports, so they are not callable under
  their Rust names: `CIE_1931_2DEG` → `cie_1931_table()` and `VISIBLE_MIN/MAX_NM` →
  `spec_visible_min/max_nm()`.
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
- Guards run in a different order from Rust: buffers are read before the null checks.
- `err_out` is set inconsistently on dimension mismatch.
- The cache vecs are never freed.
- `gpu_buffer_upload` leaks the GPU handle if a later allocation fails.

## Verification parity (tests, benchmarks, fuzz, examples)

| Rust 1.0.1 | Port today |
| --- | --- |
| 214 unit tests | 127 covered, 56 partial, 27 missing, 4 N/A |
| 123 integration tests (integration, edge_cases, proptest, photoshop_reference, visual_regression, spectral) | 34 covered, 59 partial, 30 missing |
| 161 doctests (118 runnable, 110 with asserts, 42 `no_run`, 1 `ignore`) | ~85 covered, 16 partial, 10 missing (of the 110) |
| Property tests (31, proptest) | **None.** 3 are covered by exhaustive 256-value sweeps; 25 are reduced to fixed examples; 3 are missing. |
| Photoshop golden vectors (7 modes on one fixed pixel pair) | **None pinned.** |
| 134 benchmarks (108 CPU, 26 GPU) | 35. 33 match by name, 1 is renamed, 1 runs a different workload. 101 are missing. |
| 8 fuzz targets | **0.** `tests/ranga.fcyr` is the scaffold stub. |
| 11 examples | **0.** `docs/examples/` is empty, and the guides are Rust-era. |

Further detail on test coverage:
- **GPU coverage depends on the host.** 152 GPU assertion sites only run when an AMD
  context opens, so CI (no GPU) and any non-AMD host skip almost all of
  `gpu_pipeline` and `gpu_kernels`. The 1,921-assertion figure is an **AMD-host**
  count, and the NVIDIA backend has no test at all.
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

The measurements above were taken with throwaway harnesses. To repeat them:

1. Build `rust-old/` with `CARGO_TARGET_DIR` pointed at a scratch directory, using
   `cargo --release` (it builds in seconds).
2. Write a small Rust driver and a Cyrius probe that print the same calls over the same
   inputs.
3. Diff the two outputs.

Making this a checked-in tool is the first item of 2.2.x in the roadmap. `rust-old/`
is due to be deleted eventually, and this oracle, plus the golden vectors captured from
it, is what has to outlive it.
