# ranga — Roadmap

> Milestone plan through the Cyrius port. State lives in [`state.md`](state.md);
> this file is the sequencing — what ships, in what order, against what
> dependency gates. The detailed capability analysis, module map, and
> internalizing list live in [`cyrius-port-plan.md`](cyrius-port-plan.md).

## Versioning

`VERSION` stays **1.0.1** for the whole port and bumps to **2.0.0** at M8, when
the Cyrius tree reaches parity with what Rust 1.0.1 shipped. `distlib` stamping
`v1.0.1` in the meantime is correct — claiming 2.0.0 earlier would advertise a
library that does not exist yet.

This continues ranga's existing numbering with a major bump rather than resetting
to 0.x, matching every AGNOS port (vidya 1.x→2.0.0, prakash 1.2.0→2.0.0, mabda
1.0.0→2.0.0, ai-hwaccel 1.2.0→2.0.0, hisab 1.4.0→2.0.0) and prakash's practice of
deferring the bump to the final milestone.

_This file replaces the generic `v0.1.0 → v1.0` template the `cyrius port`
scaffold wrote, which assumed a fresh project rather than a port._

## 2.0.0 criteria

- [ ] Rust → Cyrius surface parity verified (function-level diff against `rust-old/`)
- [ ] Test coverage at parity — 214 unit + 123 integration + 161 doctest assertions ported
- [ ] All 134 benchmarks recorded in [`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md)
- [ ] All 8 fuzz targets ported to `tests/ranga.fcyr`
- [ ] At least one downstream consumer green (rasa, tazama, aethersafta, soorat)
- [ ] CHANGELOG entry for 2.0.0 naming the language change as the breaking change
- [ ] Security audit pass (`docs/audit/YYYY-MM-DD-audit.md`)
- [ ] `VERSION` → 2.0.0, distlib drift gate green

## Milestones

Full module map and difficulty ratings: [`cyrius-port-plan.md`](cyrius-port-plan.md) §6.

### M0 — Port scaffold — ✅ shipped 2026-08-17

- Benchmark history lifted out before `_port_move` swept `benches/`
- `cyrius port` scaffold landed; 13,958 Rust lines moved to `rust-old/`
- Manifest converted from the scaffold's binary shape to library shape
  (`${file:VERSION}`, `[lib] modules`, `distlib` → `dist/ranga.cyr`)
- `.gitignore` repaired — the port's move broke every root-anchored Rust rule,
  leaving 7.6 GB of `rust-old/target/` unignored
- Doc-tree per [first-party-documentation.md](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-documentation.md)

### M1 — Foundation — in progress (183 assertions green)

Settles the raw-byte-offset accessor pattern that every later module follows
(plan §2). Automation learnings captured in
[`port-mechanics.md`](port-mechanics.md).

- [x] `src/error.cyr` — 8 error codes + `is_err`/`is_ok`, plus the Rust-semantics
      shims `_rg_pow`/`_rg_exp`/`_rg_cbrt`/`_rg_sin`/`_rg_cos`. 37 assertions.
- [x] `src/bytevec.cyr` — the `Vec<u8>` the stdlib lacks (plan §3 item 2), on
      `fl_alloc`/`fl_free` so a 1080p frame is one mmap each way.
- [x] `src/pixel.cyr` — `PixelFormat` (6 formats, overflow-checked sizing),
      `PixelBuffer`, `PixelView`/`PixelViewMut`, `BufferPool` (best-fit).
      102 assertions.
- [x] `src/color.cyr` — **complete.** sRGB transfer, `Srgba`/`LinRgba`/`CieXyz`/
      `CieLab`/`Hsl`/`Cmyk`/`Oklab`/`Oklch`, sRGB↔XYZ, XYZ↔Lab, Display P3,
      `color_temperature`, Delta-E CIE76/CIE94/CIEDE2000. 108 assertions,
      including the Sharma et al. CIEDE2000 reference pair at 2.0425.
- [ ] `src/constants.cyr` — shared EPS and matrices, once a second module needs them.

Two silent-corruption traps hit here and written up in
[`port-mechanics.md`](port-mechanics.md): untyped f32 operands emit integer
multiplies, and decimal float literals past ~9 significant digits parse wrong
(filed upstream as
`cyrius/docs/development/issues/2026-08-17-decimal-float-literal-silent-precision-loss.md`).

### M2 — Pixel ops — ✅ complete (936 assertions green across the port)

- [x] `src/composite.cyr` — all 13 fns: premultiply/unpremultiply, apply_mask,
      dissolve/fade/wipe, fill_solid/fill_checkerboard, three gradients,
      composite_at + composite_at_argb. 159 assertions, 13 doctests.
- [x] `src/histogram.cyr` — all 5 fns plus the `Hist` record standing in for
      `Vec<f64>`. 150 assertions, 5 doctests.
- [x] `src/transform.cyr` — `Affine` (+ translate/scale/rotate/then/apply/inverse/
      is_identity), `ScaleFilter`, crop, resize (nearest/bilinear/bicubic), both
      flips, `affine_transform`, `Perspective` (+ `from_quad`'s 8×8 Gaussian
      elimination with partial pivoting), `perspective_transform`.
      148 assertions; 16/16 mutants caught.
- [x] `src/filter_point.cyr` — 13 per-pixel fns: brightness, contrast,
      saturation, levels, curves, grayscale, invert, hue_shift, color_balance,
      vibrance, channel_mixer, threshold, auto_white_balance. 112 assertions;
      17/18 mutants caught.
- [x] `src/filter_kernel.cyr` — 10 neighbourhood/generative fns: gaussian_blur,
      box_blur, unsharp_mask, median (Huang), bilateral, vignette, apply_lut3d
      (+ `Lut3d` trilinear), noise_gaussian, noise_salt_pepper, flood_fill
      (scanline). 120 assertions; 17/19 mutants caught.

`filter.rs` (2007 lines, 23 fns) was split into a point half and a kernel half,
matching prakash's split of `ray/mod.rs` — Cyrius `src/` is flat and one file
that size is unwieldy.

First half fanned out to agents (two porters + two adversarial verifiers), which
found a use-after-free in the composite tests and four histogram assertions that
passed against deliberately broken code. The second half was ported inline after
three consecutive API 529 failures made the agent path unreliable.

### M3 — SIMD modules — scalar parity ✅ (1,119 assertions green across the port)

- [x] `src/blend.cyr` — `BlendMode` (12 modes), `blend_pixel`,
      `blend_pixel_argb`, `blend_row_normal` (+ argb variant), `blend_row`.
      87 assertions; 14/18 mutants caught, the other 4 provably unobservable.
- [x] `src/convert.cyr` — all 14 fns: BT.601/709/2020 in both directions,
      NV12 both ways, and the six interleaved/f32 format converters.
      96 assertions; 17/17 mutants caught.
- [x] `src/simd_u8.cyr` — plan §3 items **3 and 5**: hand-encoded saturating u8
      add/sub (`paddusb`/`psubusb`), packed unsigned min/max (`pminub`/`pmaxub`),
      and a masked lane select. Every encoding produced by `llvm-mc`, every
      primitive differentially tested against its own scalar fallback over an
      exhaustive input sweep. 34 assertions.
- [x] `brightness` vectorised end-to-end — 4 pixels per iteration with an alpha
      mask-and-restore (paddusb cannot skip a lane), plus a scalar tail. Proven
      to agree with the scalar path on a deliberately non-multiple-of-4 buffer.
- [ ] Remaining kernels — `_cv_y_row` (pmaddwd), `_cv_yuv_row_to_rgba`,
      `blend_row_normal`, `grayscale`. All four dispatch points are in place and
      the primitives + verified ABI are now proven, so these are additive.
      Plan §3 item 4 (256-bit AVX2) sits behind them.

⚠ **A correct SIMD path is invisible to output testing.** Mutating the group
count so fewer pixels take the vector path leaves every assertion passing,
because the scalar fallback agrees by construction. These tests prove the paths
AGREE; proving which one *ran* needs a benchmark or instrumentation. Budget for
that when the remaining kernels land — otherwise a silently-disabled SIMD path
looks exactly like a working one.

Not fanned out to agents, per `port-mechanics.md` — and the shift-direction bug
below is exactly why.

⚠ **Shift direction is per-call-site, and Cyrius spells it backwards from most
languages**: `>>` is LOGICAL, `>>>` is ARITHMETIC. Which is correct depends on
the signedness of the Rust operand — xorshift shifts a `u64` so `>>` is right,
while the YUV inverse shifts an `i16` whose `(U-128)` is genuinely negative, so
`>>>` is required. Using `>>` there zero-filled a negative product into a large
positive and blue came back 255 instead of 0. Neither answer generalises; check
the Rust operand type at every shift.

### M4 — ICC

`icc` + the 5 parametric curve types.

### M5 — External deps

`spectral` (prakash, verified 18/18 covered), `hwaccel` (ai-hwaccel v2.3.16).

### M6 — GPU

`gpu_shaders`, `gpu_buffer`, `gpu_context`, `gpu_pipeline` against mabda's
**native** backends; wgpu is the fallback and leaves mabda's tree at v5.1.

### M-perf — first performance pass ✅

Benchmark harness built (`tests/ranga.bcyr`, 35 CPU benchmarks with names
matching the Rust criterion series) and
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md) filled in.

**Median 10.6x slower than Rust, range 3.8x-79.9x.** Two fixes landed from
reading the spread rather than the median:

- **Flat hot loops.** Cyrius leaves general inlining off (`_INLINE_OK`), so
  helper calls in a per-pixel path are real calls. Inlining the blur inner loops
  gave **2.1x**.
- **`pixel_buffer_uninit`.** `fl_calloc` zeroes byte-at-a-time over already-zero
  mmap pages — 369x slower than the allocation itself. Filed upstream; the
  consumer workaround is 1.35x-1.75x on the memcpy-bound ops and should be
  reverted if the stdlib fix lands.

Still open: the 10x-20x band is Rust SIMD vs Cyrius scalar and needs the
remaining `asm { }` kernels. `yuv420p_to_rgba` at 44.2x should get its chroma
row pointers hoisted before any assembly is written.

### M7 — Parity

Surface count against `rust-old/`, full benchmark sweep on a quiesced machine,
`benchmarks-rust-v-cyrius.md` filled in.

### M8 — Release

distlib, docs, CHANGELOG, CI gates, **`VERSION` → 2.0.0**.

## Out of scope (for 2.0.0)

- Async GPU readback — mabda has no non-blocking buffer map
- WASM / WebGPU — blocked on the Cyrius WASM backend
- Retiring `rust-old/` — kept 1–3 releases past 2.0.0, then deleted per the
  AGNOS standard (only after Cyrius has equal or better coverage and benchmarks)
