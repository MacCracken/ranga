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

### M2 — Pixel ops — in progress (556 assertions green across the port)

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
- [ ] `src/filter_point.cyr` — 13 per-pixel functions
- [ ] `src/filter_kernel.cyr` — 10 neighbourhood/generative functions

First fan-out (two port agents + two adversarial verifiers). The verifiers
earned their keep: a use-after-free in the composite tests, and four histogram
assertions that passed against deliberately broken code. Both modules were
*correct* — every gap was test coverage, and the new assertions passed on the
first run against the existing implementations.

### M3 — SIMD modules

`convert`, `blend`. Scalar path first for correctness parity, then the `asm { }`
kernels — SIMD ships in 2.0.0, not deferred.

### M4 — ICC

`icc` + the 5 parametric curve types.

### M5 — External deps

`spectral` (prakash, verified 18/18 covered), `hwaccel` (ai-hwaccel v2.3.16).

### M6 — GPU

`gpu_shaders`, `gpu_buffer`, `gpu_context`, `gpu_pipeline` against mabda's
**native** backends; wgpu is the fallback and leaves mabda's tree at v5.1.

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
