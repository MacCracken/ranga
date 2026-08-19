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
- [x] `_sx_luma_row_sse2` — pmaddwd, 2 px/iter, the whole loop inside one
      `asm { }` block with branch displacements taken from `as`/`objdump`.
      Differentially tested against an independent scalar reference with
      DIFFERENT values per lane, across all three coefficient standards.
      Wired into `_cv_y_row`, `grayscale` and `threshold` — one kernel, four
      call sites.
- [x] `invert` — vectorised with the pass-1 primitives, no new assembly:
      `255 - c` on a byte IS a saturating subtract.
- [x] `_sx_blend_row_sse2` — Porter-Duff source-over, 2 px/iter.
      **46.70 µs → 1.87 µs (25×), now 0.5× Rust** — i.e. faster than the Rust
      original, which is the first benchmark where that is true. Proven
      byte-identical to the scalar path over 257 px × 5 opacities, and confirmed
      to actually execute by forcing the scalar path (49.8 µs).
- [ ] `_cv_yuv_row_to_rgba` — got a 1.43× loop hoist but no kernel yet.
      Plan §3 item 4 (256-bit AVX2) sits behind it.

**int→f32 probe: no cheap win, closed.** `f32_from(f64_from(n))` is cvtsi2sd +
cvtsd2ss; there is no `cvtsi2ss` path in the compiler and mabda's
`native_int_to_f32_bits` is the same two-step. Measured in the blur inner shape:
`load8` alone 1 ns, the full convert-and-multiply-add 2 ns — so eliminating the
conversion entirely would at best halve the blur, leaving it ~10× off. The
remaining blur gap is Rust **auto-vectorising** a loop Cyrius runs scalar, not
conversion overhead. Closing it needs a SIMD blur kernel, not a cheaper cast.

⚠ **A correct SIMD path is invisible to output testing.** Mutating the group
count so fewer pixels take the vector path leaves every assertion passing,
because the scalar fallback agrees by construction. These tests prove the paths
AGREE; proving which one *ran* needs a benchmark.

**The technique that settles it:** force the scalar path (set `pairs = 0`) and
re-measure. `blend_row_normal` went 1.87 µs → 49.8 µs, which is proof the kernel
executes. Do this once per kernel — it costs a minute and is the only evidence
that a vector path is not silently dead.

Not fanned out to agents, per `port-mechanics.md` — and the shift-direction bug
below is exactly why.

⚠ **Shift direction is per-call-site, and Cyrius spells it backwards from most
languages**: `>>` is LOGICAL, `>>>` is ARITHMETIC. Which is correct depends on
the signedness of the Rust operand — xorshift shifts a `u64` so `>>` is right,
while the YUV inverse shifts an `i16` whose `(U-128)` is genuinely negative, so
`>>>` is required. Using `>>` there zero-filled a negative product into a large
positive and blue came back 255 instead of 0. Neither answer generalises; check
the Rust operand type at every shift.

### M4 — ICC ✅

`src/icc.cyr` (1,018 lines), `tests/icc.tcyr` (169 assertions). **Both** profile
parsers are ported, not just the matrix/TRC one:

- **`IccProfile`** — the matrix/TRC form. Tag table, `curv` tags (count 0 =
  linear, 1 = u8Fixed8 gamma, else a u16 table), `para` tags (all five function
  types), the column-major primaries matrix, and `srgb_v2_profile`.
- **`IccLutProfile`** — the `A2B0` LUT form, `mft2` (16-bit) and `mft1` (8-bit),
  with per-channel input curves and trilinear interpolation through a 3D CLUT.
  This was nearly missed: it is a second, independent parser sharing only the
  header and tag-table code with the first, and the roadmap's carried-forward
  list had it filed as "coverage currently minimal" rather than as unported
  surface. rust-old's doc comment advertises `mAB ` (v4) support that its match
  arms do not implement; the port preserves the actual behaviour and rejects
  `mAB `, with a test pinning that so it reads as deliberate.

**This is the one module whose input is untrusted** — every other module
consumes buffers ranga itself produced. Every multi-byte read is big-endian and
bounds-checked, the tag count is capped at 1024, and the LUT grid is capped at
64 (an unchecked 255 would ask for 397 MB from a single byte).

**Mutation testing: 38 mutants, 34 killed.** The pass was worth more here than
anywhere else so far — the first round killed only 12 of 17 and the survivors
were not near-misses but *whole unexecuted paths*: the generated sRGB profile
only ever emits `curv` with count 1, so five of the six curve types and the
entire table branch had no coverage at all despite the suite passing. Two
further rounds found an assertion tolerance loose enough to hide a 13% error,
and three fixtures whose parameters made the mutant and the original agree by
coincidence (`e == f`, `in_entries == out_entries`, a value that clamped to the
same bound either way). The four documented survivors are recorded in the module
header; all are clamps whose wrong value is multiplied by zero — real
protections against a heap over-read, invisible to any output assertion.

### M5 — External deps ✅

`src/spectral.cyr` (58 assertions) over **prakash 2.2.3**, `src/hwaccel.cyr`
(48 assertions) over **ai-hwaccel 2.3.17**.

**Both are OPTIONAL, matching the Rust line.** `spectral` and `hwaccel` were
non-default cargo features (`default = ["simd"]`), so the port gates them the
same way: `[features]` + `optional = true` deps, each with its own `[lib.X]`
bundle profile. Consumers of the core bundle never clone prakash or ai-hwaccel —
transitive feature tables are not parsed, so an optional dep stays inert
downstream.

**The re-exports needed no code.** `rust-old/src/spectral.rs` is mostly a
`pub use` block naming 18 prakash items. Cyrius has a flat namespace and links
the bundle whole, so all 18 are already callable; re-declaring them would be a
collision, not a re-export. What survived porting is the six convenience
functions and the two `From` conversions — the only part that was ever real
code.

**Two allocators meet at the spectral seam.** prakash allocates with the bump
allocator and never frees; ranga's `CieXyz` uses `fl_alloc`/`fl_free`. Every
conversion copies rather than aliasing, even though the layouts are identical
(both 24 bytes, x@0 y@8 z@16) — an alias would work right up until the first
free. The tests assert the layout agreement *and* the pointer distinctness, so
the claim in the comment is checked rather than trusted.

**`Option<u32>` became a sentinel.** ai-hwaccel encodes "not reported" as
`PROFILE_NONE` (-1). That distinction is load-bearing: a GPU idling at 0% and a
GPU whose driver exposes no counter are different facts, and the offload policy
branches on which one it has. The port carries the sentinel through rather than
flattening it, with `hw_report_has_utilization` / `_has_temperature` as the
`is_some()` equivalents.

**The Rust hwaccel tests asserted nothing** — `let _ = report.has_gpu;` three
times over, because a CI box has no GPU and the assertions were unwritable. That
left the five-branch offload decision completely untested. The port splits the
policy out of the probe (`hw_should_use_gpu_report` takes a report instead of
reading hardware), so every branch is now reachable from a synthetic report and
the crossover, VRAM and utilisation rules are all pinned.

⚠ **A dep bundle silently overrode three of ranga's own functions.** Including
`lib/hisab.cyr` — which prakash's bundle references for `num_fft` — produced
three duplicate-function warnings, and Cyrius resolves duplicates as LAST
DEFINITION WINS. hisab's four-argument `premultiply_alpha(r, g, b, a)` replaced
ranga's one-argument `premultiply_alpha(buf)`; every existing call site would
have passed a PixelBuffer where four doubles were expected. hisab is only needed
by prakash's `wave_pattern`, which this bridge never calls, so it is not
included and `num_fft` links as an unreachable reference. **Read the duplicate
warnings from `cyrius distlib` — they are the only signal.**

**Mutation testing: 18 mutants, 17 killed.** The survivor is equivalent —
`has_utilization` is redundant while the sentinel is negative, since `-1 > 90`
is false either way — and is kept because it states the intent Rust spelled with
`if let Some`. One mutant exposed a genuinely under-specified helper:
`_hw_bytes_to_mb(-1)` returns 0 with or without its negative guard, because
integer division truncates toward zero, so the test now pins a negative larger
than a megabyte.

**Filed upstream:** profile `.deps` sidecars are written EMPTY at 6.5.27 for any
project following the documented "source files only need project includes"
convention — the include-scan has nothing to find. `dist/ranga.deps` is
authoritative for all three bundles.

### M6 — GPU (foundation landed; compute path blocked on a decision)

`src/gpu_context.cyr` + `src/gpu_buffer.cyr` (43 assertions, **round-trip
HW-verified on AMD Cezanne**). `gpu_shaders` and `gpu_pipeline` — the 14
operations and `GpuChain` — are NOT started; see the shader-format fork below.

**mabda is at 4.0.9, not the 1.0.0 the plan assumed.** Three backends behind one
public API: native AMD (amdgpu/GFX9), native NVIDIA (nouveau/SM75), and wgpu.

⚠ **"The public API does not change across backends" applies to a NARROWER set
of entry points than it sounds.** The `gpu_ctx_device`/`gpu_ctx_queue`
accessors, and every helper taking a `device` — `create_storage_buffer`,
`read_buffer`, `compute_pipeline_new` — are **wgpu-path only**. The native
context reuses the same struct with different field meanings: offset +16 holds a
GEM buffer-object handle where wgpu holds the device. The backend-agnostic
surface is the one taking the CONTEXT — `gpu_buffer_create`/`_write`/`_read`/
`_release`, `gpu_compute_dispatch` — which dispatches through the backend's slot
table. The first draft of this module used the wgpu helpers and would have
called through an uninitialised function table on native hardware.

⚠ **Every mabda context constructor returns a TAGGED RESULT.** Dereferencing it
as a context does not fault — it yields plausible garbage. The first version
reported a context created successfully with `backend=unknown`, and segfaulted
only on the *next* run. `payload()` after `is_err_result()` is mandatory, and
the test suite asserts the backend name is not "unknown" specifically to catch
a regression here.

⚠ **mabda's `compute_pipeline_new` does not fit ranga's layouts** — it builds a
fixed BGL (read-write at 0, read-only at 1..N, no uniform) where all three of
ranga's end in a uniform and two want read-only first. Rust used
`ComputePipeline::with_layout`; the Cyrius port has no equivalent, so
`_gc_pipeline_with_layout` assembles one from mabda's public `bglb_*` builder.

**THE FORK — shader format.** `_backend_native_shader_module_create` accepts
**only SPIR-V**; WGSL and even pre-compiled GFX9 return 0. `gpu_shader_module_create`
forwards GFX9 for native AMD, which that handler then rejects. So:

- **wgpu** takes ranga's 1,047 lines of WGSL as-is, but requires the consumer to
  compile in `object;` mode and be entered from a C launcher building a wgpu
  function table. mabda deprecated AMD-on-wgpu at v4.0.1.
- **native** works on this hardware (mabda's own e2e programs are HW-verified on
  Cezanne) but needs SPIR-V, and **there is no WGSL→SPIR-V compiler anywhere in
  the stack** — mabda's e2e programs hand-emit SPIR-V word by word in Cyrius.

**DECIDED: native via mabda, wgpu as the automatic fallback.** wgpu is a
backend *inside* mabda, not an alternative to it — ranga targets mabda's API and
mabda picks the backend. So the shader layer must carry BOTH forms, selected on
`backend_kind` at runtime: SPIR-V for the native backends, WGSL for the wgpu
fallback. The WGSL half is a near-verbatim transcription of the Rust; the
SPIR-V half is new work, specified below.

#### What the native lowerer accepts — the binding constraint

ranga can only emit SPIR-V that `_spirv_lower_one_instr` turns into GFX9.
Enumerated from the dispatch, this is the whole set:

- **arithmetic** IAdd, FAdd, ISub, FSub, IMul, FMul, UDiv, SDiv, FDiv, UMod,
  SRem, SMod
- **bitwise** And, Or, Xor, shifts
- **compare** FOrd{Equal,NotEqual,LessThan,GreaterThan,…}, ULessThan, SLessThan
- **convert** ConvertSToF, ConvertUToF, ConvertFToS, FConvert
- **memory** Load, Store, AccessChain
- **composite** CompositeConstruct, CompositeExtract
- **control** Label, Branch, BranchConditional, SelectionMerge, Select, Return
- **ExtInst** GLSL450 (sqrt, floor, …)

⚠ **`OpLoopMerge` and `OpPhi` both return `LOWER_ERR_CONTROL_FLOW`.** Straight-line
code and if/else only — **no loops**. Thirteen of ranga's fourteen operations are
per-pixel and fit; `gpu_gaussian_blur` convolves over a dynamic radius and does
not. Blur either unrolls to a fixed set of radii, or takes the wgpu fallback,
or stays on the CPU — that is a smaller decision to make when it is reached,
not a blocker for the other thirteen.

⚠ **f64 is gated off on native** (`MABDA_NATIVE_F64 = 0` until mabda's F.7): a
SPIR-V module using f64 is rejected up front. ranga's GPU kernels are f32
throughout, so this costs nothing today — but it rules out reusing the f64
colour paths from `color.cyr` verbatim.

#### Emitter design

Hand-writing SPIR-V per kernel is not viable: mabda's own e2e example spends
**125 lines of raw `store32`** on `data[id.x] = id.x`, the simplest kernel there
is. ranga needs `src/gpu_spirv.cyr` — a builder that owns a word buffer and an
id counter and exposes `spv_type_int` / `spv_type_vec` / `spv_constant` (all
interned), `spv_emit(op, …)`, and the standard compute preamble
(OpCapability Shader, OpMemoryModel Logical GLSL450, OpEntryPoint GLCompute,
OpExecutionMode LocalSize, the Block/ArrayStride/Binding/DescriptorSet
decorations `_spirv_check_array_strides` requires). Each kernel is then a
function against that builder rather than a wall of offsets.

`spirv_validate_stream` and `spirv_find_entry_point` are already in mabda's
surface and give the emitter a free self-check: every kernel can assert its own
module parses before it is ever handed to the GPU.

#### ✅ The emitter landed — and the whole native path is proven end to end

`src/gpu_spirv.cyr` (413 lines), `tests/gpu_spirv.tcyr` (53 assertions).
**HW-verified on AMD Cezanne**: ranga emits a 560-byte SPIR-V module, mabda
compiles it SPIR-V → GFX9, it dispatches, and all 256 lanes come back correct.
The remaining thirteen kernels are now ordinary code against this builder rather
than walls of `store32`.

**Four buffers, concatenated at finish.** A SPIR-V module has a fixed section
order — capabilities, memory model, entry point, execution modes, then ALL
annotations, then ALL types/constants/globals, then function bodies. But a
kernel discovers the types it needs *while* emitting its body, so the builder
keeps the sections apart and joins them in `spv_finish`.

⚠ **mabda's parser does not enforce section order.** Swapping the annotation and
type sections passes `spirv_validate_stream`, rebuilds every table, compiles to
GFX9, and *runs correctly on the GPU* — the mutation for it survived an
otherwise complete suite. It is still an invalid module and a stricter consumer
may reject it, so `tests/gpu_spirv.tcyr` walks the instruction stream itself and
asserts the section boundaries directly rather than trusting the lenient parser.

**Mutation testing: 12 mutants, 12 killed.** The ones worth naming, because each
is a whole-module corruption that produces a *plausible* file: a wordcount that
omits its own header word (every following instruction reinterpreted), an id
bound off by one (the highest id reads as out of range), a literal string
dropping its NUL padding word (`"main"` is TWO words, not one — four chars plus
a mandatory terminator), and each of the four interning keys being ignored,
which silently declares a duplicate type where SPIR-V demands exactly one.

### M6 — GPU (original plan)

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

**Second pass — first vector kernels landed:**

- `rgba_to_yuv420p` **2.10x** — new `_sx_luma_row_sse2` (pmaddwd, 2 px/iter),
  wired into `_cv_y_row` so all three colour standards benefit.
- `invert` **2.33x** — no new assembly; `255 - c` is a saturating subtract, so
  the primitives from the first pass covered it.
- `yuv420p_to_rgba` **1.40x** — pure loop hoist, chroma row pointers out of the
  inner loop.

**Third pass — one kernel, four call sites.** `_sx_luma_row_sse2` turned out to
serve `grayscale`, `threshold` and all three YUV forward converters, because
they share the same integer luma shape:

- `threshold` **1.62x**, `grayscale` **1.50x** — luma vectorised, scatter left
  scalar.

**Median is now 8.6x (from 10.6x), worst case 46.0x (from 79.9x).**

Still open, in priority order:

1. **`crop` 46.0x and `flip_vertical` 23.9x** — still `memcpy`-bound and still
   the worst ratios. The real fix is upstream (`fl_calloc`); the consumer
   workaround has already been applied and this is what remains.
2. **The blurs at 21.4x** are now the largest ABSOLUTE cost (~396 ms each).
   Rust's blur is *also* scalar f32 at default features, so this is pure
   language overhead, and the likely culprit is that
   `f32_from(f64_from(load8(..)))` is two conversions per sample where Rust's
   `as f32` is one. Worth investigating a direct int->f32 path before more
   assembly.
3. **`blend_row_normal` 12.9x** — needs a per-pixel alpha broadcast (pshuflw)
   plus a vector div255, materially more complex than the luma kernel.
4. Plan §3 item 4 (256-bit AVX2) sits behind all of the above.

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
