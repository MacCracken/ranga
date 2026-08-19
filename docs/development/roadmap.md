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

### M6 — GPU ✅

`src/gpu_pipeline.cyr` — the fourteen public operations and the wiring that
makes the GPU feature callable. **99 assertions, HW-verified on AMD Cezanne.**
Eleven run natively; `gpu_blend`, `gpu_noise_gaussian`, `gpu_gaussian_blur` and
bilinear `gpu_resize` report `RG_GPU_ERR_UNSUPPORTED` and have complete WGSL for
the wgpu path.

**A 36-agent audit of the new layer found six real defects in code that passed
56 assertions on the first run.** Two would have bitten a consumer directly:

1. **`gpu_gaussian_blur` returned an error code from a slot that carries a
   buffer.** Rust's is `-> Result<PixelBuffer>` — it PRODUCES, where blend and
   noise mutate in place. Returning `RG_ERR_OTHER` (-8) meant a caller written
   to the documented convention saw non-zero, took the success branch, and
   `pixel_buffer_data(-8)` dereferenced it. Now returns 0, like `gpu_resize`'s
   unsupported path, and the test pins it.
2. **The shader-module cache was process-global.** A compiled module is staged
   in a GTT buffer object on the context's own DRM fd, and mabda's dispatch
   checks only that the handle is non-null — so a second context would dispatch
   the first one's modules, and after the first was freed, against a closed fd.
   Rust kept its caches on `GpuContext`; the cache now lives there too and is
   released with the device. A test creates a second context after freeing the
   first and asserts it rebuilds.
3. **`gpu_crop` diverged from BOTH Rust and ranga's own CPU `crop`.** Rust takes
   `(left, top, right, bottom)` and CLAMPS; ranga's `transform.cyr` crop takes
   the same four and clamps the same way. This took `(sx, sy, dw, dh)` and hard-
   rejected out-of-range input, so moving a call between CPU and GPU crop would
   have silently changed what the arguments meant.
4. **`gpu_dissolve` and `gpu_wipe` compared pixel COUNT, not shape.** Rust
   guards width and height separately. 4x1 against 2x2 passed, and wipe then
   reflowed the source into a different raster — a silently wrong composite
   rather than `DimensionMismatch`.
5. **The two error spaces overlap and were being mixed.** `RG_GPU_ERR_FORMAT` is
   -4 and so is `RG_ERR_UNSUPPORTED_CONVERSION`; `RG_GPU_ERR_BUFFER_OP` is -3
   and so is `RG_ERR_BUFFER_TOO_SMALL`. Guard paths returned `err_out`'s value
   straight into a slot documented as `RG_ERR_*`, handing callers a plausible
   code from the wrong space. Everything now goes through `gpu_error_to_ranga`.
6. **A null context was dereferenced** where every sibling in the feature
   returns a coded `RG_GPU_ERR_NO_ADAPTER`.

**And five assertions that could not fail.** Saturation was tested only at
factor 0 — the one value where it collapses to bare luminance, making the
expected constants bit-identical to grayscale's and the factor entirely
unconstrained. Brightness/contrast only at the identity, which says nothing
about which param is which. Wipe only at 0 and 1, neither of which exercises the
line. Crop with `top = 0` throughout, leaving `src_y` wiring untested. And every
dispatch under 256 pixels, so `_gp_groups`' div_ceil was never asked a real
question — a wrapper hardcoding one workgroup would have passed the entire
suite. All five now have inputs that distinguish right from wrong.

⚠ The lesson is the same one M7 taught: **a green suite is evidence about the
tests, not the code.** These 56 assertions passed on the first run against code
with a dereference-a-negative-integer bug in it.

### M6 — GPU (superseded; foundation notes below)

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

#### ✅ The five per-pixel kernels

`src/gpu_kernels.cyr` (347 lines), `tests/gpu_kernels.tcyr` (26 assertions).
`invert`, `grayscale`, `fade`, `brightness_contrast`, `saturation` — all five
**HW-verified on AMD Cezanne against an f32 oracle, byte-exact on every pixel**.

⚠ **THE BRANCH SHAPE IS NOT INTERCHANGEABLE, AND THE FAILURE IS SILENT.** The
WGSL guard is `if idx >= count { return; }` followed by the body. Emitted that
way — return inside the selection, body after the merge label — the module
compiles to a **non-zero shader handle** and dispatches with **rc = 0**, and
writes nothing whatsoever. There is no diagnostic anywhere. mabda's own
`programs/native_spirv_divergent_if_e2e.cyr` is the reference shape: the body
goes in the THEN block, that block ends with `OpBranch` to the merge, and the
merge block holds the function's only `OpReturn`. So the guard is inverted to
`if (count > idx) { body } return`. The comparison is `count > idx` rather than
`idx < count` because mabda's binop map covers UGreaterThan and
UGreaterThanEqual but **not ULessThan**.

⚠ **No `OpBitcast` in the supported set.** A params word that must be read both
as a `count` and as an f32 needs two differently-typed arrays over the *same*
memory, so the params buffer is bound twice — three bindings, two buffers.

⚠ **The oracle must be f32, not f64.** Computed in f64, the expected values
agree on most inputs and then disagree on the ones near a rounding boundary —
brightness/contrast on two of five pixels — which reads exactly like a kernel
bug and is not one.

⚠ **grayscale disagrees with ranga's CPU path, and did in Rust too.** The CPU
uses BT.601 integer weights (77/150/29 >> 8); the GPU shader uses BT.709 floats
(0.2126/0.7152/0.0722). On (200, 100, 50) that is 0x7C against 0x76. Both are
preserved as-is — reconciling either side would change output the Rust line
produced. The test asserts the gap explicitly so it reads as inherited rather
than accidental.

#### ✅ Five more: dissolve, wipe, crop, flip_horizontal, flip_vertical

**Ten of fourteen operations now run on the native backend**, all HW-verified on
AMD Cezanne against exact expected output.

⚠ **Two more things the compiler silently rejects.** Both surfaced as
`gpu_shader_module_create_spirv` returning 0 with no diagnostic, and both were
isolated by bisecting a working kernel rather than guessed:

- **A selection nested inside a selection does not compile.** Two SEQUENTIAL
  selections do, and a 2D workgroup using `gid.y` does. So the WGSL's 2D bounds
  test `if x >= w || y >= h { return; }` cannot be written as two nested ifs.
- **`OpSelect` does not compile either**, despite mabda's lowerer having a case
  for it — it is dispatched but not selectable on GFX9. That rules out the
  obvious workaround of reducing two booleans to 1/0 and multiplying.

There is no OpLogicalAnd in the binop map either, so conjunction is built from
arithmetic that does survive. For `b1 > v1 && b2 > v2` on unsigned values below
2^31, form `d = (b1 - 1 - v1) | (b2 - 1 - v2)`: each term underflows and sets
bit 31 exactly when its condition fails, so one comparison against `0x80000000`
tests both. `wipe` uses this.

**The geometry kernels moved to 1D indexing.** crop, flip_horizontal and
flip_vertical dispatch flat and recover x and y with UMod and UDiv, so they need
ONE bound instead of two and avoid the conjunction entirely. The WGSL's 16x16
workgroup was for cache locality on wgpu and the fallback keeps it.

**Mutation testing: 11 mutants, 11 killed** — after three rounds of fixing the
TESTS, not the code. `dissolve` mixes all four channels including alpha, and the
first data set had alpha 0 on both sides, so dropping the alpha channel passed.
The guards compare strictly, and a one-past write lands in 64 KiB alignment
slack unless an explicit sentinel sits past `count`. And the sentinel has to
differ between source and destination: with the same value in both, a one-past
*copy* writes an identical word and stays invisible.

#### ✅ The WGSL fallback — all 21 sources

`src/gpu_shaders.cyr` (877 lines), `tests/gpu_shaders.tcyr` (59 assertions).
Every WGSL source from the Rust line, for the wgpu backend.

**Machine-transcribed, not retyped.** A script extracted each source verbatim
from `rust-old/src/gpu/shaders.rs` and escaped it for Cyrius, one WGSL line per
`str_builder_add_cstr`, so a divergence between the two lines cannot come from a
typo. The four WGSL lines too long for the 120-column limit are emitted in
several calls; the concatenation is byte-identical, only the emission is split.
The tests assert byte LENGTHS against the Rust originals, which is what catches
a dropped or doubled line.

**Twenty-one sources, not fourteen.** `gpu/mod.rs` re-exports fourteen
operations, but shaders.rs also carries LUT3D, HUE_SHIFT, COLOR_BALANCE,
RGBA_TO_Y_BT601 and the two blur passes, which the pipeline drives internally.
Dropping the ones without a public wrapper would have silently narrowed what the
fallback can do.

⚠ **No hardware test is possible for this module**, and the tests say so rather
than implying otherwise. mabda's wgpu backend requires the consumer to compile
in `object;` mode and be entered from a C launcher that builds a wgpu function
table — a `.tcyr` run has no launcher. So the WGSL is checked structurally, and
`gpu_context_from_launcher` is the entry point for a consumer that has one.

**Every native kernel has a fallback, and every native-blocked one is complete
here.** The suite asserts both directions, so adding a native kernel without its
WGSL fails the build. `resize_bilinear`, `blend` and both blur passes — the ones
the native path cannot take — are present in full. Blur's WGSL genuinely loops,
which is exactly why `OpLoopMerge` blocks it natively; the test asserts that too.

#### ✅ resize_nearest — and where the native path runs out

**Eleven of fourteen operations run on the native backend**, all HW-verified.
The remaining three are blocked by concrete mabda limits, filed in
**mabda's own repo** at
`docs/development/issues/2026-08-19-native-spirv-compile-limits.md` — it is a
mabda defect, and cyrius does not own mabda.

`resize_nearest` landed. The WGSL clamps with `min(gx * src_w / dst_w, src_w - 1u)`;
that min is **provably redundant** once the bounds guard has run, since
`gx <= dst_w - 1` forces the integer division below `src_w`. Dropped rather than
reproduced, because there is no unsigned min in the ext-inst set and faking one
through floats would add rounding to an exact integer path.

⚠ **`OpFDiv` does not compile.** Integer `UDiv`/`UMod` are fine. Isolated by
bisection: a kernel doing nothing but one float division returns 0, while the
same kernel with four texel loads, twelve lerps and a full unpack/pack round
trip compiles. Both reciprocals are now computed host-side and passed as params.

⚠ **`resize_bilinear` emits a valid module that mabda cannot lower.** This is
NOT an emitter defect and the test suite says so explicitly:
`spirv_validate_stream` accepts it, every table rebuilds, the id bound is 201 —
and a deliberately-grown kernel compiles at **229 ids** and fails at **254**, so
size is not the obstacle. Every individual construct compiles in isolation. What
is left is live-value pressure from four independent texel addresses; holding
the texels packed and extracting one channel at a time (peak liveness ~20 → ~12)
did not lower it enough. The test pins the current state — valid module, under
the cap, does not compile — so the suite reports the day mabda can take it
rather than leaving a silently dead kernel in the tree.

**Still blocked, and why:** `resize_bilinear` and `blend` (13 modes, larger
still) on register pressure; `gaussian_blur` separately on `OpLoopMerge` and
`OpPhi` being rejected outright.

#### Per-pixel kernels

**Mutation testing: 15 mutants, 15 killed.** Three needed better inputs rather
than better assertions: the clamps inside `pack_rgba` are unreachable from any
kernel that clamps in its own maths, so `fade` at 2.0 (mid-grey lands on exactly
256, whose low byte is 0 — the brightest input would come out BLACK) and at -1.0
drive them; and the bounds guard's off-by-one is invisible without a sentinel
element past `count`, since the buffer is 64 KiB-aligned and a one-past write
lands in slack.

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

### M7 — Parity ✅

Full surface audit against `rust-old/`, module by module, every claimed gap
adversarially re-checked. Report:
[`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md).

**318 Rust public items: 222 ported, 22 deliberate omissions, 51 missing** — and
**28 of the 51 are one file**, `gpu/pipeline.rs`. Everything underneath it is
ported (context, buffers, SPIR-V emitter, eleven kernels, all twenty-one WGSL
sources), so that gap is wiring rather than maths. Most of the rest is `#derive`
with no Cyrius analogue and mostly no cost; the ones that do cost something are
string-facing, chiefly the absent `FromStr` inverses.

**The audit earned its keep on four defects in code that was already green:**

1. **`pixel_view_new` accepted OVERSIZED buffers.** It tested `len < want`;
   Rust tests `data.len() != expected`. The port had silently widened the
   contract — passing a whole frame where a tile was meant produced a view
   reading the wrong rows instead of a diagnostic. Fixed to exact equality,
   error code corrected to `DimensionMismatch`, both directions now asserted.
2. **`RangaError` had no Display surface at all** — eight opaque negative
   integers, while `gpu_error_message` in the GPU module already had the idiom.
   Added `ranga_error_message`, with a test that no two codes share a string.
3. **`gpu_context_adapter_name` was a stub reporting itself as ported** — it
   returns the literal `"unknown"` and nothing ever assigns the field, because
   mabda exposes no live adapter query. Now documented, given a
   `has_adapter_name` companion, and PINNED by a test asserting the stub state
   so the suite fails the day mabda can report a real one.
4. **Two comments asserted things that were false.** `pixel.cyr` claimed
   Mut-only accessors that do not exist; `gpu_spirv.cyr` claimed mabda lowers
   "exactly five" GLSL ops when it lowers eight — and one of the three unlisted
   is `FClamp`, which several kernels here spell the long way.

⚠ **A passing suite does not mean a faithful port.** Every one of those four was
in code with green tests and clean lint. Three were found only by reading the
Rust beside the Cyrius, and the fourth only because a verifier was told to
assume the audit was lying.


Surface count against `rust-old/`, full benchmark sweep on a quiesced machine,
`benchmarks-rust-v-cyrius.md` filled in.

### M8 — Release

distlib, docs, CHANGELOG, CI gates, **`VERSION` → 2.0.0**.

## Out of scope (for 2.0.0)

- Async GPU readback — mabda has no non-blocking buffer map
- WASM / WebGPU — blocked on the Cyrius WASM backend
- Retiring `rust-old/` — kept 1–3 releases past 2.0.0, then deleted per the
  AGNOS standard (only after Cyrius has equal or better coverage and benchmarks)
