# Ranga → Cyrius Port Plan

Porting ranga from Rust (1.0.1, the final Rust release) to **Cyrius**. Ships as
**ranga 2.0.0**, matching every prior AGNOS port: vidya 1.x→2.0.0, prakash
1.2.0→2.0.0, mabda 1.0.0→2.0.0, ai-hwaccel 1.2.0→2.0.0, hisab 1.4.0→2.0.0.
None reset to 0.x.

`prakash/docs/development/cyrius-port-plan.md` is the canonical library recipe
and the closest match to ranga. **It is stale on two major points** — it predates
Cyrius v6.4.0 and states "no generics" and that float arithmetic is function-call
form. Both are wrong now: generics have been on by default since v6.4.0
(`CYRIUS_MONOMORPH`), and bare `+ - * /` operators work on `f32`- and
`f64`-typed values. Do not take that document's language claims at face value.

**Pinned to Cyrius 6.5.27** (`cyrius = "6.5.27"` in `cyrius.cyml`). Everything
below was verified against that release, several points empirically rather than
from docs.

Status: **M0 done, M1 done, M2 in progress.** Live milestone state is in
[`roadmap.md`](roadmap.md); the automation learnings are in
[`port-mechanics.md`](port-mechanics.md).

---

## 1. What the port is

13,958 lines across 18 files: 8 enums, 14 structs, ~150 public functions, 21
public WGSL shader constants, 3 associated consts. Test surface is 214 in-module
`#[test]` fns, 123 integration tests, 161 doctests, 134 named benchmarks across
10 criterion suites, 8 libfuzzer targets.

Cyrius is a sovereign C-level systems language. It has more type machinery than
prakash's plan implies — monomorphized generics with real specialized layouts,
generic enums, dot field access, `private`/`public`, element-typed arrays,
bounds-checked slices — but the typing is **edge-typed, not structure-typed**.
That distinction defines this port and is covered in §2.

Unlike prakash (dense f64 math), ranga is **u8-dominated with f32 color math**.
That asymmetry is the good news: the u8 paths — blend, composite, filter kernels,
histogram — are integer arithmetic that maps onto native Cyrius operators
directly, including saturating (`+|`) and checked (`+?`) forms. The f32 color
science is where the work is.

---

## 2. The shape-defining constraint: struct fields have no float or narrow-unsigned type

Locals, parameters, and function returns can carry `f64`/`f32`/struct/slice types
that drive codegen. **Struct fields cannot.** Field sids are `i8`/`i16`/`i32`/
`i64`/struct/`Str`/`Vec` only — there is no `f32` or `f64` field type, and no
`u8`/`u16`/`u32` field type.

So ranga's core types cannot be declared as they stand:

```rust
pub struct Srgba  { pub r: u8,  pub g: u8,  pub b: u8,  pub a: u8 }   // no u8 field type
pub struct LinRgba { pub r: f32, pub g: f32, pub b: f32, pub a: f32 } // no f32 field type
```

Two consequences, both silent-corruption class:

1. `c.r * c.g` on a conceptually-float field emits an **integer multiply**. No
   diagnostic.
2. `#derive(accessors)` emits `load64`/`store64` **unconditionally**, so it is
   wrong for every narrow field. Do not use it for pixel or color types.

**Decision: store colors as raw byte offsets with hand-written, width-correct
accessors.** `load8`/`store8` for u8 channels; `load32` + `f32_to`/`f32_from` for
f32 channels. `i32` fields are safe for f32 bit patterns if you always round-trip
through the conversion builtins and never apply an arithmetic operator directly.

This is mechanical but touches every color type, so it is settled once, up front,
in `color.cyr`, and every later module follows the pattern.

---

## 3. What must be internalized

The answer to "what needs a local shim until a lib exists." 47 candidate gaps
were adversarially re-verified against the stdlib; **36 turned out to be
buildable from existing primitives and 4 were false alarms.** Re-checked against
**6.5.29**, a further one closed upstream. These 6 remain.

> ⚠ **Every item below was RE-TESTED on 2026-08-19, not re-read.** Two claims
> elsewhere in this port turned out to have drifted — the GLSL.std.450 op count
> and the enum-derive symptom — in both cases because a recorded sentence was
> quoted rather than checked. All six still hold, but three had supporting text
> that had gone stale, and item 3 has become an active trap: the stdlib now
> offers something that *looks* like the missing primitive and has the wrong
> semantics.

> **Closed upstream — no longer ranga's problem.** The original list opened with an
> ~80-line `ranga_f32_*` shim, because `grep -c f32 lib/math.cyr lib/ganita.cyr`
> returned `0` and `0`. Filed as an issue scoped to ganita; **shipped as ganita
> 1.1.0 in cycc 6.5.24** — 23 functions (`ganita_f32_abs`/`neg`/`sign`/`min`/`max`/
> `clamp`/`lerp`/`floor`/`ceil`/`round`/`trunc`/`sqrt`/`cbrt`/`pow`/`exp`/`exp2`/
> `ln`/`log2`/`sin`/`cos`/`atan`/`atan2`/`hypot`). ranga consumes `ganita_f32_*`
> directly. Verified on 6.5.27 and still present at the 6.5.29 pin: `sqrt(9)=3`, `cbrt(27)=3`, `min(9,4)=4`,
> `pow(2,10)=1024`. Note there is **no f64 cbrt** anywhere in the tree —
> `ganita_f32_cbrt` is the only one, and it does the required sign split
> (`pow` goes through `exp(y·ln x)`, so a bare pow returns garbage for negatives).

| # | Gap | Size | Notes |
|---|---|---|---|
| 1 | ~~**`ranga_f64_powf`**~~ **CLOSED at the 6.5.31 pin** | — | ✅ ganita **1.1.4** (folded into cyrius 6.5.30) added zero-base, zero-exponent and negative-base handling to `ganita_f64_pow`, and a `cbrt(±0) = ±0` guard to `ganita_f32_cbrt`. 1.1.4 goes FURTHER than this shim did — it returns a real value for a negative base with an integral exponent, which is Rust's `powf` behaviour, where the old path gave NaN. `_rg_pow` / `_rg_cbrt` / `_rg_f32_cbrt` are now thin delegations, kept only so parity-sensitive call sites stay named and a future regression has one place to patch. Integral exponents land within 2 ULP rather than exact (the general path is `exp(y·ln x)`; Rust's `powf` is exact) — measured at 1e-15 on `pow(2,3)`, and **zero of 256 bytes differ** across a full gamma-2.0 tone-curve sweep, so it cannot move a u8. Pinned by a test. Original text: | `ganita_f64_pow` does not match Rust's `powf` for `pow(0.0, y)` and `pow(negative, integer)` — and `ganita_f32_pow` delegates straight to it, so the f32 tier inherits the mismatch. ranga's ICC tone curves ([icc.rs:99](../../src/icc.rs), :836) and gamma paths depend on the Rust semantics. Wrap once, route all pow calls through it, rather than sprinkling `if base == 0.0` at call sites. |
| 2 | **Byte-vec (`Vec<u8>`)** | ~120 lines | `lib/vec.cyr` is i64-slots only — still hardcoded `data + idx * 8` at 6.5.29. `PixelBuffer`'s entire backing store needs a byte-granular vector. Build on `fl_alloc`/`fl_free`, whose >4 KB path is a direct `mmap`/`munmap` — an 8 MB RGBA8 frame becomes one syscall each way, which is a genuinely good fit for `BufferPool`. |
| 3 | **Saturating u8 vector add/sub** | ~40 lines | ⚠ **Re-verified 2026-08-19 at the 6.5.29 pin — still needed, but the reason has narrowed and there is now a TRAP.** `lib/simd.cyr` has since grown `i8v16_add`/`i8v16_sub`, so a reader may think this item closed. It has not: those lower to `paddb`/`psubb`, which **wrap**. Pixel math needs `paddusb`/`psubusb`, which **saturate** — 250 + 10 must be 255, not 4. Using the stdlib pair here would corrupt every clipped highlight silently. Wrap `paddusb`/`psubusb` (x86) and `uqadd`/`uqsub` (NEON) in an `asm { }` raw-byte block — precedented in-tree by `_fhm_probe16` ([lib/hashmap_fast.cyr:66-110](file:///home/macro/Repos/cyrius/lib/hashmap_fast.cyr)). Encodings verified: `paddusb xmm0,xmm1` = `66 0F DC C1`. |
| 4 | **256-bit integer SIMD (AVX2)** | ~60 lines | ⚠ **Re-verified 2026-08-19 at the 6.5.29 pin — still needed; supporting text was stale.** `lib/simd.cyr` now carries `simd_has_avx2`, `simd_has_fma` and a full 256-bit **f32** tier (`f32v8_*`), plus typed 128-bit integer wrappers (`i8v16`, `i16v8`, `i32v4`, `i64v2`). What it still does NOT have is 256-bit **integer** anything — the widest integer vector is 128-bit. `blend_row_normal_avx2` needs `_mm256_cvtepu8_epi16`, `_mm256_mullo_epi16`, etc. Port the kernel as one `asm { }` block with hand-encoded VEX bytes, mirroring `_sha_ni_compress_one` ([lib/sigil.cyr:4871](file:///home/macro/Repos/cyrius/lib/sigil.cyr)). x86-only; scalar fallback elsewhere. |
| 5 | **Vector min/max and lane select** | ~50 lines | ⚠ **Re-verified 2026-08-19 at the 6.5.29 pin — min/max/compare/select claim HOLDS (grep for min\|max\|cmp\|blend\|select across `lib/simd.cyr` returns zero), but the f32 sentence was partly stale.** f64 gained `f64v2_div`/`f64v4_div` and `f64v2_sqrt`/`f64v4_sqrt`; **f32 did not** — packed f32 really is `add`/`sub`/`mul`/`fmadd`/`dot` only, so the Newton-Raphson advice below applies to the f32 tier and is unnecessary for f64. Divide via Newton-Raphson on `fmadd` (~1.5–2× a real `divps`, far better than dropping to scalar). **See the ⚠ below before implementing min/max.** |
| 6 | **serde codecs for 4 enums** — ⚠ **the blocker lifted at 6.5.31** | ~10 lines each | cyrius **6.5.31** added `#derive(Serialize)`/`#derive(Deserialize)` codec generation for ENUMS; the derive now emits a two-argument `<name>_to_json(sb, value)` where it previously emitted nothing at all. ranga has never had serde at all, so this is available parity work (Rust derived Serialize on PixelFormat/BlendMode/ColorSpace) rather than a fix to existing code — and the calling convention was not validated here. Original text: | `#[derive(Serialize)]` on `PixelFormat`, `BlendMode`, `ColorSpace`, and the one payload enum. **Footgun: `#derive(Serialize)` works on a Cyrius `struct` but NOT on an `enum`. Re-checked at the 6.5.29 pin: above an enum it compiles rc=0 with no diagnostic and emits NO CODEC AT ALL — the generated `<name>_to_json` is undefined at link time. (Earlier plan text said "misnamed, crashing codec"; the symptom changed, the conclusion did not.)** Hand-write them, following the `device_class_to_str` pattern ([lib/yukti.cyr:640](file:///home/macro/Repos/cyrius/lib/yukti.cyr)). |

> ⚠ **Do not implement vector min/max as a bare unsigned compare on the raw f32
> pattern.** This plan originally proposed exactly that, on the grounds that for
> non-negative finite f32 the raw pattern orders identically to an unsigned
> integer. That is true, and it is precisely the trap — IEEE-754 is
> **sign-magnitude, not two's complement**, so among two negatives the order
> *reverses*, and any negative compares HIGH against every positive. Pixel data is
> non-negative, so a naive version passes every plausible test and breaks the
> first time a consumer subtracts — and ranga's `Difference` and `Subtract` blend
> modes do exactly that before clamping. Apply the standard monotone-key transform
> (flip all bits when negative, else set the sign bit) so one signed compare is
> correct across the whole range. This was caught by the ganita 1.1.0 triage;
> `_f32_key` in `lib/ganita.cyr` is the reference implementation to mirror.

Total internalized surface: **~300 lines**, of which ~150 is `asm { }` blocks
confined to two SIMD kernels. None of it is novel numerics — items 3–5 are
transcriptions of instructions ranga already emits via `std::arch`.

**Nothing here needs to become a separate library.** All six are ranga-local
shims. Items 1 and 5 are the plausible candidates for upstreaming into
`ganita`/`simd` later — item 5 especially, since the ganita f32 tier just
established the correct scalar precedent; items 2–4 and 6 are either too
ranga-specific or belong in the compiler rather than a library.

---

## 4. Capability map — verified present

| ranga need | Cyrius source | Symbols |
| --- | --- | --- |
| f64 arithmetic + transcendentals | operators + `math` + `ganita` | bare `+ - * / < > <= >= == !=` on `: f64` values; `f64_sqrt/floor/ceil/round/sin/cos/exp/ln`, `f64_clamp/min/max/lerp`, `ganita_f64_pow/atan2/asin/acos/hypot` |
| scalar f32 arithmetic | operators (v6.4.56) | bare `+ - * /` on `: f32` values via `EMIT_F32_BINOP` (real `addss`/`subss`/`mulss`/`divss` + NaN-correct `ucomiss` ladder, all four backends); `f32_from`/`f32_to`. **There is no callable `f32_add`/`f32_sub`/`f32_mul`** — arithmetic dispatches only from a typed binding. Annotated params (`fn f(a: f32, b: f32)`) carry the type into the body; unannotated params arrive as untyped bit patterns and silently emit *integer* ops. Annotate every f32 param. |
| f32 math library | **ganita 1.1.0** (cycc 6.5.24) | `ganita_f32_abs/neg/sign/min/max/clamp/lerp/floor/ceil/round/trunc/sqrt/cbrt/pow/exp/exp2/ln/log2/sin/cos/atan/atan2/hypot` — 23 fns |
| SIMD (float) | `lib/simd.cyr` | `f32v4`/`f32v8` + `f32v_fmadd`, `f32v_dot`, `f32v8_fma`; runtime AVX2 gating via `simd_has_avx2()` |
| SIMD (integer) | `lib/simd.cyr` | `i8v16`/`i16v8`/`i32v4`/`i64v2` (⚠ re-verified 2026-08-19: **signed only** — there are no u8v16/u16v8/u32v4/u64v2 types, the earlier "+ unsigned variants" was wrong); `iv_add`/`iv_sub`/`iv_mul` (i16/i32 widths), `iv_dp8` (u8·i8→i32 widening dot — maps onto the YUV and grayscale-luminance kernels) |
| bounds-checked `&[u8]` | `lib/slice.cyr` | `var s: [u8]`, `s[i]` → `_slice_idx_get_W`, `slice_unchecked_get_W` for hot loops, `slice_copy_bytes` |
| large buffers + pool | `lib/freelist.cyr` | `fl_alloc`/`fl_free`/`fl_calloc`; >4 KB is direct mmap/munmap |
| raw pixel access | builtins | `load8/16/32/64`, `store8/16/32/64`, `memcpy`/`memset` |
| GPU | `lib/mabda.cyr` (26,682 lines) | `wgpu_shader_source_wgsl`, `compute_pipeline_new`, `compute_dispatch`, `gpu_buffer_create/write/read`, `shader_cache_get_or_compile`, `pipeline_cache_get/set`, ping-pong buffers |
| spectral | **prakash** (Cyrius, M2 complete) | `xyz_new`, `xyz_d65_white`, `xyz_d50_white`, `rgb_new`, `cie_1931_table()`, `Spd` — **18/18 of ranga's re-exports covered** |
| hwaccel | **ai-hwaccel** v2.3.16 | `registry_detect()`, `registry_detect_with()`, `registry_detect_no_exec()` |
| JSON (was serde) | `bayan` | `bayan_json_v_obj_new/_obj_set/_str_new/_float_new`, `_v_build`, `_parse` |
| logging (was tracing) | `sakshi` | `sakshi_info/_warn/_error/_debug/_trace`, `sakshi_log_kv` |
| tests / benches / fuzz | `test`, `assert`, `bench` | `.tcyr` / `.bcyr` / `.fcyr` |

Both AGNOS dependencies are **fully served**. ranga's prakash surface is tiny
(`Xyz`, `Rgb`, `Xyz::new`, `D50_WHITE`, `D65_WHITE`) and prakash's M2 Spectral
milestone — the one ranga needs — is complete. ai-hwaccel's surface is two
symbols. Neither blocks the port.

---

## 5. GPU

`lib/mabda.cyr` is a near-complete match — it and ranga's Rust `mabda` crate are
two ports of the same library, and mabda's integration guide names ranga as a
target consumer. **There is no architectural blocker here.**

**Backend model.** mabda runs three backends behind one public API: native AMD
(amdgpu DRM / GFX9 / PM4, shipped v3.0), native NVIDIA (nouveau DRM / SM75,
shipped v4.0), and wgpu-native. **wgpu is the transitional fallback, not the
target** — AMD-on-wgpu was deprecated at v4.0.1, NVIDIA-on-wgpu retires at v5.0,
and the whole wgpu + C launcher path leaves the tree at v5.1. Per mabda's
roadmap, "the public API does not change across backends; consumer code stays
byte-identical," and `compute_dispatch(device, queue, cp, bind_group, dims_xyz)`
has identical signatures on all three.

**Initialization is a plain library call on the native path.**
`gpu_context_new_native()` (AMD) and `gpu_context_new_native_nvidia()` (NVIDIA)
are called directly, with the backend selected at compile time via
`MABDA_BACKEND_KIND`. The C launcher and `mabda_main(fn_table, preinit)` entry
belong only to the wgpu path — and even that is not an obstacle, since Cyrius
ships C FFI in `lib/cffi.cyr`, `lib/dynlib.cyr`, and `lib/fdlopen.cyr`. So
`GpuContext::new()` ports as a normal function. Target the native entry points
and treat wgpu as the fallback it is.

**Shaders.** mabda carries its own shader compilation pipeline — `spirv_lower.cyr`
plus a full MIR→GFX9 chain (`gfx9_isel.cyr` → register allocation →
`gfx9_encode.cyr`), and `backend_nvidia_sass.cyr` on the NVIDIA side. ranga's 21
WGSL constants survive as-is on the wgpu path (largest is `BLEND_ALL` at 2,271 B
against a 1 MiB limit), and ADR 003's packed-u32 design is orthogonal to the
language. Confirm the WGSL→native lowering status against the mabda tag pinned at
port time and pick the shader input format from what that backend accepts.

Two real but small items remain:

1. No async/non-blocking buffer map, so `GpuBuffer::download_async` has no
   equivalent. Synchronous readback exists and covers the rest.
2. `compute_pipeline_new` hardcodes an all-storage bind layout; ranga's four
   uniform-terminated layouts need a ~35-line builder from `bglb_*` primitives,
   all of which are exposed.

---

## 6. Module map & sequencing

Cyrius `src/` is flat; ranga's `gpu/` submodules become prefixed files. Ship in
dependency order; each module is ported → `fmt`/`lint`/`vet` clean → its tests
ported → `cyrius test` green → next. Never port tests after the fact.

| # | Cyrius module | From | LOC | Difficulty |
|---|---|---|---|---|
| 0 | `error.cyr` | `error.rs` | 33 | mechanical — integer codes, needed everywhere |
| 0 | `constants.cyr` | scattered | — | shared EPS, matrices, white points |
| 1 | `pixel.cyr` | `pixel.rs` | 783 | mechanical — **needs the byte-vec (§3.3) first** |
| 1 | `color.cyr` | `color.rs` | 1423 | moderate — settles the §2 accessor pattern; needs f32 shim + cbrt |
| 2 | `convert.cyr` | `convert.rs` | 1272 | hard — SSE2/AVX2/NEON integer kernels; **scalar fallback is complete, so behavior parity is mechanical and only performance parity is hard** |
| 2 | `blend.cyr` | `blend.rs` | 839 | hard — same; needs internalized items 4–6 |
| 3 | `composite.cyr` | `composite.rs` | 791 | mechanical — integer/byte arithmetic |
| 3 | `histogram.cyr` | `histogram.rs` | 364 | mechanical |
| 4 | `transform.cyr` | `transform.rs` | 930 | moderate — 8×8 Gaussian elimination in `Perspective::from_quad` |
| 4 | `filter_point.cyr` | `filter.rs` (13 point fns) | ~900 | moderate — brightness/contrast/saturation/levels/curves/grayscale/invert/hue_shift/color_balance/vibrance/channel_mixer/threshold/auto_white_balance |
| 4 | `filter_kernel.cyr` | `filter.rs` (10 kernel fns) | ~1100 | moderate — blur/unsharp/median(Huang)/bilateral/vignette/lut3d/noise/flood_fill; needs `_rg_exp` |
| 5 | `icc.cyr` | `icc.rs` | 1142 | moderate — big-endian tag-table parsing, 5 parametric curve types, needs `ranga_f64_powf` |
| 6 | `spectral.cyr` | `spectral.rs` | 102 | thin re-export over prakash — **verified 18/18 covered** |
| 6 | `hwaccel.cyr` | `hwaccel.rs` | 164 | thin wrapper over ai-hwaccel v2.3.16 |
| 7 | `gpu_shaders.cyr` | `gpu/shaders.rs` | 1047 | WGSL strings survive byte-for-byte |
| 7 | `gpu_buffer.cyr` | `gpu/buffer.rs` | 207 | moderate |
| 7 | `gpu_context.cyr` | `gpu/context.rs` | 354 | moderate — `gpu_context_new_native*` is a direct call |
| 7 | `gpu_pipeline.cyr` | `gpu/pipeline.rs` | 2415 | hard — largest file; needs the §5.3 bind-layout builder |

---

## 7. Pre-port steps

Run **in this order**. `cyrius port` errors out if `rust-old/` already exists, so
there is exactly one shot at the pre-port state.

1. ✅ **DONE — extract the benchmark history first.** `_port_move`
   ([programs/cyrius-init.cyr:919](file:///home/macro/Repos/cyrius/programs/cyrius-init.cyr))
   sweeps `benches/` — and `bench-history.csv` and `benchmarks.md` — into
   `rust-old/`. ranga's `benches/history.csv` is the 534-row 0.20.3→1.0.1 series
   that [ADR 007](../decisions/007-final-rust-release.md) designates as the
   Cyrius implementation's performance target. It is relocated, not deleted, but
   per AGNOS first-party standards it must be lifted out before the sweep.
   Lifted to [`docs/benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md)
   (134 benchmarks normalised to ns, Cyrius columns blank) with the full 534-row
   series copied to [`docs/rust-v1-bench-history.csv`](../rust-v1-bench-history.csv),
   matching mabda's layout.
2. `git tag` the pre-port state — **user handles git; this gates step 3.** The
   Rust source is preserved as a git-tagged snapshot per the
   retirement-via-git-tag pattern, and it is the only way back once `_port_move`
   runs.
3. ✅ **DONE — `cyrius port /home/macro/Repos/ranga`.** 13,958 Rust lines moved to
   `rust-old/`; scaffolded `src/main.cyr`, `cyrius.cyml`, the `docs/` tree,
   `CLAUDE.md`, CI + release workflows, and `.tcyr`/`.bcyr`/`.fcyr` skeletons.
4. ✅ **DONE — manifest converted from binary to library shape.** `${file:VERSION}`
   restored as SSOT (the scaffolded `release.yml` explicitly checks for it),
   `repository` added, `output` → `build/ranga`, `[lib] modules` added and wired
   to `cyrius distlib` → `dist/ranga.cyr`. `[deps] stdlib` extended with `math`,
   `ganita`, `slice`, `freelist`, `tagged`.
   ⚠ `[lib] modules = []` is **not** valid — `cyrius distlib` fails with
   "no [build] modules or [lib] modules found in manifest", so the list needs a
   real module from the start. `src/error.cyr` is that module (M1's first).
5. ✅ **DONE — docs survived.** The scaffold added to `docs/` without clobbering;
   this plan, `benchmarks-rust-v-cyrius.md`, `rust-v1-bench-history.csv`, and
   `docs/decisions/` are all intact. It did overwrite `docs/development/roadmap.md`
   with its own template.
6. ✅ **DONE — `.gitignore`.** The port does **not** append its ignore entries, and
   `_port_move` swept the Rust build directory to `rust-old/target/` — **7.6 GB**
   that the pre-existing root-anchored `/target/` rule no longer matches. Added
   `rust-old/target/`, `/build/`, `cyrius-*.tar.gz`, `.claude/`. `lib/` and
   `dist/` stay tracked, matching prakash (108 tracked files under `lib/`).

**Settled — VERSION stays `1.0.1` through the port, bumps to `2.0.0` at M8.**
`distlib` stamps the bundle `v1.0.1` in the meantime, which is correct — the
Cyrius tree is not yet at parity with what 1.0.1 shipped, so claiming 2.0.0
before M8 would advertise a library that does not exist yet. This matches
prakash, which deferred its own `VERSION` → 2.0.0 to M8 "when the port
completed", and preserves the AGNOS precedent that ports continue numbering with
a major bump rather than resetting to 0.x. The scaffolded
`docs/development/roadmap.md` shipped a generic `v0.1.0 → v1.0` template; its
milestone numbering has been rewritten to match this decision.

`cyrius port` is a **scaffold-and-move tool, not a translator** — grepping
`programs/cyrius-init.cyr` and `cbt/cyrius.cyr` for
`transpile|translate|rs_to_cyr|parse_rust|syn::|rustc` returns zero matches. It
counts `.rs` lines, renames ~17 paths into `rust-old/`, writes
`LINES_OF_RUST.txt`, and lays down a skeleton. All translation is hand work
guarded by the ported test suite.

---

## 8. Testing, benchmarks, docs

- **Tests** (`tests/*.tcyr`): split per module — `ranga.tcyr` (smoke),
  `pixel.tcyr`, `color.tcyr`, `blend.tcyr`, `filter.tcyr`, `icc.tcyr`,
  `transform.tcyr`. Port all 214 unit + 123 integration cases. Float asserts need
  an `f32_approx_eq(a, b, eps)` helper. The 161 doctests have no direct Cyrius
  equivalent — fold their assertions into the `.tcyr` suite rather than dropping
  them.
- **Benchmarks** (`tests/ranga.bcyr`): mirror all 134 named benchmarks so the
  Rust series stays comparable. Use amplification for sub-timer-floor ops.
- **Fuzz** (`tests/ranga.fcyr`): port all 8 targets — blend, convert, filter,
  blur, lut, icc, composite, transform.
- **Parity check**: a port-completeness review counting every `pub fn`/struct/enum
  against `rust-old/`, as prakash did at 2.2.1 (282 fns, 37 structs, 5 enums).
- **`rust-old/` retirement**: kept 1–3 releases as the translation reference, then
  deleted with a CHANGELOG entry naming the
  `git checkout <tag> -- rust-old/` recovery incantation (prakash 2.2.3,
  mabda 2.1.2, ai-hwaccel 2.0.0 all did this). Per standards, delete **only after
  the Cyrius version has equal or better test coverage and benchmarks**.

---

## 9. Risks

1. **Volume.** 13,958 lines, and the two SIMD modules carry hand-written
   intrinsics. Mitigated by complete scalar fallbacks in both — correctness
   parity is mechanical; only performance parity is hard.
2. **Silent narrow-field corruption** (§2). The `#derive(accessors)` footgun and
   integer-multiply-on-float-field both compile clean. The `.tcyr` suite must
   carry bit-fidelity assertions on every color type, ported *before* the code
   that uses them.
3. **Silent enum-codec absence** (§3.7). `#derive(Serialize)` on an enum
   compiles rc=0 and emits nothing — the codec is undefined at link time.
   Works correctly on a struct.
4. **f32→f64→f32 round-tripping** changes results at tight tolerances. ranga's
   Delta-E and ICC tests have narrow bands; expect to re-baseline some.
5. **mabda backend churn.** wgpu retires across v5.0–v5.1 and Intel native is
   tentative. Pin an exact mabda tag and re-check the shader input format at
   port time rather than assuming today's answer holds.
6. **prakash is mid-port** — M0–M2 done, M3–M8 open. ranga needs only M2
   (Spectral), which is complete, so this is not currently blocking.

---

## 10. Settled decisions

1. **GPU ships in 2.0.0, targeting the native backends.** No context-injection
   workaround and no deferral — `gpu_context_new_native*` is a direct library
   call (§5). Build against native AMD/NVIDIA and treat wgpu as the fallback it
   is, since it leaves the tree at mabda v5.1.
2. **f32 uses the local shim, and the real fix is upstream in ganita.** ranga
   carries `ranga_f32_*` (§3.1) as an explicit stopgap. Filed as
   `cyrius/docs/development/issues/2026-08-13-ranga-ganita-f32-math-surface.md`,
   scoped to **ganita** rather than cyrius core — matching how
   `ganita_f64_pow`/`_atan2`/`_hypot` already live there. If a `ganita_f32_*`
   tier lands, ranga drops the shim; the other graphics consumers (soorat, rasa,
   tazama, aethersafta) would otherwise each re-roll the same eighty lines.
3. **SIMD lands in 2.0.0**, not deferred. The `asm { }` kernels ship alongside
   the scalar paths so 2.0.0 is performance-competitive against the Rust
   baseline at release rather than regressing hard and recovering later.

---

## 11. Milestones

- [x] **M0 Pre-port** — ✅ bench history extracted, tagged (`79acc52`), `cyrius port` run, manifest converted to library shape, `.gitignore` fixed, `distlib` → `dist/ranga.cyr` (130 lines), smoke `.tcyr` green.
- [ ] **M1 Foundation** — `error` ✅, `constants`, byte-vec, `pixel`, `color`. Settles the §2 accessor pattern.
      - [x] `src/error.cyr` — 8 error codes replacing `RangaError`'s 5 variants + `is_err`/`is_ok`, plus the Rust-semantics shims `_rg_pow`/`_rg_exp`/`_rg_sin`/`_rg_cos` (§3 item 1, modelled on prakash's `_prk_*`). **37 assertions green, lint clean.** The shim assertions are parity locks — each one fails against the bare Cyrius builtin and passes against Rust.
- [ ] **M2 Pixel ops** — `composite`, `histogram`, `transform`, `filter`.
- [ ] **M3 SIMD modules** — `convert`, `blend`. Scalar path first to establish correctness parity, then the `asm { }` kernels (items §3.4–3.6) in the same milestone — SIMD ships in 2.0.0.
- [ ] **M4 ICC** — `icc` + the 5 parametric curve types.
- [ ] **M5 External** — `spectral` (prakash), `hwaccel` (ai-hwaccel).
- [ ] **M6 GPU** — `gpu_shaders`, `gpu_buffer`, `gpu_context`, `gpu_pipeline` against the **native** mabda backends; ~35-line bind-layout builder; confirm shader input format against the pinned mabda tag.
- [ ] **M7 Parity** — surface count vs `rust-old/`, full benchmark sweep, `docs/benchmarks-rust-v-cyrius.md`.
- [ ] **M8 Release** — distlib, docs, CHANGELOG, CI gates, VERSION → 2.0.0.
