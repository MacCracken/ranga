# Surface parity — ranga Rust 1.0.1 vs the Cyrius port

> M7. Audited module by module against `rust-old/`, then each claimed gap
> adversarially re-checked against the files. This is a **surface** audit: it
> asks whether every public Rust item has a counterpart, not whether the
> counterpart computes the same numbers. Behavioural parity is what the 1,700+
> assertions and the per-module mutation testing cover.

## Totals

| | Count |
| --- | ---: |
| Rust public items (fns incl. impl methods, structs, enums, consts, trait impls) | **318** |
| Ported — same name or the expected mechanical rename | **222** |
| Deliberate omission — absent with a documented reason | **22** |
| **Missing** | **51** |
| Cyrius public fns | 370 |

The Cyrius surface is larger than the Rust one because Cyrius has no methods,
no `Drop`, and no `std`: every `X::y()` becomes a free `x_y()`, every owning
type needs an explicit `*_free`, and `bytevec.cyr` exists because `Vec<u8>` does
not.

## Where the 51 gaps are

| Group | Rust items | Missing | What is missing |
| --- | ---: | ---: | --- |
| gpu | 75 | **28** | **the entirety of `pipeline.rs`** |
| icc / spectral / hwaccel | 49 | 9 | derive impls, `IccLutProfile` accessors |
| error / pixel | 61 | 4 | `RangaError` Display/Error, `into_data` |
| transform / filter | 53 | 6 | derive impls, two `ScaleFilter` niceties |
| color / convert | 52 | 2 | `ColorSpace::Bt601`, one `FromStr` |
| composite / histogram / blend | 28 | 2 | two `BlendMode` derive impls |

**Half the parity debt is one file.** `gpu/pipeline.rs` holds the fourteen
public `gpu_*` operations and `GpuChain`, and none of it is ported. What IS
ported is everything underneath: the context, the buffers, the SPIR-V emitter,
eleven kernels and all twenty-one WGSL sources. So the gap is **wiring, not
maths** — each missing operation is a function that uploads, picks a kernel,
dispatches and downloads.

⚠ **A defect the audit's own summary buried under "derive impls".** ranga's
`ColorSpace` was not a renumbered Rust enum — it was a DIFFERENT enum wearing
the same name. Rust declares `Srgb, LinearRgb, DisplayP3, Bt601, Bt709, Bt2020,
CieXyz`; the port dropped **Bt601 and Bt709** and let its own additions CieLab
and Oklab take values 3-6. So a `ColorSpace` persisted by ranga 1.0.1 read back
here as the wrong space — Rust's `Bt2020` (5) as CIE L\*a\*b\*, its `CieXyz`
(6) as Oklab — with nothing to signal it, while `convert.cyr` had BT.601 and
BT.709 conversions no ColorSpace value could name. Now corrected: the first
seven discriminants are Rust's, and the two ranga additions are numbered above
Rust's range so they cannot displace anything.

**The systemic fix matters more than the fix.** PixelFormat, BlendMode and
ScaleFilter were all correct, but only PixelFormat's numbering was asserted
anywhere — the other three suites checked `*_name()` strings, which pass under
any renumbering. All four enums now have explicit discriminant guards.

The remaining gaps break down as follows. **Cyrius has `#derive`**, so "no
analogue" is not the explanation for any of them:

- `#derive(accessors)` is a documented Cyrius feature that generates
  `T_field()` / `T_set_field()` pairs. ranga does not use it: every struct here
  is hand-rolled `load64`/`store64` against a byte layout written into the
  module header. That was a choice — the layouts are load-bearing documentation
  and several are read by hand-encoded asm — not a limitation.
- `#derive(Serialize)` / `#derive(Deserialize)` exist and **work on structs**
  (cyrius's own `lib/sigil.cyr` derives Serialize on `struct ima_status`). On an
  **enum** the derive compiles rc=0 with no diagnostic and emits **no
  codec at all** — the generated `<name>_to_json` is simply undefined at link
  time. Since ranga's serde surface is exactly four enums (`PixelFormat`,
  `BlendMode`, `ColorSpace` and one payload enum), the derive is unusable here
  and the codecs are hand-written. The port plan recorded this as a "misnamed,
  crashing codec"; the symptom has since changed to a silent no-op, same
  conclusion.
- `Debug`, `Clone`, `Copy`, `PartialEq`, `Hash` genuinely have no Cyrius derive
  and mostly need none — an `i64` is already copyable and comparable.

What genuinely costs something is string-facing: no `FromStr` means a consumer
cannot parse `"Multiply"` back into a `BlendMode`, and there is no inverse of
`blend_mode_name`.

## Fixed by this audit

The audit's value was not the inventory — it was four things it found wrong in
code that was already passing its tests:

1. **`pixel_view_new` accepted oversized buffers.** It tested `len < want`; Rust
   tests `data.len() != expected` and returns `DimensionMismatch`. The port
   silently widened the contract — a caller passing a whole frame where a tile
   was meant got a view reading the wrong rows instead of a diagnostic. Fixed to
   exact equality, with the error code corrected to match Rust, and both
   directions now asserted.

2. **`RangaError`'s entire Display surface was absent.** Eight `RG_ERR_*` codes
   with no way to render them — while `gpu_error_message` in the GPU module
   already established the idiom. Added `ranga_error_message`, with a test that
   no two codes share a string.

3. **`gpu_context_adapter_name` was a stub reporting itself as ported.** It
   returns the literal `"unknown"` unconditionally; nothing ever assigns the
   field, because mabda exposes no live adapter query. Now documented as a stub,
   given a `gpu_context_has_adapter_name` companion, and PINNED by a test that
   asserts the stub state — so the day mabda can report a name, the suite fails
   and the contract gets updated rather than a consumer branching on "unknown"
   forever.

4. **Two comments asserted things that were not true.** `pixel.cyr` claimed the
   `PixelView`/`PixelViewMut` split "survives as naming plus the mutating
   accessors being defined only for the Mut form" — there are no such accessors,
   and both forms read through a writable pointer. `gpu_spirv.cyr` claimed
   mabda's lowerer handles "exactly five" GLSL.std.450 ops; it handles eight,
   and the three unlisted ones include `FClamp`, which several kernels here
   spell the long way as FMax-then-FMin.

## Divergences that are deliberate, recorded so they are not re-litigated

- **Error payloads are gone.** `BufferTooSmall { need, have }` and friends carry
  data an `i64` cannot. `ranga_error_message` returns the format string with the
  placeholders removed rather than a message implying detail it lacks. Rust's own
  tests assert on the interpolated values and have no analogue here.
- **`PixelView` and `PixelViewMut` share one representation.** Cyrius cannot
  express borrow mutability. The distinction is naming only, and now says so.
- **Panicking variants collapse into checked ones.** `PixelFormat::buffer_size`
  merges into `pixel_format_checked_buffer_size` because there is no panic idiom;
  callers who ignore the negative return are where `.expect()` used to be.
- **`GpuBuffer::download_async` is not ported.** It existed to work around
  `mabda::PendingReadback::finish()` needing a non-`Send` `&Device` — a Rust
  borrow-checker artifact. The Cyrius mabda exposes no async readback to build
  on. The consumer cost is real: no way to overlap readback with CPU work.
- **`data`/`data_mut` and `rows`/`rows_mut` merge.** Cyrius pointers carry no
  mutability, so the pairs are one function each.
- **GPU grayscale is BT.709, CPU grayscale is BT.601.** Inherited from Rust,
  which had the same split. Reconciling either side would change output the Rust
  line produced.

## Remaining behavioural divergences worth an ADR

Found while checking surface parity, not counted as gaps:

- `PixelBuffer::new` maps a size overflow to `BufferTooSmall { need: usize::MAX }`
  in Rust; the port returns `RG_ERR_ALLOCATION`.
- Rust's `PixelView::new` calls the *panicking* `buffer_size`, so `u32::MAX`
  dimensions abort the process; the port returns an error code. Likewise
  `rows()` on a planar format `debug_assert`s in Rust and yields wrong slices in
  release, where `pixel_buffer_row_offset` returns `RG_ERR_INVALID_FORMAT`. Both
  are hardening, and both mean a program that aborted under Rust now continues.


## Final behavioural sweep (pre-2.0.0 tag)

M7 asked whether a counterpart EXISTS. This asked whether it COMPUTES THE SAME
THING — every ported function compared arithmetic-by-arithmetic against
`rust-old/`, with each claimed divergence handed to a second agent told to
refute it by running both sides.

### Fixed before the tag

1. **`oklab(black)` was `(NaN, NaN, NaN)`.** `ganita_f32_cbrt(0.0)` returns NaN —
   it is `exp(y·ln x)` underneath, and `ln(0)` is -inf — where Rust's
   `f32::cbrt(0.0)` is `0.0`. The NaN propagated through the entire M2 matrix.
   Black is the most common pixel value there is. Rust pins this in its own
   `oklab_black` test, which the port never ported; that test now exists.
   Fixed with `_rg_f32_cbrt`, which intercepts ONLY zero so every other input
   keeps the f32 tier exactly rather than being widened to f64 to fix one input.
2. **`apply_lut3d` truncated where Rust rounds half-up.** Rust is
   `(v * 255.0 + 0.5).clamp(…) as u8`. A LUT entry of exactly 0.5 gave 127
   instead of 128, and half of all grey levels came out one count low. Note the
   +0.5 is NOT the general convention in `filter.rs` — `unsharp_mask` and
   `vignette` genuinely truncate — so the fix is at the call site, not in the
   shared `_fk_f32_to_u8`.
3. **An ICC LUT profile with `grid_size == 0` read below its allocation.** Rust
   checks only `> 64`, so zero underflows `grid_size - 1`: Rust panics in debug
   and indexes wildly in release. The byte comes from an untrusted file. The
   port now rejects `< 2` — a deliberate divergence, and strictly safer.

⚠ **All three were in code with green tests.** The Oklab bug survived eight
milestones because the one Rust test that would have caught it was not ported,
and every existing lut3d test used entries that did not land on a .5 boundary.

### Known divergences — MEASURED, not reasoned about

⚠ **An earlier draft of this section listed seven divergences as "real and
confirmed by running both sides." Five of them were false.** They came from an
agent sweep whose findings I recorded without re-measuring. Every claim below
has now been checked by building the actual Rust crate at `rust-old/` with
`cargo --release` and diffing its output against the port under cyrius 6.5.31.

**Did not reproduce — the port is byte-identical to Rust:**

| claimed divergence | measurement |
| --- | --- |
| `linear_to_srgb` f64 vs f32 | **14,013 inputs**, including boundaries, negatives and out-of-range — 0 differ |
| `rgbaf32_to_rgba8` rounding | identical on every sample |
| `cmyk_to_srgba` / `srgba_to_cmyk` | identical, forward and round-trip |
| `levels` gamma/scale width | byte-identical across an 8-pixel ramp |
| `affine_inverse` reassociation | 4 transforms including rotation and sub-pixel translation — identical |

**Reproduced and FIXED:**

- **blend's `f32 → u8` tail computed in f64.** Rust's
  `(result * 255.0 + 0.5).clamp(…) as u8` is single-precision throughout;
  widening first is more accurate and therefore wrong. ColorBurn on
  `src(10,200,30) / dst(254,3,180)` gave 229 against Rust's 230. The tail now
  carries `: f32` annotations and matches exactly.

**Reproduced and ACCEPTED as a divergence — see
[ADR 001](../adr/001-yuv-inverse-does-not-reproduce-rusts-i16-wrap.md):**

- **The YUV→RGB inverse does not wrap at i16.** Rust's does, and saturated red
  round-trips to **black**. That is an overflow, not a colour decision.
  Reproducing it would mean deliberately re-introducing the bug.

**Then measured, and all three were real — the worst of the whole sweep:**

- **`linear_to_srgb` returned `i64::MIN` for NaN, and for +inf.** Rust's
  float-to-int `as` cast SATURATES — NaN to 0, +inf to 255, -inf to 0 — while
  Cyrius's `f64_to` returns `i64::MIN` for anything it cannot represent. Both
  clamp comparisons are false for NaN, so the function fell through and returned
  -9223372036854775808 from something whose contract is a byte. The +inf case
  has a second cause: **`f64_exp(+inf)` returns NaN in Cyrius** rather than
  +inf, the same builtin quirk as `exp(-inf)` that ganita 1.1.4 works around for
  zero bases but not for infinity. Fixed; the 4,012-input finite sweep confirms
  no regression.
- **`levels` diverged three ways.** Rust guards only `gamma <= 0.0`, which is
  false for NaN, so Rust PROCEEDS on a NaN gamma and blackens — the port
  rejected it. Rust reports a non-positive gamma as `RangaError::Other`, not an
  invalid parameter. And `f32::max` DISCARDS a NaN operand, so a NaN white point
  gives `range = 1e-6` and the image WHITENS in Rust, where the port's bare
  comparison let the NaN survive and blackened it — the opposite result. All
  three fixed.
- **`brightness` with a NaN offset blackened the image**; Rust's saturating
  `as i16` makes it a complete no-op. Fixed.
- **`_icc_write_s15fixed16` used banker's rounding.** Rust's `.round()` is
  half-away-from-zero; `f64_round` is half-to-even, and they differ in both
  directions on an exact .5 — 0.5 gave 0 against Rust's 1, 2.5 gave 2 against 3.
  An s15Fixed16 value lands on a .5 exactly when it is an odd multiple of
  2^-17, which real ICC matrices hit. Fixed with an explicit
  `_icc_round_half_away`.
- **`pixel_format_checked_buffer_size` bounded the PIXEL count** against one
  constant equal to `floor(i64::MAX/16)` — the divisor only RgbaF32 needs — and
  so rejected sizes Rust accepts: 1e9 x 1e9 Rgba8 is `Some(4000000000000000000)`
  in Rust and was `RG_ERR_ALLOCATION` here. The guard is now per-format, and the
  comment claiming "i64 is wide enough that u32*u32*16 cannot overflow it" is
  gone — `(2^32-1)^2` alone is 1.8e19 against i64::MAX 9.2e18.

**A second Rust i16 overflow surfaced while measuring**, alongside the YUV one:
`brightness` with an offset of 1000 gives `(255,0,0)` in Rust — some channels
wrap to 0 instead of saturating — where the port saturates to `(255,255,255)`.
Same judgement as ADR 001: not reproduced.


## Exhaustive differential sweep — the measurement that should have come first

Everything above was found by sampling. This is the systematic version: the
crate at `rust-old/` built with `cargo --release`, driven over a broad input
space, diffed value-by-value against the same calls in the port under cyrius
6.5.31. **5,800 comparisons.**

| group | coverage | before | after |
| --- | --- | ---: | ---: |
| `srgb_to_linear` | all 256 u8 inputs | **214 differ** | **0** |
| `blend_pixel` | 12 modes x 256 src/dst pairs | 0 | 0 |
| colour round trips | 216 colours x 8 outputs | 504* | 49 |
| `delta_e` x3 | 12 Lab pairs | 1 | 1 |
| format conversions | rgb8, argb8, YUV BT.709, BT.2020 | 0 | 0 |
| composite | premultiply, unpremultiply, fade | 0 | 0 |

\* inflated by a harness bug of mine — comparing an f64 field against an f32
one. The real figure was 49.

**Every u8 output path is now byte-exact against Rust.** The 50 residual
differences are all in f32/f64 *intermediates* that are never rounded to a byte:

- `CieLab.l` — 34 values, max absolute error **1.4e-14** on a 0..100 scale where
  one u8 step of L\* is 0.39. Thirteen orders of magnitude below significance.
  It comes from `f64::cbrt` against ganita's `pow(x, 1/3)` — different
  algorithms, not a defect.
- `Hsl.h` — 15 values, max error **1.5e-5 degrees** out of 360. The port is the
  MORE accurate side here: it returns exactly 200.0 where Rust returns
  200.0000152588.
- one `delta_e_ciede2000` value, 1 ULP in f64.

### What this exercise actually showed

The `srgb_to_linear` result is the one worth keeping: **214 of 256 inputs were
wrong, and the entire test suite was green.** Every hand-written assertion in
`tests/color.tcyr` used a tolerance wide enough to swallow a 1-ULP error,
because that is what a human writes when checking a colour conversion by eye.
Only a bit-exact diff against the real oracle could see it.

That is the argument for building the oracle FIRST. `cargo build` on `rust-old/`
succeeds in under a second; this sweep is a few hundred lines. Doing it at the
start of M7 would have caught `srgb_to_linear`, the Oklab NaN, the LUT3D
rounding, the `linear_to_srgb` `i64::MIN`, and the `levels` NaN divergences in
one pass, instead of over a dozen separate rounds of "one more issue".
