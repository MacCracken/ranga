# Port mechanics — what to automate, what to hand-write

Observations from porting M1's first four modules (`error`, `bytevec`, `pixel`,
`color` — 183 assertions green). Written while the friction was fresh, so the
remaining ~11,000 lines cost less than the first ~2,800 did.

The headline: **the translation itself is not the bottleneck.** Transcribing
arithmetic is fast. What cost time was (a) guessing stdlib names wrong, (b)
deciding struct field widths, and (c) hand-transcribing Rust doctests into
assertions. Only (b) genuinely needs judgment.

---

## 1. Workflow candidates, highest value first

### A. Doctest extraction — the single biggest mechanical win

`rust-old/` has **161 doctests**, and they are the highest-quality assertions in
the codebase: each is a worked example with a known-correct expected value,
written by someone who had the semantics in front of them. Every one I
transcribed by hand this round (`Rgb8.checked_buffer_size(10, 10) == Some(300)`,
`linear_to_srgb(1.0) == 255`, `get_rgba(99, 99) == None`) became a one-line
`.tcyr` assertion.

The extraction is fully mechanical: pull ` ``` ` blocks out of `///` comments,
keep the `assert_eq!` / `assert!` lines, drop the `use` lines and bindings.
**Emit them as a commented `.tcyr` skeleton with the Rust expression preserved
in a trailing comment**, then hand-translate the call syntax. Do not try to
auto-translate the expressions — the value is in never *losing* an assertion,
not in avoiding the typing.

Fan out one agent per module; they are independent. This is the first thing to
build.

### B. Surface parity oracle

prakash ran this at 2.2.1 (282 fns, 37 structs, 5 enums counted against
`rust-old/`). Mechanical both sides: `grep -E '^pub (fn|struct|enum)'` over
`rust-old/src/*.rs`, `grep -E '^fn '` over `src/*.cyr`, normalise the naming
convention (`PixelBuffer::get_rgba` → `pixel_buffer_get_rgba`), diff.

Worth building **now rather than at M7**, because run per-module it tells you
what you forgot while the module is still in your head. It is also the 2.0.0
release gate, so it gets written either way.

### C. Per-module port pipeline

Each module is the same shape, and modules at the same dependency depth are
independent:

```
read rust-old/src/X.rs → write src/X.cyr → write tests/X.tcyr
  → cyrius lint → cyrius test → append to [lib] modules
```

The §6 sequence gives the batches. M2's four (`composite`, `histogram`,
`transform`, `filter`) have no interdependencies — they all sit on `pixel` +
`color`, which are done. Same for M4/M5. That is a clean `pipeline()` per
module with a lint/test verify stage, and it is where fan-out actually pays.

**Do not fan out M3** (`convert`, `blend`) — the `asm { }` kernels need
instruction-level attention and a wrong VEX byte fails in ways a test summary
will not localise.

### D. Accessor generation

Every struct came out as the same five-part shape: layout comment, `X_new`,
per-field getters, optional setters, `X_free`. Given a field list and widths
that is templated code. `color.cyr` alone has four such records; `filter.rs` and
`icc.rs` bring more.

Lower value than A–C — writing them is fast, the risk is in choosing the widths
(§2 below), and a generator does not remove that.

---

## 2. What must stay hand-written

**Field widths.** Cyrius struct fields have no f32/f64/u8/u32 type, so every
Rust struct needs a per-field decision — `u8` → `store8`, `f32` → `store32` of a
bit pattern, `f64` → `store64`. Get it wrong and it compiles clean and silently
corrupts. `LinRgba` is f32 and `CieXyz` is f64 **in the original**, and this port
keeps that split rather than promoting everything; a generator working from
field names alone would flatten it.

**Ownership and frees.** Rust's `Drop` is invisible in the source, so there is
nothing to translate — it has to be reconstructed. `srgba_to_cie_lab` allocates
two intermediates and must free both; `buffer_pool_release` frees on overflow
because Rust dropped there. Each of these is a judgment call about who owns
what, and it is exactly what a mechanical pass would miss.

**Numeric edge-case shims.** Finding that `pow(0.0, y)` and `exp(-inf)` diverge
from Rust required knowing Rust's semantics *and* probing Cyrius's. prakash hit
the identical wall independently and shimmed it in its own `error.cyr` — so
**check prakash before deriving anything numeric**; `_prk_pow`/`_prk_exp`/
`_prk_sin`/`_prk_cos` transferred almost verbatim.

**Error mapping.** `Result<T, E>` becomes either an `err_out` pointer or a
negative-code sentinel, per function. `pixel_format_checked_buffer_size` returns
the size directly and encodes failure as a negative, which is safe only because
a real size is never negative. That reasoning does not generalise.

---

## 3. Idiom cheat-sheet

Each of these cost a compile cycle or a wrong guess this round.

| Need | Correct form | Note |
|---|---|---|
| string equality assert | `assert_streq` | **not** `assert_str_eq` — cost a compile |
| print a string | `print(msg, strlen(msg))` | takes 2 args; `print(msg)` fails |
| available asserts | `assert`, `assert_eq`, `assert_neq`, `assert_lt/lte/gt/gte`, `assert_streq`, `assert_nonnull`, `assert_fatal`, `assert_summary` | that is the whole list |
| `a <= b` on f64 | `f64_le(a, b) == 1` | stdlib `lib/math.cyr:437` — do not hand-roll as `f64_lt` + `f64_eq` |
| `a >= b` on f64 | `f64_ge(a, b) == 1` | same |
| float constant ≤ 9 sig digits | `0.4124564` | decimal literals lex as f64 and are exact here — more readable than a bit pattern |
| float constant > 9 sig digits | `0x3FDA61D629F2E197; # 0.4122214708` | ⚠ **decimals are SILENTLY WRONG past ~9 digits** — see below. prakash's hex style is right after all, just not needed everywhere |
| f32 arithmetic | bare `+ - * /` on a `: f32` binding | there is no callable `f32_add`; **annotate every f32 param** or it silently emits integer ops |
| f32 math | `ganita_f32_*` (23 fns) | ships in cycc 6.5.24 |
| f64 cbrt | `_rg_cbrt` (ours) | there is **no** f64 cbrt in the tree |
| negative literal | `(0 - 1)` | |
| right shift | `>>` logical, `>>>` arithmetic | ⚠ **reverse of JS/Java.** Pick by the SIGNEDNESS of the Rust operand: a `u64`/`u16` shift needs `>>`, an `i16`/`i32` shift that can go negative needs `>>>`. Getting it wrong is silent — a negative product zero-fills into a large positive |
| `stack` as an identifier | reserved | v5.5.36 stack-local array qualifier. Using it yields `expected '(', got ')'` pointing at an unrelated line several statements later |
| byte from a buffer | `load8` | zero-extends, so `0xFF` reads back 255 |
| `[lib] modules = []` | invalid | `distlib` fails; needs ≥1 real module |
| line length | ≤ 120 chars | lint warns; hoist long assertion args into locals |
| print an integer | `fmt_int(n)` | there is no `print_int` |
| `f32_from(0)` | valid | integer 0 is a usable +0.0f source |
| clamp with a possible NaN | guard explicitly | `ganita_f32_clamp` does **not** match Rust's `f32::clamp` on NaN — Rust passes NaN through (and a following `as u8` saturates it to 0); ganita's monotone key ranks NaN above every finite value and returns the HIGH bound. Any kernel that can produce a NaN before a clamp needs its own guard — `filter.rs` and the blend kernels are the candidates |

⚠ **`cyrius lint` and `cyrius test` report different diagnostics.** Lint reported
0 warnings on a `.tcyr` whose compile — driven by `cyrius test` — emitted
`#must_use result of 'chi_squared' is discarded`. A module is not clean until the
`cyrius test` output is warning-free too, not just the two lint lines. Always run
all three commands.

### Two silent-corruption traps, both hit while porting `color.cyr`

Neither errors, neither warns, and both produce plausible-looking output. They
are the reason the assertions in §4 matter.

**0. There are TWO correct spellings of f32 arithmetic, and one wrong one.**
The widen-compute-narrow form is *fine* — `f32_from(f64_mul(f32_to(a), f32_to(b)))`
is bit-identical to native f32 for a single operation, because f64 carries more
than 2·24+2 mantissa bits so double-rounding cannot bite. `ganita_f32_sqrt`
already relies on this. Only an end-to-end f64 chain that never narrows its
intermediates diverges. Do not "fix" a correct widen-narrow helper into the
trap below.

**0b. The compiler DOES have a guard for this — but only `cyrius test` shows it.**
`f64 arithmetic with a non-f64 right operand` is a real diagnostic, and it fires
on exactly the mixed-type mistake above. It is emitted by the compile that
`cyrius test` drives and **not** by `cyrius lint`, which is another reason to run
all three commands.

**BOTH operands must be typed, not just the left.** These are all wrong:

```
store64(out, a * px + c * py + load64(m + 32));   # untyped RIGHT  -> warns
var factor: f64 = load64(a + i) / pivot;          # untyped LEFT   -> silent, gives 0
s = s + _tf_cubic_weight(x);                       # call result on the right -> warns
```

Bind every operand to a `: f64` local first, then combine. The left-operand form
is the dangerous one: it produced a `from_quad` that reported every quad as
degenerate, with no diagnostic at all.

**1. f32 arithmetic on an untyped operand becomes an integer multiply.**
The operators dispatch through `EMIT_F32_BINOP` only when the operand carries
`F32_TYID`, which comes from a typed binding or a typed parameter. A **call
result is untyped**:

```
var r: f32 = load32(c);
var x = f32_from(3.0) * r;      # ← integer multiply. measured: 0, not 6.
```

The fix is to route matrix rows through a helper with typed parameters —
`fn _rg_f32_dot3(m0: f32, m1: f32, m2: f32, x: f32, y: f32, z: f32)`. Annotating
the *parameters* is what makes the body single-precision. I documented this trap
in the plan and then walked straight into it anyway; assume you will too.

**2. Decimal float literals past ~9 significant digits parse to a wrong value.**
Exact through `3.14159265`; `3.1415926535` becomes `0.95822` and
`3.141592653589793` becomes `0.061575`. Filed upstream as
`cyrius/docs/development/issues/2026-08-17-decimal-float-literal-silent-precision-loss.md`.

This produced an Oklab lightness of **6.447** for linear white instead of 1.0.
Use hex bit patterns above ~9 digits (`struct.unpack('<Q', struct.pack('<d', v))`),
decimals below. Prefer stdlib constants (`F64_PI`) over retyping them, and
`f64_from(n)` for integer-valued constants like 25⁷ — exact below 2⁵³.

**Both were caught only by an identity assertion** — white → Oklab l == 1, and
`cos(180°) == -1`. Neither would have been caught by a round-trip test, because
a wrong-but-consistent matrix round-trips perfectly.

**Test-local helpers pay for themselves immediately.** `f32c(n)` /`f32i(x)` for
f32 round-tripping and `approx(a, b, eps)` for float comparison turned
120-column lint failures into readable one-liners. Define them at the top of
every `.tcyr`.

---

## 4. Assertion patterns worth repeating

Not all assertions are equal. These earned their place:

- **Exhaustive round-trips over a small domain.** `srgb → linear → srgb` across
  all 256 byte values is one loop and it pins the `+0.5` rounding term that
  Rust's `as u8` truncation makes load-bearing. Cheap, and it catches an
  off-by-one across the whole range rather than at one sampled point.
- **Both branches of a piecewise function.** `_lab_f` has a cbrt branch and a
  linear branch; white exercises one and black the other. Testing only white
  would leave the linear segment unverified.
- **Identities that pin a whole matrix.** Linear white through `SRGB_TO_XYZ`
  must reproduce the D65 white point. One assertion catches a transposed or
  mistyped coefficient anywhere in the nine.
- **The case the naive implementation gets wrong.** `ganita_f32_min(-2, -1)`
  passes trivially with a correct implementation and fails loudly with the
  raw-unsigned-compare version this plan originally proposed. Write the
  assertion *for the bug you almost shipped*.
- **Semantic invariants, not just values.** Alpha must not go through the gamma
  transfer — asserting `alpha 128 → 0.502` states the intent, so a later
  "simplification" that routes alpha through `srgb_to_linear` fails here.

### Choose inputs that can actually *discriminate*

Most assertions that look meaningful catch nothing. Four patterns, all found by
mutation-testing M2 rather than by reasoning:

- **Round numbers cannot detect an f32→f64 widening.** `100 * (85/100)`
  truncates to 85 at both widths. You have to hunt for ratios that disagree —
  `39 * (85/39)` is 85 in f32 and 84 in f64. Likewise the `+0.5` round-half-up
  term is invisible on a tidy 0/85/170/255 ramp, where every LUT entry is
  already an exact integer; it took a randomised search over 200k images to find
  a case (4/11/13/15/195) that shifts 64→62 without it.
- **Symmetric images are blind to coefficient swaps.** A 2×1 image with one pure
  green and one pure blue pixel passes *unchanged* against a BT.601 kernel with
  the green and blue coefficients swapped — the mass just moves between the two
  pixels and bins 149 and 28 still hold 0.5 each. Only one primary per buffer
  pins which coefficient is which. Same family as the wrong-but-consistent
  matrix above.
- **Alpha 255 is not a test of alpha preservation.** It survives essentially any
  corruption. Use a mid value (40, 200).
- **Boundaries need both sides.** `bins/256` vs `bins/255` are indistinguishable
  at `bins == 256`; grey 219 with 7 bins separates them (bin 5 vs bin 6). Truncation
  vs rounding needs an input landing on `x.5`, not on a bin edge.

- **Guards mask each other, so exercise them one at a time.** `resize` has four
  zero-dimension guards and `crop` has two ordering guards; a 0×0 source or a
  fully-inverted rectangle is caught by whichever guard runs first, which
  short-circuits and hides the rest. Removing any single guard still passed. A
  0×4 source, a 4×0 source, an x-only inversion and a y-only inversion each pin
  exactly one.
- **Probe every basis vector.** A rotation's `b` and `c` have opposite signs, but
  applying it only to (1,0) exercises `a` and `b`; flipping the sign of `c` —
  rotating the wrong way — passed until (0,1) was probed too. Same for
  `affine_inverse`: the first round-trip used translate-then-scale, where
  `b == c == 0`, so the off-diagonal negations were never exercised.

**Budget a numeric-oracle pass per module.** Deriving discriminating inputs by
hand gets the rounding wrong; a short Python/numpy search finds them reliably.

**The oracle also stops you "fixing" correct code.** Two examples from
`transform`, both of which looked like port bugs and were not:

- *"A uniform image survives bicubic unchanged"* is FALSE. The four cubic weights
  sum to exactly 1.0 in isolation, but after sixteen multiply-accumulates the
  interior accumulator lands on `76.99999999999996`, and `as u8` truncates it to
  76. Rust produces the same 76. Asserting 77 would have meant rewriting a
  faithful port.
- A hand-written "degenerate perspective" matrix `[[1,0,0],[0,1,0],[1,0,0]]` has
  `det == 0`, so it is rejected up front and never reaches the NaN path it was
  meant to test. The oracle found `[[1,0,0],[0,1,0],[0.4,0,-0.4]]`, which is
  invertible and whose *inverse* is singular exactly at a sampled coordinate.

### Hand-encoded `asm { }` is defensible only with a differential oracle

The mechanics, all verified on 6.5.27 and none of them documented:

- Raw bytes, semicolon-separated, inside `asm { ... }`, gated with
  `#ifdef CYRIUS_ARCH_X86` / `#ifndef` and a scalar fallback.
- **Parameters live at `[rbp-8]`, `[rbp-16]`, `[rbp-24]`, … in declaration
  order, and the first local follows.** Verified with a probe function before
  writing anything real. Adding or reordering a parameter silently breaks every
  asm block in that function, so signatures there are load-bearing.
- Produce every encoding with `llvm-mc -arch=x86-64 -show-encoding`. Never from
  memory: `paddusb` and `paddb` differ in one byte (`0xDC` vs `0xFC`) and the
  wrong one wraps instead of saturating.
- In-tree precedents: `_fhm_probe16` (lib/hashmap_fast.cyr:66) for the SSE2
  shape, `_sha_ni_compress_one` (lib/sigil.cyr:4871) for VEX-encoded AVX2.

The scalar fallback is the *oracle*, not a consolation prize: write it first,
then differentially test the asm against it across an exhaustive sweep. That is
what makes hand-assembled bytes reviewable — a reader checks the fallback and
trusts the test, rather than checking the hex.

### Hot loops must be written flat — helper calls are real calls

Cyrius leaves general function inlining **off by default** (`_INLINE_OK`, a
compiler-internal default-0 a consumer cannot switch on). The readable factoring
that is right for module surface is wrong on a per-pixel or per-tap path.
Measured by rewriting the separable blur passes to do their f32 arithmetic
inline rather than through `_fk_fma` / `_fk_u8_to_f32`:

| | before | after | speedup |
|---|---:|---:|---:|
| `box_blur_r3_1080p` | 833.03 ms | 395.10 ms | **2.11x** |
| `gaussian_blur_r3_1080p` | 832.98 ms | 399.32 ms | **2.09x** |
| `vignette_1080p` | 178.43 ms | 122.98 ms | 1.45x |

Keep the helpers for clarity everywhere else; flatten the innermost loop.

### Check the allocator before blaming your own code

`fl_calloc` zeroes byte-at-a-time on top of an `mmap` the kernel already zeroed —
measured **10.9 ms vs 29.6 us** for an 8.3 MB frame, a 369x gap. That single
stdlib detail was most of `crop` being 79.9x slower than Rust, and none of it
was in ranga's code. Filed upstream as
`2026-08-17-fl-calloc-byte-loop-over-already-zero-mmap.md`.

The lesson generalises: when a ported function is dramatically slower than its
Rust original and the arithmetic looks equivalent, measure the primitives it
sits on before rewriting the function.

⚠ The consumer-side workaround (`bv_uninit`) has a hazard that **cannot be
tested**: large allocations come from fresh mmap, so an incorrect use still
reads as zero in every test and only breaks once the allocator recycles. A
mutation swapping `affine_transform` to the uninitialised path SURVIVED the full
suite for exactly that reason. The comment on `bv_uninit` is the only guard —
treat it as review discipline, not a tested invariant.

### Mutation-test each module the moment its suite goes green

Twenty sed-level defects take about two minutes and are a far better use of time
than writing more assertions blind. On `histogram` this exposed three assertions
that looked meaningful and caught nothing, plus one piece of genuinely
unreachable defensive code — which is better documented as unreachable than
papered over with a fake assertion. Fold it into the per-module pipeline (§1C)
as a stage after test-green.
