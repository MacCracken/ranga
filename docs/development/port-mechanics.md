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
| byte from a buffer | `load8` | zero-extends, so `0xFF` reads back 255 |
| `[lib] modules = []` | invalid | `distlib` fails; needs ≥1 real module |
| line length | ≤ 120 chars | lint warns; hoist long assertion args into locals |

### Two silent-corruption traps, both hit while porting `color.cyr`

Neither errors, neither warns, and both produce plausible-looking output. They
are the reason the assertions in §4 matter.

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
