# ADR 002 — f32 transcendentals follow correct rounding, not the oracle host's libm

**Status:** Accepted, 2026-11-02, for ranga 2.2.0
**Context:** the port's bar is "matches what Rust did", and divergence needs an ADR.
`scripts/oracle` now measures that bar bit for bit. This ADR covers the one class of
difference that remains in the colour code once every arithmetic path is in Rust's
precision and order.

## The divergence

Rust's f32 `sin`, `cos`, `powf` and `ln` are not Rust's own code. They lower to the
platform C library: glibc on the Linux host the oracle runs on, Apple's libm on macOS,
the UCRT on Windows, musl on Alpine. ranga computes the same functions through
ganita's f32 tier. That tier evaluates in f64 and rounds once to f32, so its result
is the correctly rounded f32 except in vanishingly rare hard cases.

glibc's float functions are not correctly rounded; their documented error bound
reaches about 1 ULP. So where glibc rounds the wrong way, ranga and Rust-on-glibc
differ by one f32 ULP at the function, and by at most 2 ULPs after the product that
follows it. Measured with `scripts/oracle` at 2.2.0:

| group | compared | differ | worst | max abs |
| --- | ---: | ---: | ---: | ---: |
| `oklch_to_oklab.a` (`c * cos h`) | 140,608 | 1,419 | 2 ULP | 3.0e-8 |
| `oklch_to_oklab.b` (`c * sin h`) | 140,608 | 1,830 | 2 ULP | 3.0e-8 |
| `color_temperature` (`powf`, `ln`) | 16,716 | 6 | 2 ULP | 1.2e-7 |

**Every** differing value was checked against an independent model: correctly rounded
cos/sin/pow/ln (an f64 evaluation, then one rounding to f32), with Rust's arithmetic
around it. The model reproduces ranga's value every time:
- 400 of 400 sampled points in Oklch;
- 6 of 6 in `color_temperature`.

It reproduces Rust's value never. The difference is glibc's rounding, not ranga's.

## Decision

**ranga keeps the correctly rounded results, and the oracle accepts these groups
within a 2-ULP bound.** `scripts/oracle/compare.py` lists:
- `oklch_to_oklab.a`;
- `oklch_to_oklab.b`;
- `color_temperature`.

Each is accepted only while its worst difference stays ≤ 2 ULPs. A larger difference
in any of them still fails the oracle, so this ADR cannot mask a real arithmetic or
ordering bug in those functions.

The scope is f32 *fields* produced through a libm transcendental. The following stay
held to bit-exactness by the oracle:
- **Byte outputs.** No byte output in ranga depends on these functions today; every
  u8 group matches Rust exactly.
- **Arithmetic-only paths** such as sqrt, atan2 or the HSL/CMYK conversions. Those
  matched Rust bit for bit once they ran in f32 in Rust's order.

## Consequences

- **Positive:**
  - ranga's results do not depend on which C library the host has. Rust 1.0.1 never
    had that property: the same Rust build gives different bits on glibc, macOS and
    Windows.
  - The accepted differences are bounded, measured, and re-checked on every oracle
    run.
- **Negative:**
  - On glibc, ranga is not bit-identical to Rust for these three groups. A consumer
    that hashed Rust's Oklch→Oklab output on Linux would see a different hash for
    about 1% of inputs.
- **Neutral:**
  - If ganita's f32 tier ever moves (a stdlib snapshot changed ICC f64 `pow` results
    by 1–2 ULPs between cyrius 6.6.9 and 6.7.18, per the 2.1.2 audit), the oracle
    shows it. The 2-ULP bound then decides whether it is still covered here.

## Alternatives considered

- **Port glibc's `sinf`, `cosf`, `powf` and `logf`.** Rejected:
  - it would reproduce one platform's rounding errors deliberately;
  - Rust itself does not give those bits on macOS or Windows;
  - it is a few hundred lines of table-driven numerics to own for a ≤ 2-ULP effect.
- **Leave the groups failing.** Rejected: an oracle that always fails gets ignored,
  and these differences are understood.
- **Widen ranga to f64 here.** Rejected: that is what 2.1.2 and earlier did. It made
  the differences larger and more numerous (93,514 differing values in
  `oklch_to_oklab.a` instead of 1,419), because the f64 path also changed the
  arithmetic around the transcendental.
