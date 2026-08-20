# ADR 001 — The YUV→RGB inverse does not reproduce Rust's i16 wrap

**Status:** Accepted, 2026-08-19, for ranga 2.0.0
**Context:** the Rust→Cyrius port's correctness bar is "matches what Rust did".
This is the one place we knowingly do not.

## The divergence

`yuv420p_to_rgba` computes the chroma contribution differently from the Rust
line. Rust performs the whole inverse in `i16`; the port computes it wider and
clamps at the end.

Measured on real toolchains — the crate at `rust-old/` built with `cargo
--release`, and the port under cyrius 6.5.31 — round-tripping eight saturated
pixels through `rgba_to_yuv420p` then `yuv420p_to_rgba`:

| input | Rust 1.0.1 | ranga 2.0.0 |
| --- | --- | --- |
| red `(255,0,0)` | **`(0,0,0)`** | `(254,0,0)` |
| blue `(0,0,255)` | **`(0,0,0)`** | `(0,0,253)` |
| yellow `(255,255,0)` | `(196,198,195)` | `(196,198,255)` |
| white `(255,255,255)` | `(225,227,224)` | `(225,227,255)` |

## Why we are not matching it

Rust's output is wrong. Saturated red decoding to black is an `i16` overflow,
not a colour-science decision: the chroma term `(V-128) * 179` exceeds
`i16::MAX` before the `>> 7`, wraps negative, and the subsequent clamp pins it
to 0. Nothing about BT.601 says red becomes black.

Reproducing it would mean deliberately re-introducing an overflow so that ranga
2.0.0 turns saturated colour channels black on a YUV round trip. That is a worse
outcome than the divergence, and it is not what any consumer of an image library
wants from a bug-for-bug port.

## What we do instead

The port computes the inverse without wrapping and clamps to `[0, 255]` at the
end, which is what the arithmetic means. Every non-saturated pixel is
unaffected — the divergence only appears where Rust's intermediate exceeded
`i16`, which is exactly where Rust was already wrong.

## Consequences

- **A consumer diffing ranga 2.0.0 output against 1.0.1 will see differences on
  saturated colours after a YUV round trip.** They are improvements, but they
  are differences, and anything pinning golden images will need regenerating.
- The port can no longer claim bit-identical YUV output against the Rust line.
  `docs/development/parity-rust-v-cyrius.md` records it as the sole intentional
  behavioural divergence.
- If bug-compatibility is ever genuinely needed, it belongs behind an explicit
  opt-in rather than as the default.

## A second instance

`filter::brightness` overflows the same way. Measured against the real crate, an
offset of 1000 on `(0,64,128)` gives `(255,0,0)` — red saturates, green and blue
WRAP to 0 — where the port gives `(255,255,255)`. The same judgement applies for
the same reason, and this ADR covers both.

## How this was found

The final pre-tag parity sweep. Worth recording that the same sweep produced
five other "confirmed divergences" — in `linear_to_srgb`, `rgbaf32_to_rgba8`,
`cmyk_to_srgba`, `levels` and `affine_transform` — and **all five failed to
reproduce** when measured against the real crate rather than reasoned about:
14,013 inputs through `linear_to_srgb` alone, byte-identical. Only this one and
a one-count blend rounding difference survived measurement, and the blend one
was fixed rather than accepted.
