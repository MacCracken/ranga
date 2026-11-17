# ranga's sRGB and premultiply functions share names with hisab's: a consumer with both can't compile, or gets wrong colours

**Status:** 🔴 **OPEN**, companion to the primary request in hisab,
`hisab/docs/development/issues/2026-11-15-color-names-collide-with-ranga.md`.
**Filed:** 2026-11-15, from soorat's Cyrius-port probe P11:
- the probe: `soorat/scripts/probes/bundle/p11_full/`, reproduced by `bundle/run.sh p11`;
- the write-up: `soorat/docs/development/port-parity.md` §1.4, ledger row L18 in `soorat-port-plan.md`.

**Placement:** `src/color.cyr` (`srgb_to_linear` at line 316, `linear_to_srgb` at 338),
`src/composite.cyr` (`premultiply_alpha` at 183, `unpremultiply_alpha` at 205), and the guidance comment in
`src/spectral.cyr` (around line 50).
**Severity:** **High** for consumers that need both libraries; none for ranga alone.
**Affects:** ranga 2.4.0 with hisab 3.4.1 or 3.5.0, in any bundle that holds both. soorat's does: hisab
arrives through goonj and ushma, and ranga directly (selah also depends on ranga).

## Summary

hisab exports `srgb_to_linear(c)`, `linear_to_srgb(c)`, `premultiply_alpha(r, g, b, a)` and
`unpremultiply_alpha(r, g, b, a)` with f64 / `HVec4` meanings. ranga exports the same four names with
byte / f32 / `PixelBuffer` meanings. In one compile (cyrius 6.7.18):
- the two alpha functions are an **error**: `duplicate fn 'premultiply_alpha' disagrees about arity: this one
  takes 4, the one in lib/ranga.cyr takes 1`;
- the two sRGB functions **silently swap**. With hisab's definitions last, ranga's `srgba_to_lin_rgba(128).r`
  returned **`0xa`** instead of `0x3e5d0a8b`, and the round trip gave 129 instead of 128. A ranga-only build
  gives `0x3e5d0a8b` and 128.

## What this asks of ranga

The rename is requested in **hisab**, whose four colour functions have no caller outside hisab, while
ranga's are documented public API (see the hisab issue). From ranga, two things:

1. **Correct the guidance in `src/spectral.cyr`.** It says a consumer that takes prakash-fft and hisab "must
   then rename around the collision", but a consumer cannot rename a dependency's functions. Point instead
   at the hisab request (or at whatever resolution lands), so the next consumer doesn't plan on a workaround
   that doesn't exist.
2. **Add a co-compile test once hisab renames.** One test that includes ranga's `dist/ranga.cyr` together
   with hisab's dist and checks `srgba_to_lin_rgba(128).r == 0x3e5d0a8b`. It keeps the collision from coming
   back from either side.

If hisab declines to rename, the fallback is a ranga-side prefix for these four, which is a breaking change
to ranga's documented API (its migration guide maps Rust's `ranga::color::srgb_to_linear` to the bare name).

## Measure / close

From `soorat/scripts/probes/bundle/`, `sh run.sh p11` compiles the full bundle. Close when it reports no
`duplicate fn` for these four names and ranga's colour answers are unchanged with hisab present.
