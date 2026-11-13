# 007 — `.cube` domains are honoured, and Resolve's INPUT_RANGE is read

**Status**: Accepted
**Date**: 2026-11-13

## Context

`Lut3d::from_cube` (rust-old/src/filter.rs:892) parses the Adobe Cube LUT format.
Rust 1.0.1 handled two kinds of line in a way that is wrong for real files:

- **`DOMAIN_MIN` / `DOMAIN_MAX` were skipped.** The format uses them to give the
  input range the table spans, per channel, defaulting to [0, 1]. Rust's `Lut3d`
  had no domain field, and its lookup assumed [0, 1]. A LUT built for [0, 2] (or
  any log-encoded range) therefore loaded without complaint, and every lookup
  through it was wrong: an input of 1.0 read the top of the table instead of its
  middle.
- **DaVinci Resolve's `LUT_3D_INPUT_RANGE min max` rejected the whole file.** It has
  three tokens, so Rust's data-line rule tried to parse `LUT_3D_INPUT_RANGE` as an
  f32 and returned `RangaError::Other("bad LUT value")`.

The port's rule is that Rust's behaviour is not correct by default. A Rust defect is
fixed in Cyrius, not reproduced, and recorded here.

## Decision

The domain is read and honoured. Everything else in `from_cube` is Rust's.

- **`Lut3d` gains a per-channel domain**, f32 min and max. `lut3d_new` sets the
  default, [0, 1]. `lut3d_domain_min/max` read it and `lut3d_set_domain` writes it.
  The format requires min < max, both finite; anything else is
  `RG_ERR_INVALID_PARAMETER`.
- **Lookup maps each input through `(v - min) / (max - min)` first**, in f32, and
  then clamps as before. That is the format's meaning, and inputs outside the
  domain clamp to its ends. For the default domain the map is skipped, by a flag
  cached in the record. So a [0, 1] LUT looks up bit for bit as Rust's did, and costs
  nothing extra per pixel.
- **`DOMAIN_MIN r g b` / `DOMAIN_MAX r g b` are parsed** into that domain.
  `LUT_3D_INPUT_RANGE min max` sets all three channels.
- **Malformed domain lines are errors (`RG_ERR_OTHER`):** fewer values than the
  keyword takes, a value that is not an f32, or a channel that ends with min ≥ max
  or a non-finite bound. Rust skipped such lines, which left the LUT silently
  mis-applied. Refusing them is the safer outcome.
- **A prefix that is not the keyword stays Rust's.** `DOMAIN_MINX …` is still
  skipped by Rust's prefix rule, and `LUT_3D_INPUT_RANGEX …` is still a bad data
  line.

## Consequences

- **Positive**
  - Files with a non-default domain apply correctly.
  - Resolve's legacy files load.
  - A file without a domain line, or with the default one, parses to exactly the LUT
    Rust built. `tests/cube.tcyr` checks this bit for bit over 64 inputs.
- **Negative**
  - `Lut3d` grew from 16 to 48 bytes. `lut3d_clone` copies the domain.
  - A file Rust loaded with a malformed `DOMAIN` line is refused. Such a LUT was
    being applied wrongly anyway.
- **Neutral**
  - `lut3d_new` + `lut3d_set` LUTs default to [0, 1], so code that builds LUTs by
    hand sees no change.

## Alternatives considered

- **Reproduce Rust: skip the domain.** Rejected. It produces wrong colours without
  any diagnostic, which is the failure this port exists to avoid.
- **Refuse any non-default domain.** That would be safe, but it would refuse valid,
  common files that the format defines precisely. Honouring the domain is a
  division per channel.
- **Support only `DOMAIN_*`, not `LUT_3D_INPUT_RANGE`.** Rejected. It is the same
  information in Resolve's older spelling, and refusing it rejects files a colourist
  will have.
