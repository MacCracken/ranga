# 008 — serde is JSON in serde_json's form, with correctly rounded numbers

**Status**: Accepted
**Date**: 2026-11-13

## Context

Rust 1.0.1 derived `Serialize` and `Deserialize` on 14 types: the 8 colour structs,
`PixelFormat`, `ColorSpace`, `BlendMode`, `ToneCurve`, `Affine` and `Perspective`.
The colour structs also carry `deny_unknown_fields`. serde itself is
format-agnostic. Cyrius has no serde, and its derive covers only an enum's variant
name, so the port had none of this: the parity review counted it as MISSING ×11.

Nothing in Rust's own tests or examples serialized these types. But a Rust consumer
persisting them would have used JSON, and the Cyrius stdlib's data-format library
(bayan) speaks JSON too.

## Decision

Each of the 14 types gets `<type>_to_json` and `<type>_from_json` (src/serde.cyr).
They reproduce **serde_json 1.0.151**, the version rust-old/Cargo.lock pins, with
serde-derive's defaults:

- **The wire form is serde's:**
  - structs are objects in field order, and also decode from serde's array form;
  - unit enums are their variant name (and also decode from `{"Name":null}`);
  - ToneCurve is externally tagged (`{"Gamma":2.2}`, `{"Table":[…]}`);
  - Perspective is `{"m":[[…],[…],[…]]}`.
- **Floats are written as serde_json writes them:** the shortest decimal that reads
  back to the same float, in the notation of serde_json's formatter, zmij
  (src/decimal.cyr). Non-finite floats are written `null`.
- **Decoding follows serde_json's control flow:**
  - its grammar, its whitespace and its string escapes;
  - its error order (a duplicate field is reported at its key, a missing one at the
    closing brace);
  - its type rules (`-0` and `1.0` are not u8s);
  - its range rule (`1e400` is a syntax error);
  - its iterative skip of unknown fields at any depth.

  Errors map to two codes: `RG_ERR_PARSE` for serde_json's Syntax and Eof, and
  `RG_ERR_INVALID_FORMAT` for its Data.
- **One deliberate difference: numbers decode correctly rounded.** serde_json's
  default float parser is best effort (its `float_roundtrip` feature is off), and an
  f32 field rounds that f64 a second time. ranga rounds each number once, straight
  to the field's type.
- **ColorSpace's two ranga variants** (CieLab, Oklab) are `"CieLab"` and `"Oklab"`.
  Rust has no such variants, so it cannot read them.

## Consequences

- **Positive**
  - The oracle checks this against serde_json itself, through a new `serde` group
    (scripts/oracle, with inputs from `make_serde_inputs.py`):
    - **encoding:** 3,525 values (edges and random bit patterns of every type) are
      byte-identical;
    - **decoding:** all 435 hand-written inputs get the same accept or reject and
      the same error category, and every u8, f32 and enum value is identical;
    - **long decimals:** 4,000 f32 decimals of 17 to 25 digits decode identically.
  - The golden digests carry that evidence once rust-old/ is gone.
  - ranga decodes 4,000 of 4,000 long f64 decimals correctly rounded, checked
    against Python's `float`. serde_json gets 2,782.
- **Negative**
  - **f64 values can differ from serde_json's by up to 2 ulp** where serde_json
    misrounds. That is 4 of 238 f64 values in the decode group (`1e-45` and `1e-50`
    among them) and 1,218 of 4,000 long decimals. The oracle accepts
    `serde_dec.f64` within 1 ulp and `serde_long.f64` within 2. A larger difference
    still fails.
  - The codecs are about 1,250 lines (src/serde.cyr) that serde gave Rust for free, and they have to
    follow serde_json if a later version changes its notation. That happened once
    already: zmij replaced ryu and added the `+` in `e+38`.
- **Neutral**
  - Other formats (TOML, bincode) are not provided. JSON is the one a consumer can
    exchange with the Rust line.

## Alternatives considered

- **Put serde out of scope.** Rejected by the maintainer. The surface is part of
  the Rust API, and a consumer persisting a grade or a transform needs it.
- **Use bayan's JSON parser and writer.** Its DOM parser reads a whole document
  before any type check, so a malformed tail would be reported in place of serde's
  earlier data error. Its float writer is Grisu2, which is not always shortest and
  is not zmij's notation. A schema-directed decoder was the only way to report
  serde_json's errors.
- **Reproduce serde_json's best-effort float parsing.** Rejected. Reproducing a
  rounding error bit for bit would mean porting its approximation. Correct rounding
  is what both serde_json's `float_roundtrip` mode and Python give.
