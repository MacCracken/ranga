# 006 — bytevec is `rg_bv_*`, and xorshift64 is private

**Status**: Accepted
**Date**: 2026-11-13

## Context

Cyrius has one flat namespace. When two definitions share a name, the last one wins
and the compiler prints a warning, nothing more. ranga has already been bitten by this
twice:
- the `RG_GPU_ERR_*` codes are prefixed because mabda's unprefixed `GPU_ERR_*` would
  have flipped their signs;
- including hisab would have replaced composite.cyr's `premultiply_alpha(buf)` with
  hisab's four-argument one.

Two parts of ranga's public surface did not follow the module-prefix convention the
rest of the port uses (`pixel_buffer_*`, `affine_*`, `icc_*`):

- **`bytevec.cyr`'s 19 functions were bare `bv_*`.** `bv_` is a natural name for any
  byte vector, so a consumer, or a library a consumer vendors, could define `bv_len`
  or `bv_free`. If it did, the warning would be the only signal that ranga's pixel
  buffers now run on someone else's code. No vendored library defines `bv_*` today.
- **`xorshift64_*` (5 functions) was public in the port but private in Rust**, where
  `xorshift64` is a plain `fn` inside `filter.rs`. Exporting it committed ranga to
  a PRNG's name, signature and sequence that Rust never promised. The noise filters
  take a seed, and that seed is the whole reproducibility contract.

ranga has no Cyrius consumer yet, so a rename costs nothing downstream. It will never
be cheaper.

## Decision

- **`bv_*` becomes `rg_bv_*`**, all 19 functions, with no aliases. The bytevec layout
  and behaviour are unchanged.
- **`xorshift64_*` becomes `_fk_xorshift64_*`**, private under `filter_kernel.cyr`'s
  `_fk_` prefix, matching Rust. The tests keep white-box access, as they have to
  every module's internals.

## Consequences

- **Positive**
  - A consumer's `bv_*` cannot replace ranga's buffer primitive.
  - ranga's public surface now has a prefix on every name: `rg_bv_` for bytevec, the
    type prefix elsewhere.
  - The PRNG can change (for speed, or for a better generator) without breaking any
    public API. The noise filters' output is still pinned by the tests and the
    oracle.
- **Negative**
  - It is a breaking rename of 24 public names. A consumer written against 2.3.x must
    rename its `bv_*` calls (mechanically: `\bbv_` → `rg_bv_`), and any use of
    `xorshift64_*` must move to its own PRNG. There are no known consumers.
- **Neutral**
  - All four bundles are regenerated with the new names.
  - The parity doc's two namespace items are resolved.

## Alternatives considered

- **Keep `bv_*` until something collides.** Rejected: the collision would arrive as a
  warning in a consumer's build log, after the fact, and would by then be a breaking
  change for that consumer.
- **Keep the old names as aliases for a release.** Rejected: an alias is a second
  public definition of the same name, and the old names would still be in the global
  namespace, which is the problem being removed.
- **Keep `xorshift64_*` public and document it.** Rejected: it would make permanent a
  commitment Rust deliberately did not make, for a function no caller needs.
