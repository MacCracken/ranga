# Architecture Decision Records

Decisions about ranga — what we chose, the context, and the consequences we accept. Use these when a future reader would reasonably ask *"why did we do it this way?"*

## Conventions

- **Filename**: `NNNN-kebab-case-title.md`, zero-padded to four digits. Never renumber.
- **One decision per ADR.** If a decision supersedes a prior one, add a new ADR and set the old one's status to `Superseded by NNNN`.
- **Status lifecycle**: `Proposed` → `Accepted` → (optionally) `Superseded` or `Deprecated`.
- Use [`template.md`](template.md) as the starting point.

## ADR vs. architecture note vs. guide

| Kind | Lives in | Answers |
|---|---|---|
| ADR | `docs/adr/` | *Why did we choose X over Y?* |
| Architecture note | `docs/architecture/` | *What non-obvious constraint is true about the code?* |
| Guide | `docs/guides/` | *How do I do X?* |

## Index

- [001 — The YUV→RGB inverse does not reproduce Rust's i16 wrap](001-yuv-inverse-does-not-reproduce-rusts-i16-wrap.md)
- [002 — f32 transcendentals follow correct rounding, not the oracle host's libm](002-f32-transcendentals-follow-correct-rounding-not-glibc.md)
- [003 — The separable blurs keep an f32 intermediate and round once](003-blur-keeps-an-f32-intermediate-and-rounds-once.md)
- [004 — `bilateral` rounds its weighted mean](004-bilateral-rounds-its-weighted-mean.md)
- [005 — GPU compute runs only on the native AMD backend](005-gpu-compute-runs-only-on-native-amd.md)
- [006 — bytevec is `rg_bv_*`, and xorshift64 is private](006-bytevec-is-rg-bv-and-xorshift-is-private.md)
- [007 — `.cube` domains are honoured, and Resolve's INPUT_RANGE is read](007-cube-domain-is-honoured.md)
- [008 — serde is JSON in serde_json's form, with correctly rounded numbers](008-serde-codecs-are-json-in-serde-json-form.md)

The first two ADRs use three-digit numbers. New ones continue that series (`003-…`).
