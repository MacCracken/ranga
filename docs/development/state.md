# ranga — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**1.0.1** — the final Rust release. The Rust→Cyrius port is in flight and lands
as **2.0.0** at M8, following the AGNOS precedent that a port bumps major
without resetting the series.

## Toolchain

- **Cyrius pin**: `6.5.27` (in `cyrius.cyml [package].cyrius`)

## Source

Rust reference: **13,958 lines** at `rust-old/` (frozen — the parity oracle, do
not edit).

Cyrius port: **7,639 lines** across 14 modules, bundled to
`dist/ranga.cyr` (7,624 lines).

| Module | Lines | Ported from |
| --- | ---: | --- |
| `error.cyr` | 149 | `error.rs` |
| `bytevec.cyr` | 217 | — (internalized, see plan §3) |
| `simd_u8.cyr` | 426 | — (hand-encoded SSE2 kernels) |
| `pixel.cyr` | 460 | `pixel.rs` |
| `color.cyr` | 895 | `color.rs` |
| `composite.cyr` | 712 | `composite.rs` |
| `histogram.cyr` | 517 | `histogram.rs` |
| `transform.cyr` | 827 | `transform.rs` |
| `filter_point.cyr` | 653 | `filter.rs` (split) |
| `filter_kernel.cyr` | 932 | `filter.rs` (split) |
| `blend.cyr` | 335 | `blend.rs` |
| `convert.cyr` | 483 | `convert.rs` |
| `icc.cyr` | 1,018 | `icc.rs` |
| `spectral.cyr` | 137 | `spectral.rs` (optional) |
| `hwaccel.cyr` | 209 | `hwaccel.rs` (optional) |
| `main.cyr` | 15 | — |

**Not yet ported** — the five `gpu/` modules (4,047 lines). See
[`roadmap.md`](roadmap.md) M6.

### Bundles

| Bundle | Lines | Contents |
| --- | ---: | --- |
| `dist/ranga.cyr` | 7,624 | core, no external deps |
| `dist/ranga-spectral.cyr` | 7,769 | core + `spectral.cyr` (needs prakash) |
| `dist/ranga-hwaccel.cyr` | 7,835 | core + `hwaccel.cyr` (needs ai-hwaccel) |

⚠ The two profile `.deps` sidecars are written empty by cycc 6.5.27 and must not
be trusted; `dist/ranga.deps` is authoritative for all three. Filed upstream.

## Tests

**1,517 assertions, 0 failures. 0 lint warnings, 0 untracked deferrals.**

| Suite | Assertions |
| --- | ---: |
| `icc.tcyr` | 169 |
| `spectral.tcyr` | 58 |
| `hwaccel.tcyr` | 48 |
| `composite.tcyr` | 159 |
| `filter_point.tcyr` | 150 |
| `histogram.tcyr` | 150 |
| `transform.tcyr` | 148 |
| `filter_kernel.tcyr` | 120 |
| `color.tcyr` | 108 |
| `pixel.tcyr` | 102 |
| `convert.tcyr` | 100 |
| `blend.tcyr` | 89 |
| `simd_u8.tcyr` | 65 |
| `ranga.tcyr` | 37 |

Every module is mutation-tested once it goes green; the discipline and its
findings are in [`port-mechanics.md`](port-mechanics.md). Expected values come
from Python oracles replicating the exact Rust formula, not from reading the
Cyrius back.

## Performance

35 CPU benchmarks in `tests/ranga.bcyr`, compared against the frozen Rust
criterion series in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md).

**Median 8.6x slower than Rust** (from 10.6x at the first pass), worst case
46.0x (from 79.9x). `blend_row_1920px` is 0.5x — faster than Rust. The
remaining gap is concentrated in the convolution filters, where Rust
auto-vectorises a loop Cyrius runs scalar; closing it needs a SIMD blur kernel,
which is deferred to 2.0.0 per the SIMD decision.

## Dependencies

Direct (declared in `cyrius.cyml`):

- stdlib — 25 leaves. The last ten (result, simd, fnptr, callback, bayan,
  sakshi, fs, process, thread, hashmap) are used by no core module; they are the
  leaves the two optional libraries need, and are declared here because that is
  where the resolver looks.
- **prakash 2.2.3** — optional, feature `spectral`
- **ai-hwaccel 2.3.17** — optional, feature `hwaccel`

Both are pinned by tag with no `path =` override, so the commit pin in
`cyrius.lock` (52 deps locked, 4 commit-pinned) actually verifies.

`mabda` (GPU) arrives at M6 the same way, so CPU-only consumers never compile it.

## Consumers

_None yet._

## Next

M6 — the five `gpu/` modules (4,047 lines) against mabda's **native** backends;
wgpu is the fallback. See [`roadmap.md`](roadmap.md).
