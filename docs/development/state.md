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
| `main.cyr` | 15 | — |

**Not yet ported** — `spectral.rs` (102), `hwaccel.rs` (164), and the five
`gpu/` modules. See [`roadmap.md`](roadmap.md) M5–M6.

## Tests

**1,409 assertions, 0 failures. 0 lint warnings, 0 untracked deferrals.**

| Suite | Assertions |
| --- | ---: |
| `icc.tcyr` | 169 |
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

- stdlib — syscalls, string, alloc, str, fmt, vec, io, args, assert, math,
  freelist, bench

`mabda` (GPU) arrives at M6 as a separate `[lib.gpu]` bundle, so CPU-only
consumers never compile it.

## Consumers

_None yet._

## Next

M5 — `spectral` (prakash covers 18/18 of the surface, verified) and `hwaccel`
(ai-hwaccel v2.3.16). See [`roadmap.md`](roadmap.md).
