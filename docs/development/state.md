# ranga — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**2.0.1** — a maintenance release on top of the Cyrius port. The Rust line
ended at 1.0.1 and is preserved at `rust-old/` as the parity oracle, following
the AGNOS precedent that a port bumps major without resetting the series.
2.0.1 moves the toolchain pin and the optional-dependency pins; `src/` is
untouched.

## Toolchain

- **Cyrius pin**: `6.5.33` (in `cyrius.cyml [package].cyrius`)

The pin is the single source of truth — CI reads it out of the manifest and
hands it to the upstream installer, so bumping it here is what moves CI. Before
2.0.1 the pin said 6.5.31 while the local wrapper was already 6.5.33, and every
build printed a drift warning; the two now agree.

## Source

Rust reference: **13,958 lines** at `rust-old/src` (frozen — the parity oracle,
do not edit), of which `gpu/` is 4,047.

Cyrius port: **11,713 lines** across 21 modules plus `main.cyr`, bundled to
`dist/ranga.cyr`.

| Module | Lines | Ported from |
| --- | ---: | --- |
| `error.cyr` | 199 | `error.rs` |
| `bytevec.cyr` | 217 | — (internalized, see plan §3) |
| `simd_u8.cyr` | 429 | — (hand-encoded SSE2 kernels) |
| `pixel.cyr` | 496 | `pixel.rs` |
| `color.cyr` | 955 | `color.rs` |
| `composite.cyr` | 712 | `composite.rs` |
| `histogram.cyr` | 517 | `histogram.rs` |
| `transform.cyr` | 834 | `transform.rs` |
| `filter_point.cyr` | 673 | `filter.rs` (split) |
| `filter_kernel.cyr` | 941 | `filter.rs` (split) |
| `blend.cyr` | 347 | `blend.rs` |
| `convert.cyr` | 483 | `convert.rs` |
| `icc.cyr` | 1,044 | `icc.rs` |
| `spectral.cyr` | 145 | `spectral.rs` (optional) |
| `hwaccel.cyr` | 211 | `hwaccel.rs` (optional) |
| `gpu_spirv.cyr` | 656 | — (SPIR-V emitter; no Rust counterpart) |
| `gpu_kernels.cyr` | 887 | `gpu/shaders.rs` → SPIR-V (11 of 14 native) |
| `gpu_shaders.cyr` | 877 | `gpu/shaders.rs` → WGSL (all 21, wgpu fallback) |
| `gpu_pipeline.cyr` | 540 | `gpu/pipeline.rs` (14 ops; 11 native) |
| `gpu_context.cyr` | 402 | `gpu/context.rs` (optional) |
| `gpu_buffer.cyr` | 133 | `gpu/buffer.rs` (optional) |
| `main.cyr` | 15 | — |

All five `gpu/` modules are ported as of 2.0.0 — the "not yet ported" note that
stood here through M6 is retired. What remains outstanding is per-item, not
per-module; see [Parity](#parity).

### Bundles

Line counts are the ones `cyrius distlib` reports.

| Bundle | Lines | Contents |
| --- | ---: | --- |
| `dist/ranga.cyr` | 7,847 | core, no external deps |
| `dist/ranga-spectral.cyr` | 7,992 | core + `spectral.cyr` (needs prakash) |
| `dist/ranga-hwaccel.cyr` | 8,058 | core + `hwaccel.cyr` (needs ai-hwaccel) |
| `dist/ranga-gpu.cyr` | 11,342 | core + gpu context/buffer (needs mabda 4.1.0) |

All four `.deps` sidecars are correct as of the 6.5.33 pin, and are byte-for-byte
unchanged from 2.0.0. They were written empty for profile bundles through
6.5.27 — filed upstream and fixed there.

The 2.0.1 bundles differ from the 2.0.0 bundles in exactly one line each: the
`# Version:` header. `cyrius distlib --check` compares bytes, so a version bump
alone makes them stale and they must be regenerated with `cyrius distlib --all`
before the tag.

## Tests

**1,921 assertions across 19 suites, 0 failures.** Identical before and after
the 2.0.1 toolchain and dependency bumps.

| Suite | Assertions |
| --- | ---: |
| `icc.tcyr` | 177 |
| `filter_point.tcyr` | 161 |
| `composite.tcyr` | 159 |
| `transform.tcyr` | 151 |
| `histogram.tcyr` | 150 |
| `color.tcyr` | 139 |
| `filter_kernel.tcyr` | 124 |
| `pixel.tcyr` | 112 |
| `blend.tcyr` | 107 |
| `convert.tcyr` | 106 |
| `gpu_pipeline.tcyr` | 99 |
| `simd_u8.tcyr` | 65 |
| `gpu_shaders.tcyr` | 59 |
| `spectral.tcyr` | 58 |
| `ranga.tcyr` | 56 |
| `gpu_spirv.tcyr` | 53 |
| `gpu_kernels.tcyr` | 52 |
| `hwaccel.tcyr` | 48 |
| `gpu.tcyr` | 45 |

Every module is mutation-tested once it goes green; the discipline and its
findings are in [`port-mechanics.md`](port-mechanics.md). Expected values come
from Python oracles replicating the exact Rust formula, not from reading the
Cyrius back.

### Lint

**0 warnings and 0 untracked deferrals across all 41 files** — 22 in `src/`,
19 in `tests/`. That is the standing bar, and as of 2.0.1 CI actually enforces
it.

It did not before. `cyrius lint` accepts a single path and **always exits 0**,
reporting its counts rather than signalling through the exit status, so the old
step — `cyrius lint src/*.cyr tests/*.tcyr` — linted `src/blend.cyr`, discarded
the other 40 arguments, and passed no matter what it found. The gate now loops
every file and fails on the reported totals.

Two quirks worth knowing before touching that step:

- cyrlint prints `N warnings` on **stdout** but `N untracked deferrals` on
  **stderr**. A capture without `2>&1` loses the deferral count entirely —
  this was caught only because the gate defaults an unparsed count to
  *dirty* rather than *clean*. Keep that default.
- The deferral rule matches a term (`TODO`, `not yet`, `for now`,
  `NOT_IMPLEMENTED`, `deferred`, …) and then looks for a tracking pointer
  **on the same line**: `CHANGELOG`, `roadmap`, `docs/`, `issue`, `See `,
  `v5.`, `v6.`. Note that `cannot yet` contains `not yet`.

Turning the gate on surfaced 10 warnings and 2 untracked deferrals that had
accumulated in `tests/`; all are fixed in 2.0.1 and itemised in the CHANGELOG.
The one that was more than cosmetic: `tests/pixel.tcyr` declared `exact` and
`big` twice each at file scope, and Cyrius binds a name to the **last**
declaration — the values happened to be right because the earlier `var` wrote
the same global before the read, but the shadowing was real. They are now
`view_exact` and `pool_big`.

## Parity

**318 Rust public items: ~250 ported, 22 deliberate omissions, ~23 missing.**
`gpu/pipeline.rs` — 28 of the original 51 gaps — has since been ported, so the
"Where the 51 gaps are" table in
[`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md) reads as of the audit that
wrote it, not as of today.

## Performance

35 CPU benchmarks in `tests/ranga.bcyr`, compared against the frozen Rust
criterion series in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md).

**Median 8.6x slower than Rust** (from 10.6x at the first pass), worst case
46.0x (from 79.9x). `blend_row_1920px` is 0.5x — faster than Rust. The
remaining gap is concentrated in the convolution filters, where Rust
auto-vectorises a loop Cyrius runs scalar; closing it needs a SIMD blur kernel.
Not re-measured for 2.0.1 — `src/` did not change.

## Dependencies

Direct (declared in `cyrius.cyml`):

- **stdlib — 29 leaves.** The last 13 (simd, fnptr, callback, bayan, sakshi,
  fs, process, thread, hashmap, mmap, dynlib, sankoch, thread_local) are used
  by no core module; they are the leaves the optional libraries need, and are
  declared here because that is where the resolver looks.
- **prakash 2.2.8** — optional, feature `spectral` (was 2.2.3)
- **ai-hwaccel 2.3.18** — optional, feature `hwaccel` (was 2.3.17)
- **mabda 4.1.0** — optional, feature `gpu`; already the newest tag

All three are pinned by tag with no `path =` override, so the commit pin in
`cyrius.lock` actually verifies. The lock records **59 deps locked, 7
commit-pinned** (up from 52 and 4), and `cyrius deps --verify` reports
59 verified, 0 failed.

Transitive pins that moved with the direct bumps: **hisab 2.11.1 → 2.11.2**,
**sakshi 2.4.10 → 2.4.11**. samvada 0.4.1 and chitra 0.3.1 are unchanged.

> ⚠ The old hisab pin was **unresolvable**. `cyrius.lock` recorded hisab
> `2.11.1`, a tag that has since been retracted upstream and now 404s, so a
> clean `cyrius deps` on a fresh checkout could not have fetched it. It went
> unnoticed because `lib/hisab.cyr` was already vendored and tracked. Worth
> remembering that a green local tree does not prove the lock is fetchable.

Re-vendoring the stdlib subset from the 6.5.33 snapshot left every existing
leaf byte-identical and added one new file, **`lib/hashmap_fast.cyr`**. It is
hash-locked and verified but is not a declared leaf, does not appear in any
`.deps` sidecar, and nothing in `src/` or `lib/` includes it — `src/simd_u8.cyr`
only cites it in a comment as the in-tree precedent for hand-encoded `asm { }`.
It is inert; it is tracked so the lock verifies.

## Consumers

_None yet._

## Next

Per-item parity cleanup against
[`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md), and the SIMD blur kernel
that closes the convolution benchmark gap. See
[`roadmap.md`](roadmap.md).
