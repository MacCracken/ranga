# ranga — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**2.1.2**: closes out the toolchain bumps with a P(-1) review
([`docs/audit/2026-11-02-audit.md`](../audit/2026-11-02-audit.md)). It fixes:
- two ICC parse regressions caused by cyrius 6.6.7's `fl_alloc(0)` returning 0;
- a trig guard the toolchain had outgrown, which made `affine_rotate(1e20)` NaN;
- the last compiler warnings in ranga's own code.

It also moves prakash to 2.15.1, removes two unused lib files, rewrites the bench
script for Cyrius and collapses the CI lint gate into one strict call. A
cross-toolchain diff shows that the bumps changed no ranga byte output except
one, which they fixed (a 1e25° gradient), plus 1–2 ULP of ICC `pow` churn.

**2.1.1**: toolchain `6.7.9` → `6.7.18`, with the optional deps at their latest
releases: prakash 2.15.0, ai-hwaccel 2.5.6 and mabda 4.2.0. One source change was
required: `src/icc.cyr`'s private `_TC_SIZE` is renamed `_ICC_TC_SIZE`, because the
6.7.18 stdlib `lib/bayan.cyr` declares a const of the same name and 6.7.18 refuses
the clash. No public name changed. The compiler warnings are the same set as 2.1.0.

**2.1.0**: toolchain `6.6.20` → `6.7.9`, the first pin on the 6.7 language minor.
The optional deps moved to the releases that pin that line: prakash 2.11.0 and
mabda 4.2.0. ai-hwaccel stays at 2.4.2 because it has no release between 6.6.20
and 6.7.9. `src/` and `tests/` are untouched, and the compiler warnings are the
same set as 2.0.4.

**2.0.4**: toolchain `6.6.9` → `6.6.20`, and the optional deps moved to the
releases that pin that line: prakash 2.10.1, ai-hwaccel 2.4.2 and mabda 4.1.7 (mabda
has no 6.6.20 release; 4.1.7 pins 6.6.18). `src/` and `tests/` are untouched. One new
compiler warning appears, at `src/blend.cyr:143` (see the CHANGELOG).

**2.0.3** — toolchain `6.6.2` → `6.6.9`, and one test input corrected: the negative
fade factor in `tests/gpu_kernels.tcyr` was written `0 - 1.0`, an INTEGER subtraction
of 1.0's bit pattern that passed -4.0 where the row (and its message) meant -1.0. It is
`-1.0` now (unary minus flips the sign bit of a float literal since cyrius 6.6.8). The
row's expectations are unchanged: every negative factor clamps to zero. `src/` is
untouched. (2.0.2 moved the pin 6.5.33 → 6.6.2 and migrated to the `Result` value form.)

**2.0.1** — a maintenance release on top of the Cyrius port. The Rust line
ended at 1.0.1 and is preserved at `rust-old/` as the parity oracle, following
the AGNOS precedent that a port bumps major without resetting the series.
2.0.1 moves the toolchain pin and the optional-dependency pins; `src/` is
untouched.

## Toolchain

- **Cyrius pin**: `6.7.18` (in `cyrius.cyml [package].cyrius`, since 2.1.1)

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
| `gpu_shaders.cyr` | 877 | `gpu/shaders.rs` → WGSL (all 21; nothing dispatches them, see Parity) |
| `gpu_pipeline.cyr` | 540 | `gpu/pipeline.rs` (14 ops; 11 native; no `GpuChain`) |
| `gpu_context.cyr` | 402 | `gpu/context.rs` (optional) |
| `gpu_buffer.cyr` | 133 | `gpu/buffer.rs` (optional) |
| `main.cyr` | 15 | — |

Every `gpu/` file has a Cyrius counterpart, but `pipeline.rs` is not fully ported.
`GpuChain` is missing, and `gpu_blend`, `gpu_noise_gaussian` and `gpu_gaussian_blur`
are stubs. See [Parity](#parity).

### Bundles

Line counts are the ones `cyrius distlib` reports.

| Bundle | Lines | Contents |
| --- | ---: | --- |
| `dist/ranga.cyr` | 7,875 | core, no external deps |
| `dist/ranga-spectral.cyr` | 8,027 | core + `spectral.cyr` (needs prakash) |
| `dist/ranga-hwaccel.cyr` | 8,086 | core + `hwaccel.cyr` (needs ai-hwaccel) |
| `dist/ranga-gpu.cyr` | 11,399 | core + the six `gpu_*` modules (needs mabda 4.2.0) |

All four `.deps` sidecars are regenerated by `cyrius distlib`, which (since 6.6.20)
compile-verifies each one and trims it to the leaves that bundle needs: `ranga.deps`
is math, ganita, freelist and string (29 leaves through 2.0.3); the three profile
sidecars are three leaves each as of 6.7.18. Each bundle opens with a `# Requires`
block that includes those same leaves. The bundle bodies did not change at 2.0.4,
and at 2.1.0 (6.7.9) only the `# Version:` line moved. At 2.1.1 (6.7.18) the body
changes only by the `_TC_SIZE` → `_ICC_TC_SIZE` rename, and distlib drops `freelist`
from the hwaccel sidecar and `math` from the spectral one.

`cyrius distlib` warns that `num_fft` is still undefined in `ranga-spectral.cyr`.
The name comes from prakash (6.7.9 reported it too), so a consumer of that bundle
must supply it.
The sidecars were written empty for profile bundles through 6.5.27, a bug that was
filed upstream and fixed there.

The 2.0.1 bundles differ from the 2.0.0 bundles in exactly one line each: the
`# Version:` header. `cyrius distlib --check` compares bytes, so a version bump
alone makes them stale and they must be regenerated with `cyrius distlib --all`
before the tag.

## Tests

**1,939 assertions across 19 suites, 0 failures, on an AMD host.** That is the 1,921
that held through the 2.0.1–2.1.1 bumps, plus 18 added by the 2.1.2 P(-1) review,
each of which fails on 2.1.1. ranga's own code compiles with zero warnings; the
three left come from the vendored prakash. 152 GPU
assertion sites run only when an AMD context opens. CI has no GPU, so it runs far
fewer, and nothing on CI says how many were skipped.

**There is no fuzzing.** `tests/ranga.fcyr` is the `cyrius port` scaffold stub and
includes no ranga module, and none of the 8 Rust fuzz targets is ported. There are no
property tests either. See [roadmap.md](roadmap.md) 2.3.x.

| Suite | Assertions |
| --- | ---: |
| `icc.tcyr` | 190 |
| `filter_point.tcyr` | 161 |
| `composite.tcyr` | 159 |
| `transform.tcyr` | 153 |
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
| `ranga.tcyr` | 59 |
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

It did not before. Up to cyrius 6.6.4, `cyrius lint` accepted a single path and
always exited 0, so the old step (`cyrius lint src/*.cyr tests/*.tcyr`) linted
`src/blend.cyr` and passed no matter what it found. From 2.0.1 to 2.1.1 the gate
looped over every file and parsed the printed totals. Since 2.1.2 it is one call,
`cyrius lint --strict --strict-deferrals src/*.cyr tests/*.tcyr`. At the 6.7.18 pin
that exits 2 on any warning or untracked deferral, while the plain form exits 0.

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

**Review of 2026-11-02 (at 2.1.1), 303 Rust public items, updated for the 2.1.2 fixes:**

| Status | Count |
| --- | ---: |
| Parity | 194 |
| Improved on Rust | 5 |
| Divergent, documented | 28 |
| Partial | 19 |
| Missing | 33 |
| **Suspect** (computes differently from Rust, no ADR) | **24** |

What the 24 suspect items are:
- the f64-instead-of-f32 colour and filter paths;
- `affine_inverse`'s reassociation;
- ICC grid size 1 being rejected;
- GPU rounding and size ceilings.

2.1.2 fixed the zero-tag ICC case. The cross-toolchain diff confirmed that every
remaining suspect item predates the 6.6.9 → 6.7.18 bumps.

The WGSL fallback is never dispatched. The full inventory is in
[`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md), and the fixes are sequenced in
[`roadmap.md`](roadmap.md).

## Performance

35 CPU benchmarks in `tests/ranga.bcyr`, compared against the frozen Rust
criterion series in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md).

**Median 8.6x slower than Rust** (from 10.6x at the first pass), worst case
46.0x (from 79.9x). `blend_row_1920px` is 0.5x — faster than Rust. The blurs
(~21x) are slow because Rust auto-vectorises a loop Cyrius runs scalar.
`equalize` (19.5x) and `auto_levels` (23x) are about as slow, and in their case the
cause is per-channel call overhead.

Cyrius benchmark history is recorded by `scripts/bench-history.sh` into
`bench-history.csv` (since 2.1.2). Each row carries the version, commit, cyrius pin
and timer-floor regime. `noise_gaussian_1080p` runs one iteration and swings ±50%
between runs, so do not read a single row of it as a trend.

`benchmarks-rust-v-cyrius.md` is stale: its table still shows `blend_row` at 12.8x,
and its Cyrius source-size column is empty. Only 35 of Rust's 134 benchmarks have a
Cyrius counterpart.

## Dependencies

Direct (declared in `cyrius.cyml`):

- **stdlib — 31 leaves.** The last 15 (simd, fnptr, callback, bayan, sakshi,
  fs, process, thread, hashmap, mmap, dynlib, sankoch, thread_local, sync,
  chrono) are used by no core module. They are the leaves the optional libraries
  need, and are declared here because that is where the resolver looks. `sync`
  (ai-hwaccel 2.4.2) and `chrono` (mabda 4.1.7) were added in 2.0.4.
- **prakash 2.15.1**: optional, feature `spectral` (was 2.15.0; pins 6.7.18)
- **ai-hwaccel 2.5.6**: optional, feature `hwaccel` (was 2.4.2; pins 6.7.18). Its
  sidecar also lists `atomic`, which is not declared here: the ai-hwaccel bundle
  includes `lib/atomic.cyr` itself, and `cyrius deps` asks for nothing more.
- **mabda 4.2.0**: optional, feature `gpu` (pins 6.7.5); still the newest release

All three are pinned by tag with no `path =` override, so the commit pin in
`cyrius.lock` actually verifies. The lock records **69 deps locked, 7
commit-pinned**, and `cyrius deps --verify --features spectral,hwaccel,gpu`
reports 69 verified, 0 failed. The 6.7.18 `lib/bayan.cyr` brought in
`lib/unicode/` (7 files).

Transitive pins that moved with the direct bumps in 2.1.1: **hisab 3.4.0 → 3.4.1**,
**sakshi 2.5.8 → 2.5.10**. samvada 1.0.1 and chitra 1.0.3 are unchanged. (In 2.1.0:
hisab 3.3.7 → 3.4.0, sakshi 2.5.7 → 2.5.8. In 2.0.4: hisab 2.11.2 → 3.3.7, samvada
0.4.1 → 1.0.1, chitra 0.3.1 → 1.0.3, sakshi 2.4.11 → 2.5.7.)

> In 2.1.1, **`lib/flags.cyr`** and **`lib/hashmap_fast.cyr`** were removed. 6.7.18's
> `cyrius deps` reports both as outside the declared footprint and no longer vendors
> them, and nothing includes either one. `rm <file> && cyrius deps --lock` dropped
> them, and the lock records **69 deps locked** (69 verified, 0 failed).

> ⚠ After a tag bump, `cyrius deps` (6.6.20) keeps the old commit pins in
> `cyrius.lock` next to the new ones instead of replacing them. In 2.0.4 that left
> 14 `commit` lines for 7 deps. Deleting the lock and resolving again with
> `--features spectral,hwaccel,gpu` produces a clean lock.

> ⚠ The old hisab pin was **unresolvable**. `cyrius.lock` recorded hisab
> `2.11.1`, a tag that has since been retracted upstream and now 404s, so a
> clean `cyrius deps` on a fresh checkout could not have fetched it. It went
> unnoticed because `lib/hisab.cyr` was already vendored and tracked. Worth
> remembering that a green local tree does not prove the lock is fetchable.

## Consumers

_None yet._ The four named candidates (rasa, tazama, aethersafta, soorat) are
still Rust crates on the Rust ranga.

## Next

**2.2.x, bit-exact again:** check in the differential oracle, then fix the 24 suspect
items. Next are verification parity (2.3.x), surface completion (2.4.x), GPU (2.5.x),
performance (2.6.x) and docs/audit/consumers (2.7.x). See [`roadmap.md`](roadmap.md).
