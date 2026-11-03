# ranga — Current State

> Refreshed every release. CLAUDE.md is preferences/process/procedures
> (durable); this file is **state** (volatile).

## Version

**2.3.1**: closes two of the three items 2.3.0 left open.
- **Blur:** Rust's blur darkening is fixed in Cyrius, not kept. The separable blurs
  keep an f32 intermediate and round once
  ([ADR 003](../adr/003-blur-keeps-an-f32-intermediate-and-rounds-once.md)). They
  are unbiased (mean error 0.00 against −1.00 before, worst 0.50 against 1.94), a
  uniform image survives exactly, and they are 12% faster.
- **GPU benchmarks:** Rust's 26 are `tests/gpu.bcyr` and run on the dev host's
  Radeon Vega iGPU. Five print `MISSING` because their operations are not ported
  (2.5.x).
- **NVIDIA:** stays open, because the dev host has no NVIDIA card.

**2.3.0**: milestone 2.3.x, "verification parity". The port now carries Rust 1.0.1's
whole verification surface outside the GPU benchmarks: the 8 fuzz targets (`fuzz/`,
run by CI under allocator redzones), the 31 properties, all 123 integration,
edge-case, Photoshop, visual-regression and spectral tests, and the 108 CPU
benchmarks on Rust's workloads. The oracle runs in CI, and `tests/oracle_golden.tcyr`
holds Rust's outputs for its 53 exact groups, so the evidence survives `rust-old/`.
`src/` is unchanged. It found one inherited Rust defect, the blur darkening, which
2.3.1 fixed.

**2.2.0**: milestone 2.2.x, "bit-exact again". `scripts/oracle` checks the port
against the Rust 1.0.1 crate on identical inputs: 2.1 million values, and every
output boundary of the f32 → byte conversions over all of [0, 1]. After this
release every CPU computation the oracle covers matches Rust bit for bit. The
exceptions are three f32 fields that go through libm transcendentals, which differ
by ≤ 2 ULP and are accepted by ADR 002 (glibc's `sinf`/`cosf`/`powf` are not
correctly rounded). The six remaining SUSPECT parity items are all GPU (2.5.x).
`resize_nearest` is about 3× faster.

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

Cyrius port: **11,981 lines** across 21 modules plus `main.cyr`, bundled to
`dist/ranga.cyr`.

| Module | Lines | Ported from |
| --- | ---: | --- |
| `error.cyr` | 235 | `error.rs` |
| `bytevec.cyr` | 217 | — (internalized, see plan §3) |
| `simd_u8.cyr` | 429 | — (hand-encoded SSE2 kernels) |
| `pixel.cyr` | 509 | `pixel.rs` |
| `color.cyr` | 1,021 | `color.rs` |
| `composite.cyr` | 712 | `composite.rs` |
| `histogram.cyr` | 517 | `histogram.rs` |
| `transform.cyr` | 848 | `transform.rs` |
| `filter_point.cyr` | 681 | `filter.rs` (split) |
| `filter_kernel.cyr` | 993 | `filter.rs` (split) |
| `blend.cyr` | 351 | `blend.rs` |
| `convert.cyr` | 482 | `convert.rs` |
| `icc.cyr` | 1,079 | `icc.rs` |
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
| `dist/ranga.cyr` | 8,079 | core, no external deps |
| `dist/ranga-spectral.cyr` | 8,231 | core + `spectral.cyr` (needs prakash) |
| `dist/ranga-hwaccel.cyr` | 8,290 | core + `hwaccel.cyr` (needs ai-hwaccel) |
| `dist/ranga-gpu.cyr` | 11,603 | core + the six `gpu_*` modules (needs mabda 4.2.0) |

All four `.deps` sidecars are regenerated by `cyrius distlib`, which (since 6.6.20)
compile-verifies each one and trims it to the leaves that bundle needs: `ranga.deps`
is math, ganita, freelist and string (29 leaves through 2.0.3); the three profile
sidecars are three leaves each as of 6.7.18. Each bundle opens with a `# Requires`
block that includes those same leaves. The bundle bodies did not change at 2.0.4,
and at 2.1.0 (6.7.9) only the `# Version:` line moved. At 2.1.1 (6.7.18) the body
changes only by the `_TC_SIZE` → `_ICC_TC_SIZE` rename, and distlib drops `freelist`
from the hwaccel sidecar and `math` from the spectral one.

At 2.3.0 only the `# Version:` line moved: `src/` is unchanged. At 2.3.1 each bundle gains
20 lines, the blur passes' f32 intermediate (ADR 003).

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

**2,265 assertions across 23 suites, 0 failures, on an AMD host** (1,971 across 19 at
2.2.0). 162 of them are GPU assertions that run only when an AMD context opens:
22 in `gpu.tcyr`, 49 in `gpu_kernels.tcyr`, 83 in `gpu_pipeline.tcyr` and 8 in
`gpu_spirv.tcyr`. A host without one, such as CI, prints `SKIPPED: N hardware
assertions` for each suite and runs the other 2,103. On a GPU host each count is
asserted, so it cannot drift. The NVIDIA backend has no test (roadmap 2.3.x).

The suites now carry Rust 1.0.1's whole test surface: its in-module tests, all 123
integration tests (`integration.tcyr`, `edge_cases.tcyr`, `properties.tcyr`,
`spectral.tcyr`) and, through `oracle_golden.tcyr`, the oracle's Rust outputs for
its 53 exact groups (about 2.1 million values, checked by digest with no `cargo`).

| Suite | Assertions |
| --- | ---: |
| `icc.tcyr` | 199 |
| `transform.tcyr` | 165 |
| `filter_point.tcyr` | 163 |
| `composite.tcyr` | 159 |
| `histogram.tcyr` | 150 |
| `color.tcyr` | 148 |
| `filter_kernel.tcyr` | 138 |
| `pixel.tcyr` | 126 |
| `convert.tcyr` | 110 |
| `blend.tcyr` | 107 |
| `oracle_golden.tcyr` | 107 |
| `gpu_pipeline.tcyr` | 102 |
| `spectral.tcyr` | 66 |
| `simd_u8.tcyr` | 65 |
| `integration.tcyr` | 62 |
| `gpu_shaders.tcyr` | 59 |
| `ranga.tcyr` | 59 |
| `edge_cases.tcyr` | 55 |
| `gpu_spirv.tcyr` | 54 |
| `gpu_kernels.tcyr` | 53 |
| `hwaccel.tcyr` | 48 |
| `gpu.tcyr` | 46 |
| `properties.tcyr` | 24 |

`properties.tcyr` has one assertion per property group, each over an edge grid plus
4,096 seeded random inputs, so its count understates what it checks.

**Fuzzing:** `fuzz/` holds the 8 Rust fuzz targets as invariant-checking harnesses.
`cyrius fuzz --poison` runs them under allocator redzones in about 10 s, in CI too. A
failure prints the seed that replays it.

**Differential oracle:** `scripts/oracle/run.sh` compares 2.1 million values across 56
groups against the Rust 1.0.1 crate. It is clean apart from ADR 002's three accepted
groups, and CI runs it. ranga's own code compiles with zero warnings; the three left
come from the vendored prakash.

Every module is mutation-tested once it goes green; the discipline and its
findings are in [`port-mechanics.md`](port-mechanics.md). Expected values come
from Python oracles replicating the exact Rust formula, not from reading the
Cyrius back.

### Lint

**0 warnings and 0 untracked deferrals across all 58 files**: 22 in `src/`, 27 in
`tests/` (23 suites, `gpu_hw.cyr` and the three bench suites) and 9 in `fuzz/`. That is
the standing bar, and as of 2.0.1 CI actually enforces it.

It did not before. Up to cyrius 6.6.4, `cyrius lint` accepted a single path and
always exited 0, so the old step (`cyrius lint src/*.cyr tests/*.tcyr`) linted
`src/blend.cyr` and passed no matter what it found. From 2.0.1 to 2.1.1 the gate
looped over every file and parsed the printed totals. Since 2.1.2 it is one call,
`cyrius lint --strict --strict-deferrals src/*.cyr tests/*.tcyr tests/*.cyr tests/*.bcyr fuzz/*.fcyr fuzz/*.cyr`
(the last four globs since 2.3.0). At the 6.7.18 pin
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

**Review of 2026-11-02 (at 2.1.1), 303 Rust public items, updated for 2.1.2, 2.2.0 and 2.3.1:**

| Status | Count |
| --- | ---: |
| Parity | 208 |
| Improved on Rust | 8 |
| Divergent, documented | 29 |
| Partial | 19 |
| Missing | 33 |
| **Suspect** (computes differently from Rust, no ADR) | **6** |

The six suspect items are all GPU: half-up rounding in `gpu_fade`/`gpu_dissolve`, the
65,535-group ceiling in the flips and crop, and `gpu_error_message`. They are
scheduled in roadmap 2.5.x. 2.2.0 fixed all 18 CPU-side items, 16 to bit-exact parity
and 2 accepted by ADR 002. 2.1.2 had already fixed the zero-tag ICC case.

The WGSL fallback is never dispatched. The full inventory is in
[`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md), and the fixes are sequenced in
[`roadmap.md`](roadmap.md).

## Performance

All 134 of Rust 1.0.1's benchmarks, under Rust's names and on Rust's workloads:
`tests/ranga.bcyr` (100), `tests/spectral.bcyr` (8) and, since 2.3.1,
`tests/gpu.bcyr` (26). The comparison in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md) ran both sides on
one machine: the CPU suites in one session at 2.3.0, and the GPU suite on the dev
host's Radeon Vega iGPU at 2.3.1.

**Median 7.9× slower than Rust over the 108 CPU benchmarks, from 0.5× to 367×** (8.0×
at 2.3.0, before ADR 003 made the blurs 12% faster). 54 of the 108 are under 8×.
`blend_row` is 0.5×, faster than Rust. Most of the worst ranga-side ratios have one of
two causes. The 6.7.18 stdlib's `memcpy`/`memset` are byte loops: the buffer pool is
127×, crop 53×, `flip_vertical` 32×. And the per-byte f32 helper calls are not
inlined: fade 57×, dissolve 50×, the gradients around 40×. Both are in roadmap 2.6.x.
The 367× is prakash's `color_rendering_index`.

**GPU:** 21 of the 26 GPU benchmarks run; five print `MISSING` (blend, blur, noise,
bilinear resize, `GpuChain`). Every native GPU call takes about 30 ms at 1080p
whatever the kernel. mabda's buffer write and read are each one byte-loop `memcpy`,
moving 16 MB per call. Rust's wgpu path takes 2.4–3 ms on the same iGPU. This is the
same `memcpy` item (2.6.x).

The 2.2.0 figure (median 8.6× over 35 benchmarks) is not comparable: several of those
benchmarks ran different workloads from Rust's.

Cyrius benchmark history is recorded by `scripts/bench-history.sh` into
`bench-history.csv`, which runs all three suites. Each row carries the version, commit,
cyrius pin and timer-floor regime. Rows from before 2.3.0 measure the old workloads.

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

**2.3.x** has one item open: NVIDIA coverage, which needs NVIDIA hardware or an ADR.
Then come surface completion (2.4.x), GPU (2.5.x), performance
(2.6.x) and docs/audit/consumers (2.7.x). See [`roadmap.md`](roadmap.md).
