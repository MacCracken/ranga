# Changelog

## [Unreleased]

## [2.3.2] - 2026-11-02

`bilateral` no longer darkens: it rounds its weighted mean, the same class of fix as
2.3.1's blurs. And NVIDIA was checked on real hardware for the first time: a GTX 1660
SUPER (TU116, SM75) on nouveau, which is exactly the target mabda's NVIDIA backend
names. Context creation and buffers work. Compute does not, and the cause is upstream.

### Changed

- **`bilateral` rounds its weighted mean**
  ([ADR 004](docs/adr/004-bilateral-rounds-its-weighted-mean.md)). Rust 1.0.1 and
  the port up to 2.3.1 truncated it with `as u8`. Measured on 40,824 channel values
  against the exact weighted mean:

  | | Mean error | Worst error |
  |---|---:|---:|
  | Before | −0.47 | 1.00 |
  | Now | 0.00 | 0.50 |

  A uniform image used to come back one count darker in 1,423 of 3,072 measured
  cases, and now in none. The weights are unchanged (f32, as in Rust), and the speed
  is unchanged (141 ms against 140 ms). Against Rust, values move by 0 or +1, never
  down.
- **The oracle accepts `bilateral` within 1 count** (`compare.py`, ADR 004);
  `bilateral.code` stays exact. `golden_rust.cyr` is regenerated and skips it, so
  `tests/oracle_golden.tcyr` checks 52 exact groups, down from 53 (105 assertions,
  down from 107). The total stays at 2,265.
- **`fuzz_blur`** asserts that bilateral leaves a uniform image exactly unchanged,
  replacing Rust's [v − 1, v]. The unused `within()` helper is gone.
- **`tests/filter_kernel.tcyr`**: two new tests, 140 assertions in all, up from 138.
  Both fail on 2.3.1:
  - every value 0..255 at radii 1..3 with two sigma pairs;
  - a pin on the rounding: the middle of (0, 0, 96) with both sigmas at 1000 is
    31.706, which rounds to 32 where Rust gave 31.

### Found: NVIDIA compute is blocked upstream, and ranga makes it fault

The four GPU suites were run against `gpu_context_new_nvidia` on a GeForce GTX 1660
SUPER (nouveau kernel driver, Mesa NVK) by building them here and copying the static
binaries over:

- **What works:** the context opens and buffer upload and download round-trip. 45 of
  46 in `gpu.tcyr` pass; the 46th is the AMD-only backend-name check.
- **What doesn't:** every compute dispatch fails. mabda 4.2.0's NVIDIA backend takes
  **precompiled SM75 SASS**, not SPIR-V; its SPIR-V→SASS compiler is mabda's
  unshipped phase N9. Its dispatch also binds a single output buffer, not a binding
  array.
- **ranga makes it worse.** ranga hands its SPIR-V words over, mabda uploads them
  verbatim, and the GPU executes them as machine code. nouveau then kills the
  channel (`Xid:13 Graphics SM Warp Exception … Out Of Range Register`). The driver
  recovered each time.
- **Two test helpers segfault** once a dispatch has failed. They read through the
  null buffer that comes back, instead of failing an assertion.

Nothing in `src/gpu_*` changed in this release. Guarding NVIDIA contexts and making
the helpers fail cleanly are next (roadmap 2.3.x).

## [2.3.1] - 2026-11-02

Closes two of the three items 2.3.0 left open. Rust's blur darkening is **fixed, not
kept**: the separable blurs now keep an f32 intermediate and round once, so they are
unbiased and 12% faster. Rust's 26 GPU benchmarks are ported and run on the AMD dev
host's Radeon Vega iGPU. NVIDIA coverage stays open, because the dev host has no
NVIDIA card.

### Changed

- **`gaussian_blur`, `box_blur` and `unsharp_mask` no longer darken**
  ([ADR 003](docs/adr/003-blur-keeps-an-f32-intermediate-and-rounds-once.md)).
  Rust 1.0.1, and the port up to 2.3.0, truncated the horizontal pass to `u8` and
  truncated again after the vertical pass. The new code works as follows:
  - the horizontal pass writes three f32 per pixel;
  - the vertical pass reads them back and rounds once, half up;
  - alpha is still copied from the source.

  Measured on 65,328 channel values against an exact f64 convolution:

  | | Mean error before | Mean error now | Worst error before | Worst error now |
  |---|---:|---:|---:|---:|
  | gaussian | −1.00 | 0.00 | 1.94 | 0.50 |
  | box | −0.88 | 0.00 | 1.85 | 0.50 |

  A uniform image now survives exactly at every value and radius. On 1080p, gaussian
  and box r3 drop from 408 ms to 358 ms and unsharp r2 from 475 ms to 430 ms, because
  the vertical pass no longer converts a byte to f32 at every tap. Against Rust,
  values move by 0 to +2 and never down. The scratch buffer is 12 bytes per pixel
  instead of 4.
- **`fuzz_blur`** asserts that gaussian, box and unsharp leave a uniform image exactly
  unchanged. It used to allow Rust's [v − 2, v].
- **`tests/filter_kernel.tcyr`:**
  - The two tests that pinned Rust's 98 assert 100.
  - New: every value 0..255 at radii 1..10 through both blurs.
  - New: a point-source test that pins the single rounding (centre 158 and diagonal
    3, where Rust gave 157 and 2).
  - 138 assertions, up from 134.
- **The roadmap's rule for parity items:** a Rust defect is fixed in Cyrius and
  recorded in an ADR, not reproduced. `rust-old/` shows what is missing and catches
  porting mistakes; it does not make Rust's behaviour correct.

### Added

- **`tests/gpu.bcyr`: Rust's 26 GPU benchmarks** (`rust-old/benches/gpu.rs`), with
  Rust's names, inputs and parameters, and each GPU operation beside its CPU
  counterpart.
  - Five operations are not ported yet, so they print `MISSING:` with the reason
    instead of a time: `gpu_blend`, `gpu_gaussian_blur`, `gpu_noise_gaussian`,
    bilinear `gpu_resize` and `GpuChain`. The first four are probed with a real call,
    and an operation that stops returning UNSUPPORTED is timed automatically.
  - Without a GPU, the suite prints one SKIPPED line and runs the CPU half.
- **`scripts/bench-history.sh`** runs `tests/gpu.bcyr` whenever `lib/mabda.cyr` is
  vendored, so there are three default suites.

### Found

- **A native GPU call costs about 30 ms at 1080p whatever the kernel.**
  - mabda's native `gpu_buffer_write` and `gpu_buffer_read` are each one stdlib
    `memcpy`, which is a byte loop at 6.7.18. A one-buffer operation copies 16 MB,
    and a two-buffer one 24 MB (about 45 ms).
  - Rust's wgpu path does the same operations in 2.4–3 ms on the same iGPU.
  - This is the 2.6.x `memcpy` item. The fix is upstream, and a resident-buffer
    `GpuChain` (2.5.x) would avoid most of the copies.
- **`bilateral` truncates after its weighted division,** so a uniform image can come
  back one count darker, as in Rust. This is separate code from the separable blurs
  and is not changed here.

## [2.3.0] - 2026-11-02

Milestone 2.3.x, "verification parity". Rust 1.0.1 was checked by 8 fuzz targets, 31
properties, Photoshop reference pixels, integration pipelines, edge-case tests and 134
benchmarks. Up to 2.2.0 the port had a scaffold fuzz stub, no property tests and 35
benchmarks. It now has all 8 fuzz targets (run in CI under allocator redzones), all 31
properties, every Rust integration and edge-case test, and 108 benchmarks under Rust's
names and workloads. The differential oracle runs in CI, and its Rust outputs are
captured in `tests/`, so they survive the deletion of `rust-old/`. `src/` is
unchanged: nothing here found a port bug, and one inherited Rust behaviour is
reported below for a decision.

### Added

- **`fuzz/`: the 8 Rust fuzz targets.** Each harness asserts invariants taken from
  Rust's definitions, not just "no crash", and fails with the seed that replays it
  (`./build/h 1 <seed>`). `cyrius fuzz --poison` runs all 8 under allocator redzones
  in about 10 s. `fz_common.cyr` holds the seeded xorshift and image helpers.
  - `fuzz_icc`: random bytes, mutated sRGB v2 profiles, and generated mft1/mft2 LUT
    profiles with grid sizes 1–5. A profile parses exactly when the error is NONE;
    tone curves stay within their table's range; LUT output stays in [0, 1].
  - `fuzz_convert`: every size from 1×1 to 62×62 through the three YUV standards,
    NV12 and the repacks, including arbitrary planes into the inverses.
  - `fuzz_filter`, `fuzz_blend`, `fuzz_blur`, `fuzz_composite`, `fuzz_lut`,
    `fuzz_transform`: the per-module contracts, with special floats (NaN, ±inf, ±0,
    extremes) for every f32 parameter. The SSE2 row paths are checked against the
    per-pixel ones.
  - `tests/ranga.fcyr`, the scaffold stub, is removed.
- **`tests/properties.tcyr`: all 31 `proptest.rs` properties.** Each runs over an
  edge grid (6³ colours) plus 4,096 seeded random inputs, and prints the first
  failing input. This includes the three that had no coverage at all
  (`lab_roundtrip`, `oklch_roundtrip`, the BT.2020 grey round trip).
- **`tests/integration.tcyr`: every test in three Rust files, under Rust's names:**
  `integration.rs` (23, including the six multi-step pipelines), `photoshop_reference.rs`
  (12: the seven reference modes within Rust's ±2, half opacity and three identities)
  and `visual_regression.rs` (10, including 2-D blur symmetry and Screen
  commutativity).
- **`tests/edge_cases.tcyr`: Rust's 30 edge cases, one for one,** under its section
  headings. The error Display tests check the message prefix, since the port drops
  payloads (parity doc).
- **Golden vectors from the oracle.** `tests/oracle_golden.tcyr` runs every oracle
  group in-process and checks each against Rust 1.0.1's line count and FNV-1a 64
  digest, recorded in `scripts/oracle/golden_rust.cyr` by `make_golden.py` from
  `rust.out` (never from the port's output). That covers 53 exact groups, about 2.1
  million values, with no `cargo` or `rust-old/` needed. The group code moved to
  `scripts/oracle/probe_core.cyr` so the probe and the test share it.
- **Oracle groups:** `resize` (8 target sizes × 3 filters, from a 23×17 image and
  from 1×1), `perspective` (matrices, inverses and transforms for four quads,
  including a degenerate one) and forward conversion (the three YUV standards,
  NV12 and five repacks at six sizes from 1×1 to 37×23). 15 new groups, all
  identical to Rust.
- **Tests: 2,261 assertions across 23 suites** (1,971 across 19 at 2.2.0).
- **Unit tests for the gaps the parity review listed:** `perspective_inverse_roundtrip`;
  XYZ, P3, CMYK (k ≠ 0) and HSL (l > 0.5) round trips away from white; resize
  downscales on all three filters; `view_mut` write-through; pool re-zeroing; alpha
  preservation in threshold and salt-and-pepper; and the YUV white/black checks
  through the public API.
- **`tests/spectral.tcyr`: the 8 `spectral.rs` tests for re-exported prakash items**
  (CMF table, illuminants, colour temperature, sRGB gamma, Wien peak, CRI), so a
  prakash bump that changes them fails here too.
- **CI:** a `cyrius fuzz --poison` step, and a step that runs `scripts/oracle/run.sh`
  against Rust 1.0.1. The lint gate also covers `fuzz/`, `tests/*.cyr` and the bench
  suites.

### Changed

- **The benchmark suite has Rust's 108 CPU benchmarks, on Rust's workloads.**
  `tests/ranga.bcyr` has the 100 outside spectral, and the new `tests/spectral.bcyr`
  has 8 (it needs prakash, so it is separate, as Rust's is feature-gated).
  `scripts/bench-history.sh` runs both. Up to 2.2.0, several benchmarks shared a
  Rust name but not its workload: blend at opacity 255 rather than 200, varied
  frames where Rust uses uniform 128 ones, a different crop window, CIE76/94 on the
  CIEDE2000 pair, saturation 1.2 rather than 1.5. Two were renamed to Rust's names
  and workloads: `composite_at_1080p` → `composite_at_640x480_on_1080p`, and
  `rgba_to_yuv420p_1080p` → `rgba_to_yuv420p_bt601_1080p`. Rows in
  `bench-history.csv` from before 2.3.0 are not comparable with later ones.
- **`docs/benchmarks-rust-v-cyrius.md` is rewritten** from a same-session run of
  both sides on one machine (Rust 1.0.1 rebuilt with rustc 1.99.1 from a scratch
  copy of `rust-old/`). Across the 108, the port is a median 8.0× slower, from 0.5×
  (`blend_row`, faster) to 367× (`color_rendering_index`, which is prakash's). The
  largest ranga-side gaps trace to two causes, now in roadmap 2.6.x. First, the 6.7.18
  stdlib's `memcpy`/`memset` are byte loops, which accounts for the pool at 127×,
  crop at 53× and the flips. Second, per-byte f32 helper calls that Cyrius does not
  inline: fade 57×, dissolve 50×, the gradients around 40×.
- **GPU coverage is visible.** The checks decided before a context is touched (the
  four UNSUPPORTED stubs, the format guard, null-context refusal, workgroup
  arithmetic, the bilinear module's structure) run on every host. A host without a
  GPU now runs 21 more GPU assertions: 19 moved out of the gate and 2 new. Each gated section prints `SKIPPED: N hardware
  assertions` when no AMD context opens, and asserts N when one does
  (`tests/gpu_hw.cyr`).
- **Test hygiene:** duplicate file-scope globals renamed (`b6`/`b8` in
  `filter_point.tcyr`, `got` in `pixel.tcyr`).
- **`make_golden.py`** writes no `__pycache__` into the tree.
- **Bundles** regenerated; only the `# Version:` line changed.

### Found, not changed: Rust's blurs darken a uniform image

Writing the fuzz invariants showed that Rust 1.0.1's `gaussian_blur` and `box_blur`
**do not leave a uniform image unchanged**: both truncate between their two passes,
so a flat image comes back up to 2 counts darker. Measured on rust-old: gaussian in
979 of 2,816 cases, box in 4,278 of 13,824. The port matches Rust exactly, so
`fuzz_blur` asserts Rust's measured bounds (gaussian and box within [-2, 0],
bilateral within [-1, 0], unsharp within [0, 2·amount + 1], median exact) rather
than the invariant one would expect. Fixing it means diverging from Rust under an
ADR; that decision is open.

## [2.2.0] - 2026-11-02

Milestone 2.2.x, "bit-exact again". `scripts/oracle` runs the frozen Rust 1.0.1 crate
and the port on identical inputs: 2.1 million values across 41 groups, every u8
output, the transition points of every monotone f32 → byte conversion, and the f32
and f64 fields. At 2.1.2, 27 groups differed. At 2.2.0 none does, apart from three
accepted within 2 ULPs by ADR 002. Every CPU-side SUSPECT item from the 2026-11-02
parity review is fixed or ADR'd. The six that remain are GPU items, scheduled for
2.5.x. Each behavioural fix has a test that fails on 2.1.2: 32 new assertions.

### Added

- **`scripts/oracle/`: the differential oracle.** It contains:
  - a Rust driver depending on `rust-old/` by path, which never builds inside it or
    modifies it;
  - a Cyrius probe;
  - `compare.py`;
  - `run.sh`, which exits non-zero on any unaccepted difference.

  Byte outputs of one f32 input are compared by **transition search**. Both sides
  binary-search the f32 at which each output level begins, which pins the output on
  every f32 in [0, 1] at about 7,700 evaluations per function. See
  `scripts/oracle/README.md`.
- **ADR 002: f32 transcendentals follow correct rounding, not the oracle host's
  libm.** Rust's f32 `sin`/`cos`/`powf`/`ln` are glibc's here, and glibc's are not
  correctly rounded. `oklch_to_oklab` (1,419 and 1,830 of 140,608 values) and
  `color_temperature` (6 of 16,716) differ from Rust by ≤ 2 ULP. An independent
  correctly rounded model reproduces ranga's value at every differing point checked
  and Rust's at none. The oracle accepts these three groups only within that bound.

### Fixed: computations now match Rust 1.0.1 bit for bit

- **Colour, single precision as in Rust:**
  - `linear_to_srgb` and `lin_rgba_to_srgba` (alpha and RGB);
  - `rgbaf32_to_rgba8`;
  - `srgba_to_hsl` / `hsl_to_srgba`, including the f32 `1e-6` threshold;
  - `cmyk_to_srgba` / `srgba_to_cmyk`;
  - `oklab_to_oklch`, with f32 sqrt, atan2 and to_degrees, and an exact f32
    `rem_euclid`.

  All of these ran in f64 or widened before the final `* 255 + 0.5`. The f64
  results were more accurate and therefore wrong, rounding the other way next to
  each half-step: f32 `0x3f010101` gave 128 where Rust gives 129. `linear_to_srgb`
  had 171 of 255 output boundaries off by up to 4 ULPs, and `srgba_to_cmyk.c` was
  off by up to 248 ULPs.
- **`levels`:** the LUT is built in f32. 7 of 336 settings were one count off.
  NaN black, white and gamma are checked too.
- **`bilateral`:**
  - the coefficients and `exp` are f32;
  - σ ≤ 0 returns `RG_ERR_OTHER`, as Rust's `Other` (it was `INVALID_PARAMETER`);
  - a NaN σ proceeds and copies each pixel unchanged, as in Rust. The port used to
    reject it.
- **`hue_shift`:** inherits the HSL fix, and does its `(h + degrees).rem_euclid(360)`
  in f32. At ±1e9° the f32 add swallows the hue, as in Rust.
- **`affine_inverse`:** computes the translation in Rust's evaluation order. 99 of
  206 inverses had a different `tx`, and Nearest sampling then took whole columns
  from the wrong source pixel.
- **`affine_is_identity`:** a NaN term is not the identity.
- **ICC grid size 1** (a constant LUT) is accepted, as in Rust. Grid 0 is still
  rejected.
- **ICC errors:** the public parsers report every malformed profile as
  `RG_ERR_INVALID_FORMAT`, as Rust does. The internal `_icc_*_parse` functions keep
  the PARSE / INVALID_FORMAT split for tests.
- **`pixel_buffer_new`:** an overflowing size is `RG_ERR_BUFFER_TOO_SMALL`, as Rust's
  `PixelBuffer::new`, rather than `RG_ERR_ALLOCATION`.
- **`ranga_error_message(RG_ERR_INVALID_FORMAT)`** reads "invalid pixel format",
  which is Rust's text without its payload.

### Fixed: robustness

- **`BufferPool` is a growable array, like Rust's `Vec`.**
  - `buffer_pool_new(1 << 40)` no longer fails.
  - Acquire and release scan `len` buffers, not `max_buffers` slots.
  - Removal is a `swap_remove`, as in Rust.
  - A negative `acquire` size is refused.
- **The RGBA → YUV converters** no longer return an incomplete Y plane as a success
  when the row kernel's scratch allocation fails.
- **`lut3d_new`** refuses sizes whose byte count would overflow i64 (above 916015).
- **`flood_fill`'s stack grows** instead of assuming `w*h` entries is a bound, which
  it is not: a run can be pushed from both neighbouring rows.

### Performance

- **`resize_nearest_1080p_to_720p` went from 13.0 ms to 4.3 ms.** The inner loop
  called `memcpy(.., 4)` per pixel, and that call made the loop's speed depend on
  code layout: unrelated growth in `error.cyr` and `color.cyr` took it to 22 ms
  during this release. It is now a single 32-bit load/store. Against Rust 1.0.1's
  1.59 ms that is about 2.7× (it was 7.8×). The other benchmarks are within ±10% of
  2.1.2 (median ratio 1.01), and every run is in `bench-history.csv`.

### Changed

- **ADR 001** names all four YUV inverses it covers:
  - BT.601, BT.709, BT.2020 and NV12;
  - its "all five failed to reproduce" note is corrected: all five did reproduce and
    are fixed here.
- **Stale source comments corrected:**
  - `error.cyr`'s `INVALID_PARAMETER` scope and its "logged" claim;
  - `f64_to`, which saturates NaN to 0 (probed at 6.6.9 and 6.7.18);
  - the affine NaN-coordinate note (NaN matrices sample index 0, exactly as Rust's
    `as usize`; checked by the oracle on a 1×1 source);
  - the SIMD headers in `convert.cyr` and `blend.cyr`;
  - `blend_row_normal`'s length guard;
  - a `histogram.cyr` cross-reference.
- **Bundles** regenerated. The sidecars are unchanged.

## [2.1.2] - 2026-11-02

Closes out the toolchain bumps from 2.0.3 to 2.1.1 (cyrius 6.6.9 → 6.7.18) with a
P(-1) review: [`docs/audit/2026-11-02-audit.md`](docs/audit/2026-11-02-audit.md). That
review found 9 items (2 medium, 4 low, 3 info). Eight are fixed here; the ninth is
upstream. Each behavioural fix comes with a test that fails on 2.1.1.

### Fixed

- **ICC: an `mft2` LUT profile with zero input-table entries is accepted again.** Since
  cyrius 6.6.7 `fl_alloc(0)` returns 0. From 2.0.3 on, the profile was therefore
  rejected as `RG_ERR_ALLOCATION`, where Rust parses it with identity input curves.
  Empty curves are now left unallocated, and `_icc_apply_curve` treats them as
  identity, as Rust does.
- **`affine_rotate` and the Oklch/CIEDE2000 trig no longer turn a huge angle into NaN.**
  `_rg_sin`/`_rg_cos` returned NaN for |x| ≥ 2^63, a guard for x87 `fsin`/`fcos`
  returning their argument there. At the 6.7.18 pin `f64_sin`/`f64_cos` reduce every
  finite argument and match glibc bit for bit, which is what Rust calls. So
  `affine_rotate(1e20)` was NaN where Rust is finite. NaN now comes out only for a NaN
  or infinite angle.
- **ICC: a profile with zero tags reports `RG_ERR_INVALID_FORMAT`, not
  `RG_ERR_ALLOCATION`.** This is the same `fl_calloc(0)` change. Rust parses the empty
  table and fails the first lookup with InvalidFormat, from both parsers.
- **`src/blend.cyr:143` compiler warning.** ColorBurn subtracted an `: i64`-declared
  call's f32 bits with a bare `-`. It now uses the typed `_bl_sub`. The exhaustive
  probe (12 modes × 65,536 pairs) is still byte-identical to Rust.
- **`tests/icc.tcyr`: three discarded `#must_use` results** are now asserted to be 0,
  which removes three warnings. ranga's own code now compiles with zero warnings.

### Changed

- **prakash `2.15.0` → `2.15.1`** (feature `spectral`; pins 6.7.18). Only
  `lib/prakash.cyr` changed. The lock records 69 files and 7 commit pins, each matching
  its GitHub tag, and `cyrius deps --verify` reports 69 verified, 0 failed.
- **`lib/flags.cyr` and `lib/hashmap_fast.cyr` removed** (after the 2.1.1 commit).
  6.7.18's `cyrius deps` reports both as outside the declared footprint and no longer
  vendors them, and nothing includes either one. They were dropped with
  `rm … && cyrius deps --lock`, and a CI-style resolve does not bring them back.
  `src/simd_u8.cyr` and `docs/development/port-mechanics.md` cited
  `lib/hashmap_fast.cyr:66` as the in-tree `asm { }` precedent, and now name the cyrius
  stdlib's copy.
- **`scripts/bench-history.sh` rewritten for Cyrius.** It was still the Rust script
  (`cargo bench` into a `benches/` directory that does not exist), so it failed on every
  run and no Cyrius history had ever been recorded. It now runs `cyrius bench`, and
  each row records the version, commit, cyrius pin and timer-floor regime. The new
  `bench-history.csv` holds the 2.1.1 baseline and the 2.1.2 run: 34 of 35 are within
  ±10%, and the median ratio is 1.01.
- **CI lint gate:** a single `cyrius lint --strict --strict-deferrals` call replaces the
  per-file loop that parsed printed totals. The loop's comment ("takes ONE path, ALWAYS
  exits 0") has been false since cyrius 6.6.5. The strict form was checked at the pin:
  it exits 2 on a warning or an untracked deferral and 0 on the repo.
- **Source comments that stated toolchain or dependency facts the bumps made false**
  are corrected, with no code change:
  - the `error.cyr` shim table;
  - ganita's NaN min/max and the old cosine limit in `composite.cyr`;
  - `#derive(Serialize)` on enums, which works at 6.7.18 and emits serde's variant
    names;
  - prakash's three renamed re-exports;
  - the ai-hwaccel and mabda version headers;
  - the GPU lowerer limits mabda 4.1.0 lifted.

  The GPU workarounds themselves stay, and are retired in roadmap 2.5.x.
- **Docs:**
  - `docs/development/parity-rust-v-cyrius.md` is a new item-by-item review of all 303
    Rust public items;
  - `docs/development/roadmap.md` is rewritten future-facing, with milestones 2.2.x
    to 2.x;
  - `docs/architecture/001` records what mabda 4.2.0's native GPU path accepts;
  - `ROADMAP.md` is now a summary.
- **Bundles** were regenerated for the changes above. The sidecars are unchanged.

### Corrected

- **There is no real fuzz harness.** `tests/ranga.fcyr` is still the 16-line `cyrius port`
  scaffold stub: it includes no ranga module, calls an empty `fuzz_main` once and prints
  "fuzz: ok". The "fuzz re-verified" lines in the 2.0.3, 2.0.4, 2.1.0 and 2.1.1 entries
  meant only that the stub ran. They have been removed from 2.0.4, 2.1.0 and 2.1.1, and
  none of the 8
  Rust fuzz targets is ported. Tracked in `docs/development/roadmap.md` (2.3.x).

### Known

- prakash 2.15.1 brings three warnings into every compile that includes it:
  `num_fft` undefined, and two f64/int comparisons. `cyrius distlib` reports `num_fft`
  as undefined in `ranga-spectral.cyr`. These are upstream, and vendored `lib/` is not
  modified here.
- `cyrius deps` still warns that stdlib `sakshi` overlaps the sakshi package hisab
  resolves. The kept stdlib snapshot is byte-identical to sakshi 2.5.10.
- The P(-1) cross-toolchain diff shows that ICC results computed through `pow` move by
  1–2 ULP between stdlib snapshots: net toward Rust, but not stable across pins. Every
  other probed output was identical under 6.6.9 and 6.7.18, except one, which the bump
  fixed: `gradient_linear_angled` at 1e25°.

## [2.1.1] - 2026-11-02

### Fixed

- **`src/icc.cyr`: the private tone-curve size `_TC_SIZE` is now `_ICC_TC_SIZE`.**
  The 6.7.18 stdlib `lib/bayan.cyr` declares its own `const _TC_SIZE = 80`, and 6.7.18
  refuses a `var` that reuses a top-level const name. `tests/icc.tcyr` therefore failed
  to compile ("'_TC_SIZE' is already declared as a top-level const"), and so would any
  consumer that loads a ranga bundle next to bayan 6.7.18. The new name follows the
  module's `_ICC_` prefix. No public name changed. Every top-level `var` / `const` /
  `fn` / `enum` name in `src/` was checked against `lib/` (stdlib and dep bundles,
  enum members included), and this was the only collision.

### Changed

- **Toolchain `6.7.9` → `6.7.18`.** `lib/` re-vendored with
  `cyrius deps --features spectral,hwaccel,gpu`, `cyrius.lock` resolved from scratch,
  all four bundles regenerated with `scripts/version-bump.sh`. Build, the full test suite
  (19 files, 1,921 assertions), lint (41 files, 0 warnings, 0 untracked deferrals),
  bench and `cyrius distlib --check` re-verified.
- **Optional deps moved to their latest releases.**
  - prakash `2.11.0` → `2.15.0` (feature `spectral`; pins 6.7.18). 2.15.1 is not
    released yet.
  - ai-hwaccel `2.4.2` → `2.5.6` (feature `hwaccel`; pins 6.7.18)
  - mabda stays at `4.2.0` (feature `gpu`; pins 6.7.5). It is still the newest release.
  - Transitive pins moved with them: hisab `3.4.0` → `3.4.1`, sakshi `2.5.8` → `2.5.10`.
    samvada `1.0.1` and chitra `1.0.3` are unchanged.
- **`lib/`:** 42 files updated, plus a new `lib/unicode/` (7 files) that the 6.7.18
  `lib/bayan.cyr` includes. The lock records 71 files and 7 commit pins, each matching
  its GitHub tag, and `cyrius deps --verify` reports 71 verified, 0 failed. The
  `[deps] stdlib` list is unchanged. ai-hwaccel 2.5.6's sidecar adds `atomic`, but its
  bundle includes `lib/atomic.cyr` itself and `cyrius deps` asks for nothing more.
- **Bundles:** besides the version line and the rename above, the 6.7.18
  `cyrius distlib` trims two sidecars and their `# Requires` blocks: `freelist` from
  `ranga-hwaccel` and `math` from `ranga-spectral`. Both still arrive through other
  leaves.

### Known

- The compiler warnings are the same set as 2.1.0, including `src/blend.cyr:143:62`.
  `cyrius distlib` reports `num_fft` as undefined in `ranga-spectral.cyr`. The name
  comes from prakash, 6.7.9 reported it too, and it matches the "undefined function
  'num_fft'" warning in the test run.
- `cyrius deps` still warns that stdlib `sakshi` overlaps the sakshi package hisab
  resolves. The kept stdlib snapshot is byte-identical to sakshi 2.5.10.

## [2.1.0] - 2026-11-02

### Changed

- **Toolchain `6.6.20` → `6.7.9`**, the first pin on the 6.7 language minor (checked
  traits, `dyn`, slices, tuples). ranga uses none of the new constructs, and none of the
  new compile errors fire on it. `lib/` re-vendored with
  `cyrius deps --features spectral,hwaccel,gpu`, `cyrius.lock` resolved from scratch,
  all four bundles regenerated with `scripts/version-bump.sh`. `src/` and `tests/` are
  untouched. Build, the full test suite (19 files, 1,921 assertions), lint (41 files,
  0 warnings, 0 untracked deferrals), bench and `cyrius distlib --check` re-verified.
- **Optional deps moved to the releases that pin the 6.7.9 line.**
  - prakash `2.10.1` → `2.11.0` (feature `spectral`; pins 6.7.9)
  - mabda `4.1.7` → `4.2.0` (feature `gpu`; pins 6.7.5)
  - ai-hwaccel stays at `2.4.2` (feature `hwaccel`; pins 6.6.20). It has no release
    between 6.6.20 and 6.7.9, and the next tag, 2.5.0, pins 6.7.18.
  - Transitive pins moved with them: hisab `3.3.7` → `3.4.0`, sakshi `2.5.7` → `2.5.8`.
    samvada `1.0.1` and chitra `1.0.3` are unchanged.
- The lock records 64 files and 7 commit pins, each matching its GitHub tag, and
  `cyrius deps --verify` reports 64 verified, 0 failed. No file was added to or
  removed from `lib/`, and the `[deps] stdlib` list is unchanged.
- The bundles differ from 2.0.4 only in the `# Version:` line; the `.deps` sidecars
  are unchanged.

### Known

- The compiler warnings are the same set as 2.0.4 under 6.6.20, including
  `src/blend.cyr:143:62` ("f32 arithmetic with a non-f32 right operand"). 6.7.9 adds
  none. `cyrius deps` still warns that stdlib `sakshi` overlaps the sakshi package
  hisab resolves; the kept stdlib snapshot is byte-identical to sakshi 2.5.8.

## [2.0.4] - 2026-11-02

### Changed

- **Toolchain `6.6.9` → `6.6.20`.** `lib/` re-vendored with
  `cyrius deps --features spectral,hwaccel,gpu` (the way CI resolves), `cyrius.lock`
  regenerated, all four bundles regenerated with `scripts/version-bump.sh`. `src/` and
  `tests/` are untouched. Build, the full test suite (19 files, 1,921 assertions), lint
  (41 files, 0 warnings, 0 untracked deferrals), bench and `cyrius distlib --check`
  re-verified.
- **Optional deps moved to the releases that pin the 6.6.20 line.**
  - prakash `2.2.8` → `2.10.1` (feature `spectral`; pins 6.6.20)
  - ai-hwaccel `2.3.18` → `2.4.2` (feature `hwaccel`; pins 6.6.20)
  - mabda `4.1.1` → `4.1.7` (feature `gpu`; pins 6.6.18). mabda has no 6.6.20 release,
    and the next tag, 4.2.0, pins 6.7.5.
  - Transitive pins moved with them: hisab `2.11.2` → `3.3.7`, samvada `0.4.1` → `1.0.1`,
    chitra `0.3.1` → `1.0.3`, sakshi `2.4.11` → `2.5.7`.
- **`sync` and `chrono` added to `[deps] stdlib`.** ai-hwaccel 2.4.2's sidecar needs
  `sync` and mabda 4.1.7's needs `chrono`; neither serves a core module. The declared
  subset is 31 leaves, the last 15 for the optional libraries. `lib/poison.cyr` is also
  new: the 6.6.20 `alloc.cyr` and `freelist.cyr` include it.
- **Bundle headers and sidecars.** The 6.6.20 `cyrius distlib` opens each bundle with a
  compile-verified `# Requires` block that includes its sidecar leaves, and trims each
  `.deps` sidecar to the leaves that bundle needs (`ranga.deps` 29 → 4). The bundle
  bodies are unchanged; only the header and the `# Version:` line differ.

### Fixed (lock)

- `cyrius.lock` carried the old commit pins next to the new ones after the re-resolve
  (14 `commit` lines for 7 deps). It was deleted and resolved from scratch with the
  features: 64 files locked, 7 commit-pinned, each pin matching its GitHub tag, and
  `cyrius deps --verify` reports 64 verified, 0 failed. The file-hash lines are
  identical to the ones the merged lock held.

### Known

- **New compiler warning at `src/blend.cyr:143:62`:** "f32 arithmetic with a non-f32
  right operand". The line is `one - ganita_f32_min(..)`. The callee is declared `: i64`
  but returns f32 bits, so the value is right at runtime. The cyrius 6.6.10 downstream
  survey reported this exact site. Not fixed here because `src/` is untouched in this
  release. Every other warning in the test run (`num_fft`, the `#must_use` results in
  the icc suite, two comparisons inside `lib/prakash.cyr`) also appears in 2.0.3
  under 6.6.9.
- `cyrius deps` warns that stdlib `sakshi` overlaps the sakshi package hisab 3.3.7
  resolves. It keeps the stdlib snapshot, which is byte-identical to sakshi 2.5.7.

## [2.0.3] - 2026-09-28

### Fixed

- **`tests/gpu_kernels.tcyr`: the negative fade factor was -4.0, not -1.0.** The row
  passed `f64_to_f32(0 - 1.0)`, and `0 - 1.0` is an INTEGER subtraction of 1.0's bit
  pattern (an int-left operator takes its integer arm), which is -4.0. The row still
  passed — any negative factor clamps to zero — but it did not test what its message
  says ("white faded by -1 is black"). It is `f64_to_f32(-1.0)` now; unary minus flips
  a float literal's sign bit since cyrius 6.6.8, which also reports the `0 - 1.0`
  shape as `integer arithmetic with an f64 right operand`. Found by the cyrius 6.6.10
  ecosystem scan (the only such code site in the ecosystem).

### Changed

- **Toolchain `6.6.2` → `6.6.9`.** `lib/` re-vendored with `cyrius deps` (the declared
  stdlib subset), `cyrius.lock` refreshed, all four bundles regenerated with
  `scripts/version-bump.sh` (reproducible; the `.deps` sidecars now list only the
  leaves each bundle references). `src/` is untouched. Build, the full test suite
  (19 files), bench and fuzz re-verified.


### Fixed (lock)

- `cyrius.lock` is resolved the way CI resolves (`--features spectral,hwaccel,gpu`). The pin move
  re-locked with the default feature set, which dropped the seven feature deps' commit pins and left
  `lib/flags.cyr` — reachable only through those features — hashed at its 6.6.0–6.6.4 content, so CI's
  `cyrius deps --features …` refused it under the unchanged 6.6.9 pin. Re-locked with the features:
  the seven commit pins are back unchanged and `lib/flags.cyr` carries the 6.6.9 content. src/ untouched.

## [2.0.2] - 2026-09-11

### Changed

- **Toolchain `6.5.33` → `6.6.2`.** Migrated to the `Result` value form:
  2 first-party file(s) changed. Every surface re-verified — build, tests, and any
  bench/fuzz/distlib target the repo ships.


## [2.0.1] — 2026-08-21

Maintenance release. `src/` is untouched — the toolchain pin and the
optional-dependency pins move, the bundles are restamped, and the lint gate
starts doing its job. All 1,921 assertions across 19 suites pass before and
after, with byte-identical bundle contents apart from the version header.

### Changed

- **Toolchain pin `6.5.31` → `6.5.33`.** The wrapper installed locally had
  already moved to 6.5.33, so every build was emitting a drift warning; CI
  installs whatever `[package].cyrius` names, so it was still building against
  6.5.31. The pin is now the version that is actually in use. Re-vendoring the
  declared `[deps].stdlib` subset from the 6.5.33 snapshot changed no existing
  leaf byte-for-byte and added one new file, `lib/hashmap_fast.cyr`.
- **prakash `2.2.3` → `2.2.8`** (optional, feature `spectral`).
- **ai-hwaccel `2.3.17` → `2.3.18`** (optional, feature `hwaccel`).
- **mabda stays at `4.1.0`** — already the newest tag.
- Transitive pins move with them: **hisab `2.11.1` → `2.11.2`** and
  **sakshi `2.4.10` → `2.4.11`**. `cyrius.lock` now records 59 deps locked and
  7 commit-pinned, up from 52 and 4.
- All four `dist/` bundles restamped `v2.0.1`. The `.deps` sidecars are
  unchanged, and the bundle bodies are byte-identical to 2.0.0.

### Fixed

- **`cyrius.lock` pinned a tag that no longer exists.** The lock recorded
  hisab `2.11.1`, which has since been retracted upstream — the tag now 404s,
  so a clean `cyrius deps` on a fresh checkout could not resolve it. Re-resolving
  moves the pin to `2.11.2`, which is present. Nothing in-tree had noticed
  because `lib/hisab.cyr` was already vendored locally.
- **The CI lint gate could not fail, and covered one file in forty-one.**
  `cyrius lint` accepts a single path and always exits 0 — it reports its
  counts rather than signalling through the exit status. The step read
  `cyrius lint src/*.cyr tests/*.tcyr`, so it linted `src/blend.cyr`, silently
  discarded the other 40 arguments, and passed regardless of what it found.
  It now loops every file and gates on the reported totals. Note for anyone
  writing a similar gate: cyrlint prints `N warnings` on **stdout** and
  `N untracked deferrals` on **stderr**, so the capture needs `2>&1` or the
  deferral count is silently lost.
- **Ten lint warnings and two untracked deferrals in `tests/`**, which the
  broken gate had been hiding. `src/` was already clean and stays clean.
  - `tests/pixel.tcyr` declared `exact` and `big` twice each at file scope.
    Cyrius binds a name to the LAST declaration, so cyrlint flagged the earlier
    reads as `silent zero at init`. The runtime values were correct — the
    earlier `var` writes the same global before the read — but the duplicate
    names were real. The second of each is now `view_exact` / `pool_big`.
  - Three lines over 120 characters wrapped (`icc.tcyr` ×2, `ranga.tcyr`), and
    four doubled blank lines closed up (`blend.tcyr`, `icc.tcyr`,
    `pixel.tcyr`, `transform.tcyr`).
  - The two `resize_bilinear` deferrals in `tests/gpu_kernels.tcyr` now
    cross-reference the 2.0.0 **Not ported** entry above, which is where that
    work is tracked. cyrlint matches `not yet` inside `cannot yet` and wants
    the pointer on the same line.

  No assertion changed: 1,921 before, 1,921 after, every suite matching its
  previous count exactly.

## [2.0.0] — 2026-08-19

**The Cyrius port.** ranga is now written in Cyrius. The Rust line ends at
1.0.1 and is preserved at `rust-old/` as the parity oracle. Major bumps without
resetting the series, matching every prior AGNOS port (vidya 1.x→2.0.0,
prakash 1.2.0→2.0.0, mabda 1.0.0→2.0.0, ai-hwaccel 1.2.0→2.0.0).

**This is not a source-compatible upgrade.** Cyrius has no methods, no traits and
no `Drop`, so every `X::y()` is now a free `x_y()`, every owning type has an
explicit `*_free`, and errors are `i64` codes rather than a `RangaError` enum.
Consumers rewrite against the new surface; `docs/development/parity-rust-v-cyrius.md`
maps it item by item.

### Added

- **`spectral` and `hwaccel` as optional features**, as they were in Rust
  (`default = []`). `spectral` bridges **prakash 2.2.3**; `hwaccel` bridges
  **ai-hwaccel 2.3.17**.
- **`gpu` as an optional feature over mabda 4.0.9**, with native AMD (amdgpu /
  GFX9) and NVIDIA (nouveau / SM75) backends and wgpu as the fallback.
- **A SPIR-V emitter** (`src/gpu_spirv.cyr`). mabda's native backends accept
  SPIR-V and nothing else, and nothing in the stack translates WGSL, so ranga
  emits SPIR-V directly. Eleven of the fourteen GPU operations run natively;
  all fourteen have WGSL for the wgpu path.
- **`ranga_error_message`** — the `RangaError` Display strings, which the port
  had been missing entirely.
- **`bytevec.cyr`** — a byte-granular `Vec<u8>` substitute. Cyrius's `lib/vec.cyr`
  stores i64 slots, so a 4-byte pixel would occupy 32.
- Four bundle profiles: `dist/ranga.cyr` (core, no external deps) plus
  `-spectral`, `-hwaccel` and `-gpu`.

### Changed

- **`ColorSpace` gains `Bt601` and `Bt709`** and renumbers to match Rust
  exactly. The port had dropped both and let its own `CieLab`/`Oklab` additions
  take values 3-6, so a `ColorSpace` persisted by 1.0.1 read back as a different
  space. The first seven discriminants are now Rust's; ranga's additions are
  numbered above that range.
- **`PixelView`/`PixelViewMut` require an EXACT buffer length**, as Rust did.
  The port had accepted any buffer large enough, which silently widened the
  contract.
- **GPU grayscale is BT.709 where CPU grayscale is BT.601** — inherited from
  Rust, which had the same split. Documented rather than reconciled.

### Not ported

- **`gpu_blend`, `gpu_noise_gaussian`, `gpu_gaussian_blur` and bilinear
  `gpu_resize`** report `RG_GPU_ERR_UNSUPPORTED` on the native backends and run
  on wgpu. Blocked by limits in mabda's SPIR-V→GFX9 lowering — register
  pressure for the first three, `OpLoopMerge` rejection for blur. Filed at
  `mabda/docs/development/issues/2026-08-19-native-spirv-compile-limits.md`.
- **`GpuBuffer::download_async`** — existed to work around a Rust borrow-checker
  constraint; the Cyrius mabda exposes no async readback to build on.
- **`Lut3d::from_cube`** — the `.cube` file parser. The LUT itself is ported;
  loading one from disk is not.
- **`PixelBuffer::into_data`**, **`GpuContext::new_with_hwaccel`**, and the
  `FromStr` inverses of `blend_mode_name` / `color_space_name` / `pixel_format_name`.
- **Serde codecs for the four enums** are hand-written. `#derive(Serialize)`
  works on a Cyrius struct but silently emits nothing on an enum; filed at
  `cyrius/docs/development/issues/2026-08-19-derive-serialize-on-enum-emits-nothing.md`.

### Toolchain

- **Cyrius pin 6.5.31** (was 6.5.27), **mabda 4.1.0** (was 4.0.9 — aligns with
  the toolchain's own fold and clears a shadow warning).
- **ganita 1.1.4** (folded into cyrius 6.5.30) closes internalized-shim item 1.
  It fixed `ganita_f32_cbrt(0.0)` returning NaN and gave `ganita_f64_pow`
  zero-base, zero-exponent and negative-base handling. `_rg_pow`, `_rg_cbrt` and
  `_rg_f32_cbrt` are now thin delegations. 1.1.4 goes further than the shims
  did — a negative base with an integral exponent now returns a real value, as
  Rust's `powf` does, where the old path gave NaN. Integral exponents land
  within 2 ULP of exact rather than bit-exact; measured across a full gamma-2.0
  tone-curve sweep, zero of 256 bytes differ. Two defects this port filed were fixed in
  between: decimal float literals past ~9 significant digits parsing to a
  silently wrong value (6.5.28), and `distlib` profile `.deps` sidecars written
  empty (6.5.29). ranga keeps its hex bit patterns for the Oklab matrices —
  they are correct and pinned by the white-point assertions, and rewriting
  eighteen verified constants would be churn with a real chance of introducing
  the very error class the hex was adopted to avoid.
- **`#derive(Serialize)` on an enum now generates a codec** as of 6.5.31 — it
  emitted nothing through 6.5.30. ranga has no serde at all, so wiring it up is
  available parity work rather than a change to existing code.

### Fixed — found by the final parity sweep

- **`oklab(black)` returned `(NaN, NaN, NaN)`.** `ganita_f32_cbrt(0.0)` is NaN
  where Rust's `f32::cbrt(0.0)` is `0.0`, and it propagated through the whole
  Oklab M2 matrix. Rust pins this in a test the port had not ported.
- **`apply_lut3d` truncated where Rust rounds half-up**, so half of all grey
  levels came out one count low.
- **An ICC LUT profile with `grid_size == 0` read below its allocation.** Now
  rejected — a deliberate divergence from Rust, which underflows there.

- **blend's `f32 → u8` tail computed in f64** where Rust is single-precision
  throughout, giving 229 against Rust's 230 on a ColorBurn near a rounding
  boundary. Now matches exactly.

- **`srgb_to_linear` computed in f64** where Rust is single-precision
  throughout. **214 of the 256 possible u8 inputs differed by one or two ULP**,
  with the whole test suite green — every hand-written assertion used a
  tolerance wide enough to swallow it. All 256 are now bit-identical.
- **`linear_to_srgb` returned `i64::MIN`** for NaN and for +inf, where Rust's
  saturating cast gives 0 and 255.
- **`levels`** rejected a NaN gamma that Rust accepts, reported non-positive
  gamma as `INVALID_PARAMETER` instead of `Other`, and blackened on a NaN white
  point where Rust whitens.
- **`brightness` with a NaN offset** blackened the image; Rust's saturating cast
  makes it a no-op.
- **`_icc_write_s15fixed16` used banker's rounding** where Rust rounds
  half-away-from-zero.
- **`pixel_format_checked_buffer_size`** bounded the pixel count against one
  format-independent constant, rejecting sizes Rust accepts.
- **Singular-matrix transforms** reported `INVALID_PARAMETER` where Rust reports
  `Other`.

A **5,800-comparison differential sweep** against the real crate now backs these:
every u8 output path is byte-exact, and the ~50 residual differences are all in
f32/f64 intermediates that never round to a byte (max 1.4e-14 on a scale where
one byte step is 0.39).

### Intentional divergence

- **The YUV→RGB inverse does not reproduce Rust's `i16` wrap.** Rust's inverse
  overflows on saturated chroma — measured against the real crate, red
  round-trips to **black**. Reproducing that would mean deliberately
  re-introducing an overflow. See
  [ADR 001](docs/adr/001-yuv-inverse-does-not-reproduce-rusts-i16-wrap.md).
  Anything pinning golden YUV images will need regenerating.

### Testing

**1,878 assertions across 19 suites, 0 lint warnings.** Expected values come
from Python oracles replicating the exact Rust formula, not from reading the
Cyrius back. Every module is mutation-tested once it goes green.

The GPU path is hardware-verified on an AMD Cezanne (gfx90c): SPIR-V emitted by
ranga, lowered to GFX9 by mabda, dispatched, and read back byte-exact against
f32 oracles.

### Upstream issues filed during the port

Two resolved in cycc during the port — decimal float literals past ~9
significant digits parsing to a silently wrong value, and `fl_calloc` re-zeroing
mmap'd pages a byte at a time. Three remain open: mabda's native SPIR-V limits,
`#derive(Serialize)` on enums, and `distlib` profile sidecars written empty.


## [1.0.1] — 2026-08-13

**Final Rust release.** ranga continues in Cyrius alongside the rest of the AGNOS
stack (mabda, prakash, ai-hwaccel have already ported). This release brings the
Rust line current on toolchain and dependencies and closes out the crates.io
publishing pipeline. No API changes — 1.0.0 consumers upgrade in place.

### Changed

- **Toolchain**: verified against Rust 1.97.1 (was 1.96.0). MSRV stays **1.89**, edition stays 2024 — the floor is deliberately unchanged so consumers still on the Rust line can take this release during the port.
- **Release pipeline**: `cargo publish` removed from `.github/workflows/release.yml`. The `publish` job is replaced by `verify`, which keeps the VERSION/Cargo.toml/tag consistency gate; `release` now depends on `[ci, build, verify]`. Releases are GitHub artifacts only.
- **Dependencies** — direct:
  - `ai-hwaccel` 1.0.0 → 1.2.0
  - `criterion` 0.5.1 → 0.8.2 (dev)
  - `rayon` 1.11.0 → 1.12.0
  - `serde` 1.0.228 → 1.0.229, `serde_json` 1.0.149 → 1.0.151
  - `thiserror` 2.0.18 → 2.0.20
  - `wgpu` 29.0.1 → 29.0.4 (patch only — see below)
- **Dependencies** — transitive: `naga`/`wgpu-core`/`wgpu-hal`/`wgpu-types` 29.0.1 → 29.0.4, `bytemuck` 1.25.0 → 1.25.2, `getrandom` 0.4.2 → 0.4.3, plus 60+ others. Dependency graph shrank from 210 to 196 crates.
- **`hwaccel::probe`**: migrated to the ai-hwaccel 1.2 iterator API — `AcceleratorRegistry::available()` and `by_family()` now return `impl Iterator` rather than slices, so the `.iter()` hops are gone. Both already filter on `available`, so probe results are unchanged.
- **Benchmarks**: `black_box` imported from `std::hint` instead of `criterion::black_box`, which criterion deprecated in 0.6. Applies to all 10 bench suites.

### Added

- `median_r5_512x512` benchmark. The 1.0.0 Huang rewrite made `median` O(n·r) per channel instead of O(n·r²·log r²), but only r=1 was ever measured — the one radius where a 256-bin-per-channel histogram costs more to set up than sorting the 9-sample window outright. Measuring a second radius keeps the tradeoff visible in the record rather than leaving a bare regression.

### Fixed

- **RUSTSEC-2026-0204** (vulnerability): invalid pointer dereference in `crossbeam-epoch`'s `fmt::Pointer` impl, reached through `rayon` → `crossbeam-deque`. Resolved by 0.9.18 → 0.9.20. This was the one hard `cargo audit`/`cargo deny` failure on 1.0.0's lockfile.
- **RUSTSEC-2026-0097** (unsoundness): `rand` unsound with a custom logger using `rand::rng()`. Resolved by 0.9.2 → 0.9.5.
- **RUSTSEC-2026-0190** (unsoundness): `anyhow` `Error::downcast_mut()`. `anyhow` dropped from the graph entirely by the criterion 0.8 upgrade.
- **Benchmarks measuring nothing**: `hsl_roundtrip`, `xyz_conversion_roundtrip`, and `pixel_view_create` bound their results to `_` instead of returning them from the `iter` closure, so nothing black-boxed the output and the optimizer was free to delete the work being timed. `hsl_roundtrip` was recording **257 picoseconds** — under one clock cycle — against a real cost of 19.6ns. The bug predates this release but was masked by criterion 0.5's weaker `black_box`; `std::hint::black_box` let the optimizer see through it completely. All three now return their values.
- **Docs**: README listed `ai-hwaccel 0.23.3` in the feature and capability tables (actual: 1.2); the GPU compute guide's install snippet still said `version = "0.20"`. The median benchmark comment still described the pre-1.0.0 O(n·radius²) complexity.

### Performance

Neutral. Full 134-benchmark sweep recorded to `benches/history.csv`. The last
recorded sweep was `04f74df` (2026-04-02), eight commits back, so a naive diff
against it attributes that intervening work to this release — it shows 50
benchmarks >10% faster (the SIMD conversion and blur paths) and 21 >10% slower.
Neither belongs to 1.0.1.

An A/B against HEAD-without-these-changes, same machine and session, confirms the
dependency and toolchain moves are performance-neutral:

| benchmark | Apr baseline | HEAD today | 1.0.1 today |
|---|---|---|---|
| `median_r1_512x512` | 15.347 ms | 30.549 ms | 29.341 ms |
| `affine_rotate_512x512` | 5.9141 ms | 7.2336 ms | 7.2699 ms |
| `resize_bicubic_1080p_to_720p` | 91.485 ms | 105.37 ms | 104.83 ms |
| `cmyk_roundtrip` | 10.155 ns | 13.417 ns | 13.076 ns |

Every apparent regression reproduces at HEAD without this release's changes. The
`median_r1` movement is the 1.0.0 Huang rewrite's small-radius tradeoff, which
landed in `2aa3b82` after the last recorded sweep; `median_r5_512x512` (40.317 ms)
is added to show the radius scaling the rewrite was for. The remainder tracks the
other post-baseline commits and machine conditions — this run was taken under a
`powersave` governor at load ~4.5, so absolute figures are not comparable to the
April run's unknown conditions. Sweeps for the port should be taken on a quiesced
machine.

### Pinned

- `wgpu` stays on **29** and `pollster` on **0.4**. mabda 1.0.0 — the last Rust release of the AGNOS GPU foundation — links `wgpu ^29` and `pollster ^0.4`. Moving ranga to wgpu 30 / pollster 1.0 would resolve a second copy of wgpu into the tree and `mabda::GpuContext`'s types would no longer unify with ours. Since mabda's Rust line is frozen, these majors are frozen with it. Rationale recorded inline in `Cargo.toml`.

## [1.0.0] — 2026-04-02

P(-1) scaffold hardening pass — full audit and fix cycle across all modules.

### Changed

- **License**: AGPL-3.0-only → GPL-3.0-only (Cargo.toml, fuzz/Cargo.toml, CONTRIBUTING.md, README.md)
- **GPU backend**: migrated from raw wgpu to mabda 1.0 (pipeline cache, shader cache, buffer helpers, pollster)
  - `GpuContext` wraps `mabda::GpuContext` — no more `RefCell`, raw pointers, or custom `block_on`
  - `GpuBuffer` uses `mabda::buffer` helpers; now stores `PixelFormat` (was hardcoded Rgba8)
  - All `gpu_*` functions and `GpuChain` take `&mut GpuContext` (was `&GpuContext`)
  - All `expect()`/`unwrap()` in GPU code replaced with `Result` propagation
- **PixelBuffer**: fields now `pub(crate)` — use `data()`, `data_mut()`, `into_data()`, `width()`, `height()`, `format()` accessors
- **BufferPool::acquire**: best-fit allocation (was first-fit)
- **Flood fill**: rewritten with scanline algorithm (was naive 4-neighbor push with O(4n) stack bloat)
- **Median filter**: rewritten with histogram-based Huang approach — O(n*r) per channel (was O(n*r^2*log(r^2)))
- **Bilateral filter**: spatial Gaussian weights precomputed into table (was `exp()` per sample per pixel)
- **Vertical blur**: parallelized with rayon (horizontal was already parallel)
- **SSE2**: replaced fake SIMD stubs (extracted lanes → scalar) with real implementations
  - `compute_y_row` — `_mm_madd_epi16` horizontal dot-product, 2 pixels/iter
  - `grayscale` — `_mm_madd_epi16` luminance, 2 pixels/iter
  - `blend_row_normal_argb` — full SSE2 Porter-Duff, 2 pixels/iter
  - YUV-to-RGBA inverse (BT.601, BT.709, BT.2020, NV12) — 8 pixels/iter with U/V broadcast
- **GPU shaders**: `pack_rgba` uses `round()` (was `floor(x+0.5)`); saturation luminance standardized to BT.709; noise R/B decorrelated with independent Box-Muller pair
- **`chi_squared`**: returns `Result<f64, RangaError>` with length validation (was silent truncation)
- **`composite`**: `premultiply_alpha`/`apply_mask` use `div255` (was `/255`, off-by-one)
- **`blend`**: doc corrected to "straight alpha" (was "premultiplied"); SIMD slice checks upgraded from `debug_assert` to runtime guard
- **`perspective_transform`**: accepts `ScaleFilter` parameter (Nearest/Bilinear/Bicubic)
- **prakash**: 1.1.1 → 1.2 (version spec widened)
- **deny.toml**: removed stale advisory ignore, license entries, fuzz exception; cleaned cargo-vet imports

### Added

- `PixelFormat::checked_buffer_size()` — overflow-safe dimension validation
- `blend_row()` — row-level blend for any `BlendMode` (dispatches Normal to SIMD)
- ICC `para` curve types 1, 2, 4 (was only 0 and 3) — all 5 ICC parametric curve types now supported
- `#[inline]` on all `From` impls in `color.rs`, all `PixelBuffer`/`PixelView`/`PixelViewMut` accessors, `ToneCurve::apply`, ICC read helpers, filter scalars, spectral wrappers, GPU buffer accessors
- `#[must_use]` on `PixelBuffer::set_rgba`
- `scripts/bench-history.sh` — benchmark CSV tracking with Criterion output parsing
- `benches/history.csv` — baseline + post-hardening benchmark data

### Fixed

- **buffer_size overflow**: `checked_buffer_size()` prevents silent wrap on large dimensions; `buffer_size()` panics instead of wrapping
- **Perspective NaN**: degenerate projections (w≈0) now correctly skip pixels instead of indexing with NaN
- **Resize 0-dim source**: early return for `width==0 || height==0` prevents `usize` underflow
- **ICC parser**: `tag_count` capped at 1024, `grid_size` capped at 64 (prevents OOM from malformed profiles); `read_u32_be`/`read_u16_be`/`read_i32_be`/`read_s15fixed16` return `Result` (was panicking on OOB)
- **`rows()`/`rows_mut()`**: debug_assert rejects planar formats (Yuv420p, Nv12) instead of silently producing wrong slices
- **Bilateral filter**: `sigma=0` now returns error (was `-inf` coefficient)
- **Levels**: `gamma=0` now returns error (was `powf(infinity)`)
- **`color_temperature`**: NaN input returns neutral `[1.0, 1.0, 1.0]` (was propagating NaN through clamp)
- **GPU doctest**: `wgpu::Maintain::Wait` → `wgpu::PollType::wait_indefinitely()`

## [0.29.4] — 2026-03-29

### Changed

- **wgpu 24 → 29** — updated GPU compute backend to wgpu 29 (naga 29, wgpu-hal 29, wgpu-core 29)
  - Migrated `Maintain::Wait` → `PollType::wait_indefinitely()`
  - Migrated `Instance::new` to owned `InstanceDescriptor` via `new_without_display_handle()`
  - Migrated `request_adapter` from `Option` to `Result` return type
  - Migrated `PipelineLayoutDescriptor`: `push_constant_ranges` → `immediate_size`, `bind_group_layouts` now `Option`-wrapped
  - Removed extra `None` trace path argument from `request_device`
- **prakash 1.1.0 → 1.1.1** — bijli 1.0.1 dropped unnecessary soorat dependency; removed `default-features = false` workaround

## [0.29.3] — 2026-03-28

### Changed

- **Major dep upgrades**: ai-hwaccel 0.23.3→1.0.0, prakash 0.23.3→1.1.0 (hisab 0.22.4→1.3.0 transitive)
- Dependency update: wasm-bindgen 0.2.114→0.2.115, js-sys/web-sys 0.3.91→0.3.92, zerocopy 0.8.47→0.8.48
- Version bump from 0.24.3 to 0.29.3

### Added

- `#[must_use]` attributes on public functions in `composite` (11 functions), `pixel` (3 methods) — completing API hardening sweep
- Benchmarks: `auto_white_balance`, `delta_e_cie94`, `fill_solid`
- `Perspective` struct field documentation (3×3 homogeneous matrix element descriptions)
- `make msrv` target for local MSRV verification (`cargo +1.89 check --all-features`)
- 1.0.0 roadmap at `docs/development/roadmap.md` with pre-1.0 and post-1.0 backlog

### Fixed

- `make fuzz` now runs all 8 fuzz targets (was only 3: blend, convert, filter)
- Added `GPL-3.0-only` to `deny.toml` license allow list (hisab fixed SPDX identifier in 1.3.0)

## [0.24.3] — 2026-03-24

### Added

- **Spectral module** (`spectral` feature) — prakash integration for physically-based color science
  - `Spd` (spectral power distribution) type with CIE 1931 CMF integration
  - Bridged `From` conversions between `prakash::spectral::Xyz` and `ranga::color::CieXyz`
  - Convenience functions: `spd_to_xyz`, `xyz_to_cct`, `wavelength_to_xyz`, `d65_white`, `d50_white`, `blackbody_spd`
  - Re-exported standard illuminants (D65, D50, A, F2, F11), CIE 1931 2° CMFs, color rendering index
  - Re-exported `color_temperature_to_rgb`, `cct_from_xy` (inverse color temperature — new capability)
  - Re-exported high-precision sRGB gamma functions (`linear_to_srgb_gamma`, `srgb_gamma_to_linear`)
- **CieXyz white point constants** — `CieXyz::D65_WHITE` and `CieXyz::D50_WHITE` associated constants (always available, no feature gate)
- **ColorSpace::CieXyz variant** — added to the `ColorSpace` enum
- **Full test coverage sweep** — 37 new tests across 4 new test files (379 → 433 total)
  - `tests/edge_cases.rs` — 30 tests: error formatting, ARGB blend, composite_at_argb, histogram edge cases, filter edge cases, pixel edge cases, transform edge cases, convert edge cases
  - `tests/spectral.rs` — 17 tests: white points, XYZ roundtrip, SPD→XYZ, wavelength→XYZ, CCT, blackbody, CIE CMFs, illuminants, sRGB gamma, Wien peak, CRI
  - Expanded `tests/proptest.rs` — +7 property tests: Oklab/Oklch roundtrips, Delta-E CIE94, BT.2020 YUV, fade/wipe composites
- **Full benchmark coverage sweep** — 30 new benchmarks across 2 new + 4 expanded suites (~70 → ~108 total)
  - `benches/histogram.rs` (new) — luminance_histogram, rgb_histograms, equalize, auto_levels, chi_squared
  - `benches/icc.rs` (new) — srgb_v2 generation, ICC parse, ICC apply, ToneCurve gamma/table
  - `benches/spectral.rs` (new) — spd_to_xyz, xyz_to_cct, wavelength_to_xyz, blackbody_spd, color_temperature_to_rgb, cie_cmf_at, CRI, XYZ roundtrip
  - Expanded `benches/blend.rs` — all 12 blend modes group, ARGB pixel, ARGB row
  - Expanded `benches/color_convert.rs` — BT.2020 encode/decode, argb_to_nv12, rgba8_to_argb8, rgba8_to_rgb8, rgba8_to_rgbaf32
  - Expanded `benches/transform.rs` — bicubic resize, perspective transform
  - Expanded `benches/composite.rs` — unpremultiply, apply_mask, gradient_radial, gradient_angled, composite_at_argb

### Changed

- Added `spectral` feature flag (depends on prakash 0.23.3, spectral feature only)
- `full` feature now includes `spectral`
- **GPU batched dispatch** — `GpuChain` builder for chaining multiple GPU operations without CPU readback between steps (invert, grayscale, brightness_contrast, saturation, gaussian_blur, blend, noise, dissolve, fade, wipe, crop, resize, flip)
- **GPU noise generation** — `gpu_noise_gaussian` compute shader using PCG hash + Box-Muller transform for deterministic Gaussian noise
- **GPU transition shaders** — `gpu_dissolve`, `gpu_fade`, `gpu_wipe` compute shaders for cross-dissolve, fade-to-black, and horizontal wipe transitions
- **GPU geometry shaders** — `gpu_crop`, `gpu_resize` (nearest + bilinear), `gpu_flip_horizontal`, `gpu_flip_vertical` compute shaders
- **Visual regression tests** — 10 deterministic pixel-level regression tests: gradient blur smoothness, checkerboard resize, invert idempotency, premultiply roundtrip precision, Gaussian blur symmetry, HSL hue shift 360 identity, color balance neutral, crop+resize composition, Screen blend commutativity, YUV roundtrip color fidelity
- **Extended fuzz campaigns** — 5 new fuzz targets (blur, LUT, ICC, composite, transform) added to existing 3 (blend, convert, filter) for 8 total
- **Consistent error messages** — standardized all `InvalidFormat` errors to `"<operation>: expected <format>, got <actual>"` across filter, composite, histogram, convert, transform, and GPU modules
- Bumped `ai-hwaccel` from 0.21.3 to 0.23.3 — `HwReport` now exposes `gpu_free_memory_mb`, `gpu_utilization_percent`, `temperature_c`; `should_use_gpu()` checks free VRAM and GPU utilization before recommending offload
- Added field-level doc comments to all public struct fields (`LinRgba`, `Srgba`, `CieXyz`, `Cmyk`, `PixelBuffer`, `Affine`) — 100% public API documented
- Version aligned with prakash ecosystem at 0.23.3
- Roadmap fully cleared — all backlog items completed

## [0.21.4] — 2026-03-21

### Changed

- Bumped `ai-hwaccel` dependency from 0.20 to 0.21.3
- Cleaned up roadmap: removed completed items, added ai-hwaccel 0.21.3 review task

## [0.21.3] — 2026-03-21

### Fixed

- **Compositing precision** — replaced `>> 8` (divide-by-256) with proper `div255` rounding across all blend and composite paths (scalar, SSE2, AVX2, NEON), eliminating ~0.4% cumulative brightness loss per compositing pass
- **NEON brightness OOB read** — `simd_pixels` now rounds to multiple of 8 (matching `vld4_u8` stride), preventing buffer overread on aarch64
- **ARGB fast-path alpha** — `composite_at` and `composite_at_argb` fast-path now requires full opacity, preventing raw source alpha from bypassing opacity adjustment
- **ICC LUT index ordering** — CLUT indexing corrected to have B channel varying fastest per ICC spec, fixing color output for real-world LUT-based profiles
- **BT.709 Y coefficients** — changed from (54, 183, 18) sum=255 to (54, 183, 19) sum=256 so white correctly maps to Y=255
- **YUV420p odd-dimension buffer sizing** — `buffer_size()` and all conversion functions now use `div_ceil(2)` for chroma plane dimensions, fixing undersized buffers for odd-width/height images
- **Histogram `bins=0` panic** — `luminance_histogram()` now returns an error instead of panicking on zero bins
- **`auto_levels` color shift** — switched from luminance-based offset to per-channel min/max stretching, preventing color distortion
- **`auto_white_balance` extreme scale** — raised near-zero threshold from 0.5 to 5.0 and clamped scale factors to [0.5, 3.0]
- **GPU `pixel_count` truncation** — GPU pipeline functions now return an error instead of silently truncating images exceeding `u32::MAX` pixels
- **ICC tag offset overflow** — `parse_tag_table` now validates offset+size against profile length, preventing potential bounds bypass on crafted profiles
- **Gradient interpolation clamp** — `gradient_linear` and `gradient_linear_angled` now clamp interpolated values before u8 cast
- Added `debug_assert!` guards on NEON Y-plane coefficient values to catch u8 truncation
- Updated `srgb_v2_profile` docs to clarify gamma 2.2 is a v2 approximation of the piecewise sRGB TRC

### Added

- **Oklab/Oklch color space** — `Oklab` and `Oklch` types with bidirectional conversion to/from linear sRGB (Björn Ottosson standard matrices)
- **BT.2020 color space** — `rgba_to_yuv420p_bt2020()` and `yuv420p_to_rgba_bt2020()` for UHD/HDR video wide-gamut conversion
- **Bicubic resize** — `ScaleFilter::Bicubic` variant using Catmull-Rom kernel for high-quality image scaling
- **Perspective transform** — `Perspective` struct with `from_quad()` 4-corner mapping, `perspective_transform()` function
- **Gradient radial fill** — `gradient_radial()` for center-outward radial gradients
- **Gradient angled fill** — `gradient_linear_angled()` for linear gradients at arbitrary angles
- **Histogram equalization** — `histogram::equalize()` for automatic contrast enhancement via CDF mapping
- **Auto-levels** — `histogram::auto_levels()` for linear min/max luminance stretching
- **Auto white balance** — `filter::auto_white_balance()` using gray-world algorithm
- **Embedded sRGB ICC profile** — `icc::srgb_v2_profile()` generates a minimal sRGB v2 ICC profile for embedding
- **ICC LUT-based profiles** — `IccLutProfile` struct with `from_bytes()` parser and `apply()` for mft1/mft2 tag types
- **GPU 3D LUT shader** — `LUT3D` WGSL compute shader with trilinear interpolation
- **GPU hue shift shader** — `HUE_SHIFT` WGSL compute shader (RGB→HSL→shift→RGB)
- **GPU color balance shader** — `COLOR_BALANCE` WGSL compute shader with shadow/midtone/highlight weighting
- **SIMD brightness filter** — SSE2 (x86_64) and NEON (aarch64) accelerated `brightness()` with saturating add/sub
- **SIMD grayscale filter** — SSE2 and NEON accelerated `grayscale()` using BT.601 coefficients
- **SIMD Y-plane conversion** — SSE2 and NEON accelerated luminance computation for BT.601/BT.709/BT.2020
- **Cache-aware blur tiling** — Vertical blur pass processes 64-pixel-wide strips for L2 cache locality
- **Photoshop reference test suite** — 12 blend mode golden-value tests verified against Photoshop output

### Changed

- `ColorSpace` enum now includes `Bt2020` variant
- Roadmap updated with completion status for all 0.21.3 items

## [0.20.5] — 2026-03-21

### Added

- **ARGB8 blend** — `blend_pixel_argb()` and `blend_row_normal_argb()` for ARGB channel layout (aethersafta native format)
- **ARGB8 positioned composite** — `composite_at_argb()` for ARGB8 layer compositing without RGBA conversion

### Changed

- Version bump for aethersafta ecosystem migration

## [0.20.4] — 2026-03-21

### Added

- **Geometry transforms** (`transform` module) — crop, resize (nearest/bilinear), affine transform, flip horizontal/vertical
- **Compositing** (`composite` module) — premultiplied alpha, layer masks, dissolve/fade/wipe transitions, solid/gradient/checkerboard fill, positioned composite with clipping
- **Filters** — median, bilateral, vibrance, channel mixer, threshold, flood fill (6 new, 23 total)
- **API improvements** — `Display` for `PixelFormat`/`BlendMode`/`ColorSpace`, `rows()`/`rows_mut()` iterators, `get_rgba()`/`set_rgba()` accessors, `from_view()`

### Changed

- `deny.toml` updated for cargo-deny v2 compatibility
- Coverage gate set to 75% in codecov.yml
- Documentation expanded: performance guide, testing guide, troubleshooting guide, comprehensive README

### Fixed

- Odd-dimension YUV/NV12 conversion OOB panic (chroma subsampling clamped)
- SIMD blend equivalence test tolerance for alpha rounding

## [0.20.3] — 2026-03-20

First release. Core image processing primitives for the AGNOS creative suite,
replacing inline implementations across rasa, tazama, and aethersafta.

### Color Science

- `LinRgba`, `Srgba`, `Hsl` types with bidirectional sRGB↔linear gamma conversion
- `CieXyz`, `CieLab` types with full sRGB↔XYZ↔Lab conversion chain (D65)
- `Cmyk` type with naive CMYK↔sRGB conversion
- Display P3 ↔ sRGB linear conversion (3x3 matrix)
- Color temperature (Kelvin → RGB multipliers, Tanner Helland approximation)
- Delta-E color distance: CIE76, CIE94, CIEDE2000 (Sharma 2005)
- `ColorSpace` enum: Srgb, LinearRgb, DisplayP3, Bt601, Bt709

### Pixel Buffer

- `PixelBuffer` with format validation, 6 formats (RGBA8, ARGB8, RGB8, YUV420p, NV12, RgbaF32)
- `PixelView` / `PixelViewMut` — zero-copy borrowed views for downstream integration
- `BufferPool` — reusable allocation pool for video frame pipelines

### Blend Modes

- 12 Porter-Duff blend modes: Normal, Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, SoftLight, HardLight, Difference, Exclusion
- SIMD acceleration: SSE2 (2px/iter), AVX2 (4px/iter, runtime detected), NEON (8px/iter)
- `blend_pixel` (single pixel) and `blend_row_normal` (row-level, SIMD-accelerated)

### Color Conversion

- RGBA↔YUV420p BT.601 (fixed-point)
- RGBA↔YUV420p BT.709 (fixed-point, HD video standard)
- ARGB→NV12, NV12→RGBA
- RGB8↔RGBA8, ARGB8↔RGBA8, RgbaF32↔RGBA8
- Odd-dimension safe (chroma subsampling clamped for non-even sizes)

### Filters (17 total)

- In-place: brightness, contrast, saturation, levels, curves, grayscale, invert
- Spatial: Gaussian blur (separable), box blur (separable), unsharp mask
- Color: hue shift (HSL), color balance (shadows/midtones/highlights)
- Effects: vignette, noise (Gaussian + salt-and-pepper with deterministic PRNG)
- Grading: 3D LUT application (.cube file parser with trilinear interpolation)
- Parallel blur via rayon (`parallel` feature)

### Histogram

- Luminance histogram (BT.601, configurable bins)
- Per-channel RGB histograms (256 bins, normalized)
- Chi-squared distance metric

### ICC Profiles

- Matrix-based ICC v2/v4 profile parser (pure Rust, no C deps)
- TRC support: gamma curves and lookup tables
- `IccProfile::apply()` for RGB→XYZ transform via parsed matrix + TRC

### GPU Compute (`gpu` feature)

- `GpuContext` — wgpu device/queue management (Vulkan + Metal)
- Pipeline caching — compiled shaders stored for reuse across calls
- WGSL compute shaders: blend (all 12 modes), invert, grayscale, brightness/contrast, saturation, Gaussian blur (horizontal + vertical)
- `GpuBuffer` — upload/download with async readback support
- GPU/CPU equivalence tests verify correctness within rounding tolerance

### Hardware Detection (`hwaccel` feature)

- `probe()` — GPU/Vulkan detection via ai-hwaccel
- `should_use_gpu(w, h)` — automatic CPU/GPU crossover recommendation

### Infrastructure

- CI/CD: 10-job GitHub Actions pipeline (lint, security, supply chain, test matrix, MSRV 1.89, coverage, benchmarks, docs, semver)
- Release workflow: tag-triggered 5-target build matrix, crates.io publish, GitHub Releases
- Supply chain: cargo-deny license allowlist, cargo-vet config
- 238 tests: 116 unit, 15 integration, 15 proptest, 92 doc-tests
- 37 criterion benchmarks across 6 suites (blend, convert, color science, filters, GPU)
- 3 fuzz targets (blend, convert, filter)
- 3 runnable examples
- 94.6% code coverage (75% CI gate)
- `#[non_exhaustive]` on all public enums
- All `unsafe` blocks documented with `// SAFETY:` comments
- SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, threat model, 2 ADRs, migration guide

### Feature Flags

- `simd` (default) — SSE2/AVX2/NEON blend acceleration
- `gpu` — wgpu compute shaders
- `hwaccel` — GPU detection via ai-hwaccel
- `parallel` — rayon row-parallel blur
- `full` — all features
