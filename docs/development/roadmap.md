# ranga — Roadmap

> What is left to do and in what order. Live state is in [`state.md`](state.md). The
> per-item parity inventory these milestones are drawn from is in
> [`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md) (review of 2026-11-02, ranga
> 2.1.1). Shipped work is recorded in the [CHANGELOG](../../CHANGELOG.md) and git
> history. Non-obvious constraints are in [`../architecture/`](../architecture/).

## Where things stand

The port shipped as **2.0.0** and is at **2.1.1** on cyrius 6.7.18. The 2026-11-02 review
of all 303 public Rust items found:

- 193 at parity;
- 5 improved on Rust;
- 28 that diverge on purpose, with the reason written down;
- 19 partial and 33 missing;
- **25 that compute something different from Rust without a documented reason**.

The verification surface is far behind Rust:
- no fuzzing;
- no property tests;
- 35 of 134 benchmarks;
- no examples;
- no Cyrius consumer.

Several of the original 2.0.0 criteria were never met, and they are carried below.

## Versioning

- **Patch (`2.N.x`):** toolchain and dependency bumps, and fixes within the current
  milestone.
- **Minor (`2.N.0`):** opens the next milestone below.

Milestones are ordered by what unblocks what, not by size. Correctness comes first
because the oracle that defines it (`rust-old/`) is due to be deleted.

**Rule for every item:** the port's bar is "matches what Rust did", so diverge only with
an ADR. Each SUSPECT item is either fixed to match Rust or given an ADR, and each
MISSING item is either ported or given an ADR. Sizes are S (hours), M (a day or two) and
L (more).

---

## 2.2.x — Bit-exact again

Fix every measured divergence, and make the measurement repeatable.

- [ ] **(M) Check in the differential oracle.** Add a small Rust driver outside
      `rust-old/` (e.g. `scripts/oracle/`, depending on `../../rust-old` by path),
      plus a Cyrius probe and a diff script. Run it once per release.
      - Every finding below was measured with a throwaway version of it.
      - It is the only thing that catches 1-ULP and half-step errors: the old sampled
        sweeps called four of these paths byte-exact.
- [ ] **(M) Single-precision colour byte paths:** `linear_to_srgb`,
      `lin_rgba_to_srgba`, `hsl_to_srgba` (including its `s < 1e-6` compare),
      `cmyk_to_srgba` and `rgbaf32_to_rgba8`. Use the `: f32` typed-local pattern
      `srgb_to_linear` already uses, and sweep every f32 in [0,1] against the oracle.
- [ ] **(M) Single-precision colour f32 outputs:** `srgba_to_hsl`, `srgba_to_cmyk`,
      `color_temperature`, `oklab_to_oklch` and `oklch_to_oklab`. Fix them, or write an
      ADR accepting ULP-level differences in fields that never round to a byte.
      `hue_shift` depends on `srgba_to_hsl`.
- [ ] **(S) `affine_inverse`:** restore Rust's association, `(c*ty - d*tx)*inv` and
      `(b*tx - a*ty)*inv`. Ordinary scale+translate transforms currently sample the
      wrong source column.
- [ ] **(S) `affine_is_identity`:** make a NaN field fail, using Rust's all-`<` form.
- [ ] **(S–M) `levels`:** compute `1/gamma`, the pow result, `*255` and `+0.5` in f32.
- [ ] **(S) `bilateral`:**
      - compute the coefficients and exp arguments in f32;
      - return `RG_ERR_OTHER` for σ ≤ 0 (as `levels` already does);
      - let a NaN σ through as Rust does, or write an ADR.
- [ ] **(S) `hue_shift`:** do the hue add and `rem_euclid` in f32. This lands after
      `srgba_to_hsl`.
- [ ] **(S) ICC parse regressions,** all caused by `fl_alloc(0)` / `fl_calloc(0)`
      returning 0 since cyrius 6.6.7:
      - a zero-tag profile should return `RG_ERR_INVALID_FORMAT`;
      - an `mft2` with zero input entries should parse with identity curves;
      - grid size 1 should be accepted, or rejected under an ADR.

      Add a test for each.
- [ ] **(S) ICC error codes:** some malformed input returns `RG_ERR_PARSE` where Rust
      returns InvalidFormat for all of it. Unify, or document the split.
- [ ] **(S) `src/blend.cyr:143`:** write it as `_bl_sub(one, ganita_f32_min(...))`.
      It is correct today, but only because the compiler passes f32 bits through an
      `: i64` call.
- [ ] **(S) Contradictions pinned by tests.** For each, match Rust or write an ADR:
      - the error strings: Rust's InvalidFormat reads "invalid pixel format: …", and
        `Other` carries its message;
      - `PixelBuffer::new` reporting overflow as `RG_ERR_ALLOCATION`, which has needed
        an ADR since M7.
- [ ] **(S) ADR 001:**
      - extend it to the BT.709, BT.2020 and NV12 inverses, which diverge the same way
        and whose tests already pin the values;
      - correct its "all five failed to reproduce" line.
- [ ] **(S) Robustness found by the review:**
      - `convert.cyr:103` ignores `_cv_y_row`'s return value;
      - `buffer_pool_acquire` accepts a negative size;
      - `BufferPool` allocates all `max_buffers` slots eagerly (`buffer_pool_new(1<<40)`
        returns 0);
      - `lut3d_new` has no overflow guard on `size³·12`;
      - `flood_fill`'s stack has no capacity check;
      - bilinear sampling of a 1-pixel source with a non-finite inverse reads past the
        buffer.
- [ ] **(S) Stale source comments the review listed:**
      - `error.cyr` (INVALID_PARAMETER "levels gamma = 0", "logged at the failure site");
      - `color.cyr:564` (Oklab "f32 throughout");
      - `convert.cyr:4-8`;
      - `transform.cyr:446-449`;
      - `composite.cyr`'s NaN-ordering and `f64_cos` notes;
      - `blend.cyr:4-9` and `:261-263`;
      - `hwaccel.cyr:1`;
      - `spectral.cyr:9-17`.

**Done when:** the parity doc has no SUSPECT items, and an oracle sweep is clean on
every u8 output path.

## 2.3.x — Verification parity

Make the test, fuzz and benchmark surfaces mean what Rust's meant.

- [ ] **(M) A real fuzz harness.** `tests/ranga.fcyr` is still the scaffold stub.
      - Port the 8 Rust targets behind one dispatcher with a seeded input generator.
      - Start with `fuzz_icc` (the only untrusted-input parser), then `fuzz_convert`
        and `fuzz_transform` (size arithmetic).
      - `fuzz_lut` can cover only `apply_lut3d` unless `from_cube` is ported (2.4.x).
      - Run `cyrius fuzz --poison` in CI.
- [ ] **(M) A property suite (`tests/properties.tcyr`)** with a seeded generator,
      porting the 31 `proptest.rs` properties.
      - First the three with no coverage at all: `lab_roundtrip`, `oklch_roundtrip`
        and the BT.2020 grey round trip.
      - Then the 25 that were reduced to one fixed example.
      - Sweep exhaustively where the domain is cheap, as `color.tcyr` already does.
- [ ] **(S) Golden vectors:**
      - the 7 Photoshop reference pixels from `photoshop_reference.rs`;
      - the 4 missing visual-regression checks: blur monotonicity, 2-D blur symmetry,
        Screen commutativity and hue 360 identity.
- [ ] **(S–M) `tests/integration.tcyr`:** the 6 multi-step pipelines from
      `integration.rs` and `crop_resize_composition`. Each step is unit-tested today,
      but no chain is.
- [ ] **(M) Unit, edge and doctest gaps,** highest risk first:
      - `resize_to_1x1` (the divide-by-zero guard);
      - any resize downscale, and the Bilinear/Bicubic sample paths;
      - `perspective_inverse_roundtrip`;
      - XYZ/P3 round trips on colours other than white;
      - CMYK with k ≠ 0, and the HSL branch for l > 0.5;
      - `get_rgba` on a pixel with four distinct channels;
      - `view_mut` write-through, and pool re-zeroing;
      - alpha preservation in threshold and salt-and-pepper;
      - NV12 at odd sizes, and a 1×1 YUV image;
      - the convert white/black checks through the public API.
- [ ] **(S) Make GPU coverage visible.**
      - Move the hardware-free checks (null context, shape guards, stub returns) out of
        the `ctx != 0` gate.
      - Report how many assertions were skipped.
      - Say in `state.md` that the full count is an AMD-host count.
      - Decide whether the NVIDIA backend needs coverage.
- [ ] **(S) Spectral re-export tests:** record that prakash's own suite owns the 8
      `spectral.rs` tests for re-exported items, or port them.
- [ ] **(M) Benchmarks.**
      - Add the 74 missing CPU benches, using Rust's names (`rgba_to_yuv420p_bt601_1080p`)
        and workloads (composite_at 640×480; opacity 200 for blend).
      - Re-measure both sides on a quiet machine.
      - Rewrite `docs/benchmarks-rust-v-cyrius.md`, whose table and prose are stale.
      - The 26 GPU benches wait on a decision about GPU hardware in CI.
- [ ] **(S) CI hygiene.**
      - Replace the per-file lint loop with
        `cyrius lint --strict --strict-deferrals src/*.cyr tests/*.tcyr 2>&1`. Since
        6.6.5 lint takes 1..N files, and the strict flags exit non-zero, so the old
        loop's comment is stale.
      - Add the fuzz step.
      - Rename the duplicate `b6`/`b8` globals in `filter_point.tcyr`.

**Done when:**
- all 8 fuzz targets run under `--poison` in CI;
- every Rust test case is covered, or given an ADR;
- all 108 CPU benchmarks are recorded against the frozen Rust series.

## 2.4.x — Surface completion (CPU)

- [ ] **(M–L) Decide serde first.** Rust derives Serialize/Deserialize on 14 types:
      the 8 colour structs, `PixelFormat`, `ColorSpace`, `BlendMode`, `ToneCurve`,
      `Affine` and `Perspective`. Either write the codecs by hand (on bayan; the wire
      form is serde's, i.e. variant names such as "ColorDodge"), or write an ADR that
      puts serde out of scope.
- [ ] **(S) Clone and equality for heap records:**
      - `affine_clone`/`_eq`, `perspective_clone`/`_eq` and `lut3d_clone`;
      - a clone for the colour structs.
- [ ] **(S) Writable records:**
      - setters for the colour structs, with the Hsl and Cmyk layouts added to the
        `color.cyr` header;
      - a Table accessor for `ToneCurve`;
      - `pixel_buffer_into_bv` (Rust's `into_data`).
- [ ] **(S–M) ICC constructors:** a way to build `IccProfile` / `IccLutProfile` from
      a matrix and curves, as Rust's pub fields allowed.
- [ ] **(M) `Lut3d::from_cube`:** port the `.cube` parser, or write an ADR. Either way,
      fix `docs/guides/features/filters.md` and `migration-guide.md`, which advertise
      it.
- [ ] **(S) Spectral names:** document prakash's three renamed re-exports
      (`CIE_1931_2DEG`, `VISIBLE_MIN_NM` and `VISIBLE_MAX_NM`), or provide the Rust
      names.
- [ ] **(S) Namespace decisions:** whether `xorshift64_*` stays public (it was
      private in Rust), and whether `bytevec`'s unprefixed `bv_*` functions need a
      prefix to avoid colliding with consumers.

**Done when:** outside the GPU layer, every MISSING item in the parity doc is either
ported or given an ADR.

## 2.5.x — GPU

See [architecture note 001](../architecture/001-mabda-native-gpu-constraints.md) for
what mabda 4.2.0 accepts.

- [ ] **Decide the wgpu story first.** ranga carries all 21 WGSL sources and dispatches
      none of them, and mabda's context-aware wgpu dispatch is a stub, so every op
      fails on wgpu. Either:
      - wire the WGSL through mabda's `compute_dispatch(device, queue, …)`, selected
        on `backend_kind`, which needs a C-launcher test harness (L); or
      - stop claiming a fallback in the README, CHANGELOG, guides and source comments
        (S).
- [ ] **(S) Round half to even** when packing, to match WGSL `round`. `gpu_fade` and
      `gpu_dissolve` are off by one at exact halves. Use `OpSelect`, which mabda 4.1.0
      can now lower, and add odd-byte test inputs.
- [ ] **(S) Geometry ops:**
      - dispatch in 2D (16×16, guarded by `_gk_guard_open2`), which removes the
        65,535-group ceiling that makes 4096² fail;
      - a degenerate crop should return 0×0, and resize to a 0 dimension should return
        Ok(0×0).
- [ ] **(S) Error hygiene:**
      - `gpu_error_message` for FORMAT and UNSUPPORTED (both say "no error");
      - guards in Rust's order: null, then format, then dimensions;
      - consistent `err_out` on dimension mismatch;
      - stubs that return the same code the native path does.
- [ ] **(M) `GpuChain`, natively,** with ping-pong buffers kept on the GPU (13 items).
      `hwaccel.cyr`'s cost model already assumes it.
- [ ] **(S) `GpuContext::new_with_hwaccel`,** plus a bundle profile that combines gpu
      and hwaccel.
- [ ] **(M) Native `gpu_gaussian_blur`:**
      - mabda 4.1.0 lowers loops and phi, so add loop/phi helpers to `gpu_spirv.cyr`
        and port `build_gaussian_kernel`;
      - until then, radius 0 should return a clone, as Rust did (S).
- [ ] **(M) Bilinear `gpu_resize`.** The module exceeds the 256-id cap (reason code
      -25); register pressure is not the cause. Split it into two separable passes, or
      reduce the ids. Re-pin the test on the reason code.
- [ ] **(M–L) Native `gpu_blend`:** one kernel per mode, so each fits the id cap.
      `OpFDiv` and `OpSelect` are now available, and the nested ifs need flattening.
- [ ] **(S) Retire pre-4.1.0 workarounds:** host-side reciprocals, the `count > idx`
      operand swap and the underflow conjunction.
- [ ] **(S–M) Adapter name,** from `wgpu_adapter_get_info` on wgpu, or
      `native_amdgpu_query_dev_info` natively.
- [ ] **(S) Leaks:**
      - the context's cache vecs are never freed;
      - `gpu_buffer_upload` leaks the handle when a later allocation fails.
- **Blocked upstream in mabda:**
  - `gpu_noise_gaussian`: no Log, Sin or Cos in the GLSL.std.450 lowering;
  - async readback: no API;
  - nested selections.

**Done when:**
- all 14 operations run natively;
- `GpuChain` is ported;
- wgpu either works or is not claimed.

## 2.6.x — Performance

The baseline is the frozen Rust 1.0.1 criterion series in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md).

- [ ] **(L) Vectorised or tiled separable blur.** Blurs run about 21× slower than Rust.
      Rust auto-vectorises the passes. The int→f32 conversion is not the bottleneck:
      removing it entirely would at best halve the blur.
- [ ] **(S) `equalize` (19.5×) and `auto_levels` (23×):** inline the per-channel helper
      calls. This is call overhead, not a SIMD gap.
- [ ] **(M) f32 SIMD for `contrast` and `saturation`** (12–15× slower).
- [ ] **(S–M) `hue_shift`:** remove its two heap allocations per pixel.
- [ ] **(M) A vector kernel for `yuv420p_to_rgba`.** It has had a loop hoist only.
- [ ] `crop` and `flip_vertical` are still memcpy-bound. The real fix is upstream (the
      `fl_calloc` zeroing), and the consumer workaround is already in place.
- [ ] **(L) 256-bit AVX2,** cyrius-port-plan §3 item 4. It comes after all of the
      above.

**Done when:** the blurs and the histogram ops are no longer the worst ratios, as
measured on a quiet machine.

## 2.7.x — Docs, examples, audit, consumers

- [ ] **(M) Port the 11 Rust examples** to runnable `.cyr` files in `docs/examples/`.
      `gpu_compute` and `spectral_analysis` need their features enabled.
- [ ] **(M) Rewrite the Rust-era docs:**
      - `docs/guides/features/*.md`, `testing.md`, `performance.md` and
        `troubleshooting.md`;
      - `docs/architecture/overview.md`, which still describes wgpu and
        `hwaccel::should_use_gpu`;
      - the README's GPU claims;
      - `threat-model.md`.
- [ ] **(L) Security audit** (`docs/audit/YYYY-MM-DD-audit.md`): the ICC parser, the
      raw-offset buffer accessors and the size arithmetic. Do it after the fuzz harness
      lands.
- [ ] **(L, outside this repo) A first Cyrius consumer.** The four named candidates
      (rasa, tazama, aethersafta, soorat) are all still Rust crates on ranga 0.20.5,
      0.29 or 1.x. aethersafta's compositor is the likeliest first port. The
      alternative is to drop the criterion by decision.

## 2.x — Closing the port: retire `rust-old/`

These criteria replace the unmet 2.0.0 list. `rust-old/` is the parity oracle, so
deleting it deletes the oracle. **Capture what it knows into tests first.**

- [ ] Zero SUSPECT items, and every MISSING item ported or given an ADR (2.2–2.5).
- [ ] Every Rust test case covered or given an ADR, with golden vectors from the oracle
      checked into `tests/` (2.3).
- [ ] All 8 fuzz targets in CI, and the security audit filed (2.3, 2.7).
- [ ] All CPU benchmarks recorded against the frozen Rust series (2.3).
- [ ] At least one Cyrius consumer green, or the criterion dropped by decision (2.7).
- [ ] Then delete `rust-old/`, with a CHANGELOG entry naming the
      `git checkout 1.0.1 -- rust-old/` recovery command.

## Ongoing — toolchain and dependency bumps

Each pin move is a patch release. The steps below are what the 2.0.4 → 2.1.1 bumps
needed:

1. Move `cyrius` in `cyrius.cyml`. Move each dep tag to the newest release that pins
   the same toolchain line or an older one.
2. Delete the lock and re-resolve:
   `rm cyrius.lock && cyrius deps --features spectral,hwaccel,gpu`. Run with all
   features, the way CI does. Re-resolving over an existing lock kept the stale commit
   pins (seen at 6.6.20).
3. Check that new stdlib and dep top-level names do not collide with `src/`. At 6.7.18,
   `bayan`'s `_TC_SIZE` broke `icc.cyr`. Also look for new
   `cyrius deps` notes about files outside the declared footprint.
4. Run `cyrius build`, `cyrius test`, the lint gate and `cyrius bench`. Then run
   `scripts/version-bump.sh X.Y.Z` and `cyrius distlib --check`. Compare the warnings
   against the previous pin.
5. Until 2.3.x lands, `cyrius fuzz` runs the scaffold stub. **Do not report it as
   verification.**

## Out of scope

- WASM / WebGPU: blocked on the Cyrius WASM backend.
