# ranga — Roadmap

> What is left to do and in what order. Live state is in [`state.md`](state.md). The
> per-item parity inventory these milestones are drawn from is in
> [`parity-rust-v-cyrius.md`](parity-rust-v-cyrius.md) (review of 2026-11-02, ranga
> 2.1.1). Shipped work is recorded in the [CHANGELOG](../../CHANGELOG.md) and git
> history. Non-obvious constraints are in [`../architecture/`](../architecture/).

## Where things stand

The port shipped as **2.0.0** and is at **2.3.0** on cyrius 6.7.18. The 2026-11-02 review
of all 303 public Rust items, updated for 2.1.2 and 2.2.0, found:

- 211 at parity;
- 5 improved on Rust;
- 29 that diverge on purpose, with the reason written down;
- 19 partial and 33 missing;
- **6 that compute something different from Rust without a documented reason**. All
  six are GPU and are scheduled in 2.5.x.

**2.2.0 closed milestone 2.2.x, "bit-exact again".**
- `scripts/oracle` checks the port against the Rust 1.0.1 crate on identical inputs:
  2.1 million values, and every output boundary of the f32 → byte conversions over
  all of [0, 1].
- Every CPU computation it covers matches Rust bit for bit. The exception is three
  libm-backed f32 fields, accepted within 2 ULP by ADR 002.

The 2.1.2 P(-1) review ([`../audit/2026-11-02-audit.md`](../audit/2026-11-02-audit.md))
had already shown that none of these divergences came from the toolchain bumps.

**2.3.0 closed milestone 2.3.x, "verification parity"** apart from three decisions
(below). The port now has Rust's 8 fuzz targets in CI under allocator redzones, its 31
properties, its integration, edge-case and Photoshop tests, golden vectors that
capture the oracle's Rust outputs in `tests/`, and 108 of its 134 benchmarks (the 26
left are GPU). Still missing: examples, and any Cyrius consumer.

Several of the original 2.0.0 criteria were never met, and they are carried below.

## Versioning

- **Patch (`2.N.x`):** toolchain and dependency bumps, and fixes within the current
  milestone.
- **Minor (`2.N.0`):** opens the next milestone below.

Milestones are ordered by what unblocks what, not by size. Correctness came first
(2.2.x, done) because the oracle that defines it (`rust-old/`) is due to be deleted.
Verification is next, because the oracle's evidence must be captured in tests before
that happens.

**Rule for every item:** the port's bar is "matches what Rust did", so diverge only with
an ADR. Each SUSPECT item is either fixed to match Rust or given an ADR, and each
MISSING item is either ported or given an ADR. Sizes are S (hours), M (a day or two) and
L (more).

---

## 2.3.x — Verification parity

2.3.0 closed this milestone's main work: all 8 fuzz targets under `--poison` in CI,
all 31 properties, every Rust integration and edge-case test, golden vectors from the
oracle, the oracle in CI, and Rust's 108 CPU benchmarks on Rust's workloads. What is
left is three decisions:

- [ ] **(S) Decide on Rust's blur darkening.** Rust 1.0.1's `gaussian_blur` and
      `box_blur` truncate between passes, so a uniform image comes back up to 2
      counts darker (gaussian in 979 of 2,816 measured cases, box in 4,278 of
      13,824). The port matches Rust, and `fuzz_blur` asserts Rust's bounds. Either
      keep parity, or round between passes under an ADR and tighten the fuzz bounds
      to exact.
- [ ] **(S) GPU benchmarks.** Rust's 26 GPU benchmarks (`gpu.rs`, half of them CPU
      baselines) need a GPU host to mean anything. Decide whether to run them on the
      AMD dev host only, recorded like the CPU series, or to wait for a GPU runner.
- [ ] **(S) NVIDIA coverage.** `gpu_context_new_nvidia` has no test at all; every
      gated GPU suite opens an AMD context. Decide whether it needs a host to test on,
      or an ADR that it is unverified.

**Done when** those three are decided.

## 2.4.x — Surface completion (CPU)

- [ ] **(M–L) Decide serde first.** Rust derives Serialize/Deserialize on 14 types:
      the 8 colour structs, `PixelFormat`, `ColorSpace`, `BlendMode`, `ToneCurve`,
      `Affine` and `Perspective`. Either write the codecs (the wire form is serde's,
      i.e. variant names such as "ColorDodge"), or write an ADR that puts serde out
      of scope.
      - At 6.7.18, `#derive(Serialize)` on an enum works and emits exactly that
        variant name. It was checked for 2.1.2, so the four enums can be derived.
      - The structs are hand-rolled byte layouts, so their codecs would be
        hand-written, on bayan.
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
- [ ] **(S) Spectral names:** decide whether to provide the Rust names for prakash's
      three renamed re-exports (`CIE_1931_2DEG`, `VISIBLE_MIN_NM` and
      `VISIBLE_MAX_NM`). `spectral.cyr` has documented the renames since 2.1.2.
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

The baseline is the same-session comparison of all 108 CPU benchmarks in
[`benchmarks-rust-v-cyrius.md`](../benchmarks-rust-v-cyrius.md) (2.3.0: median 8.0×
slower than Rust). Ratios below are from that run.

- [ ] **(S–M) `memcpy` and `memset` are byte loops** in the 6.7.18 stdlib (the x86
      `rep movsb` path was removed for aarch64's sake). That is most of
      `buffer_pool_acquire_release` (127×), `crop` (53×), `flip_vertical` (32×),
      `flip_horizontal` (18×) and `fill_solid` (18.5×). File an architecture-guarded
      fast path upstream, and meanwhile use a word-at-a-time copy and fill in ranga's
      hot paths.
- [ ] **(S–M) Inline the per-byte f32 helper calls** in `fade` (57×), `dissolve`
      (50×), the angled and radial gradients (37–42×), `vibrance` (27×),
      `color_balance` (21×), `channel_mixer` and `contrast` (16–17×), as was done for
      the blurs in 2.0.x. Cyrius does not inline.
- [ ] **(S) `equalize` (31×) and `auto_levels` (41×):** inline the helper calls in the
      f32 apply pass. This is call overhead, not a SIMD gap.
- [ ] **(M) A vector kernel for the YUV inverses** (37×) and `nv12_to_rgba` (52×). Rust
      has SSE2 at 8 pixels per iteration; these have had a loop hoist only.
- [ ] **(M) f32 SIMD for `contrast` and `saturation`.**
- [ ] **(S–M) `hue_shift`:** remove its two heap allocations per pixel.
- [ ] **(L) Vectorised or tiled separable blur.** About 7× against today's rustc and
      22× against the frozen series (whose blur was 3.3× faster). Rust auto-vectorises
      the passes.
- [ ] **(L) 256-bit AVX2,** cyrius-port-plan §3 item 4. It comes after all of the
      above.
- Upstream, not ranga: prakash's `color_rendering_index` takes 310 µs against Rust's
  0.84 µs (367×).

**Done when:** the items above are done and re-measured, both sides, on a quiet machine.
(The old criterion, "the blurs and the histogram ops are no longer the worst ratios",
no longer describes the table: on Rust's workloads the worst are the byte-loop and
helper-call items above.)

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
needed. A series of bumps closes with a P(-1) review filed under `docs/audit/`; the
first, for 6.6.9 → 6.7.18, is
[`../audit/2026-11-02-audit.md`](../audit/2026-11-02-audit.md).

1. Move `cyrius` in `cyrius.cyml`. Move each dep tag to the newest release that pins
   the same toolchain line or an older one.
2. Delete the lock and re-resolve:
   `rm cyrius.lock && cyrius deps --features spectral,hwaccel,gpu`. Run with all
   features, the way CI does. Re-resolving over an existing lock kept the stale commit
   pins (seen at 6.6.20).
3. Check that new stdlib and dep top-level names do not collide with `src/`. At 6.7.18,
   `bayan`'s `_TC_SIZE` broke `icc.cyr`. Also look for new
   `cyrius deps` notes about files outside the declared footprint.
4. Run `cyrius build`, `cyrius test` and the lint gate, then
   `scripts/bench-history.sh`, which appends to `bench-history.csv`. Then run
   `scripts/version-bump.sh X.Y.Z` and `cyrius distlib --check`. Compare the warnings
   against the previous pin.
5. Look for workarounds the toolchain has outgrown. At 6.7.18 the 2^63 trig guard
   had become a divergence. A cross-toolchain diff of the same probes, built under
   the old and new pins, shows exactly what a bump changed. Run
   `scripts/oracle/run.sh` too: a bump that moves a computation away from Rust fails
   it.
6. Until 2.3.x lands, `cyrius fuzz` runs the scaffold stub. **Do not report it as
   verification.**

## Out of scope

- WASM / WebGPU: blocked on the Cyrius WASM backend.
