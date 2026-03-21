# Troubleshooting

## Build Issues

### `error[E0658]: use of unstable feature`
You need Rust 1.89+. Check with `rustc --version` and update with `rustup update`.

### `wgpu` build failures
The `gpu` feature requires Vulkan or Metal SDK headers. On Linux, install
`libvulkan-dev`. On macOS, Xcode provides Metal automatically. Disable with
`--no-default-features` if you don't need GPU.

### `ai-hwaccel` detection returns no GPU
ai-hwaccel probes sysfs and CLI tools. In containers or headless CI, GPU
detection will correctly return `has_gpu: false`. GPU tests are skipped
automatically in this case.

## Runtime Issues

### Panic on odd-dimension YUV conversion
Fixed in v0.20.3. All YUV/NV12 conversions now clamp chroma indices for
non-even image dimensions.

### GPU operations slower than CPU
Expected for images below ~1080p due to upload/download overhead. Use
`should_use_gpu(w, h)` from the `hwaccel` module, or check the
[Performance Guide](performance.md) for crossover points.

### SIMD blend results differ by +/-1 from scalar
This is expected. The SIMD alpha fixup uses integer division in a different
order than the scalar path, which can produce +/-1 rounding differences on
the alpha channel. Both results are correct within the precision of 8-bit
color.

### 3D LUT parsing fails
Ensure the .cube file uses `LUT_3D_SIZE` (not `LUT_1D_SIZE`), size >= 2,
and has exactly `size^3` data lines. Comment lines starting with `#` and
`TITLE`/`DOMAIN_MIN`/`DOMAIN_MAX` headers are ignored.

### ICC profile parsing fails
Only matrix-based profiles (rXYZ/gXYZ/bXYZ + TRC) are supported. LUT-based
ICC profiles will return `RangaError::InvalidFormat`. Use the `lcms2` crate
directly for LUT profiles.

## CI Issues

### `cargo-deny` schema errors
ranga targets cargo-deny v2. If running locally with v0.x, some fields may
differ. The CI action `EmbarkStudios/cargo-deny-action@v2` uses the correct
version.

### Coverage below gate
The codecov.yml gate is 75%. Run `cargo llvm-cov --summary-only` to check
local coverage. GPU-only code paths may not execute in headless CI.
