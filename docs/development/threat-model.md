# Threat Model

## Overview

ranga is a pure-computation image processing library. It performs no network
access, no file I/O, and contains no C FFI. Its threat surface is limited to
the data passed in by callers.

## Trust Boundaries

### 1. Pixel Buffer Input (Primary Boundary)

**Source**: Downstream consumers (rasa, tazama, aethersafta) pass
user-controlled image data.

**Threats**:
- Incorrect dimensions (width/height do not match data length)
- Unsupported or mismatched pixel formats
- Extremely large dimensions causing allocation pressure

**Mitigations**:
- `PixelBuffer::new()` validates `data.len() == format.buffer_size(w, h)`
- All operations check `buf.format` before processing
- No implicit allocation beyond what the caller provides

### 2. Arithmetic Overflow

**Source**: Fixed-point BT.601 color math and blend computations.

**Threats**:
- Integer overflow in `u16`/`i16`/`i32` intermediate values
- Out-of-range results in color space conversion

**Mitigations**:
- All intermediate arithmetic uses wider types (`u16`, `i16`, `i32`)
- Results are clamped to valid ranges before narrowing casts
- Lookup tables (levels, curves) are bounds-checked by array indexing

### 3. Format Conversion (Lossy)

**Source**: YUV/NV12 conversions are inherently lossy.

**Risk**: Not a security issue, but documented: RGBA to YUV420p roundtrip
introduces error of approximately 10 levels. Callers must not assume
bit-exact roundtrips.

### 4. GPU Pipeline (Future)

**Source**: wgpu shader inputs from untrusted image data.

**Planned mitigations**:
- Validate buffer dimensions before GPU upload
- Limit shader dispatch sizes
- Validate compute shader outputs

## Non-Threats

- **Network**: No network access. No URLs, sockets, or HTTP.
- **File system**: No file reads or writes. Callers handle I/O.
- **C FFI**: No foreign function calls. Pure Rust with `std::arch` SIMD.
- **Cryptography**: No crypto operations. No secrets handled.

## Fuzz Testing

Fuzz targets cover the primary trust boundary:
- `fuzz_blend` — arbitrary pixel data through all 12 blend modes
- `fuzz_convert` — arbitrary buffers through format conversion
- `fuzz_filter` — arbitrary parameters through all 7 filters
