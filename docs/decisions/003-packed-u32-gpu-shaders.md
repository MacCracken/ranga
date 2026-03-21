# ADR 003: Packed u32 GPU Shaders

**Status**: Accepted
**Date**: 2026-03-21

## Context

rasa's GPU pipeline uploads pixel data as `Vec<f32>` (4 floats per pixel),
requiring CPU-side u8→f32 conversion before upload and f32→u8 after download.
For a 1080p image this means ~32 MB of data transfer (vs ~8 MB for u8).

## Decision

ranga's WGSL compute shaders operate on packed `u32` RGBA values. Each u32
contains 4 bytes [R, G, B, A]. Shaders include `unpack_rgba`/`pack_rgba`
helper functions that convert to/from `vec4<f32>` for computation.

## Consequences

### Positive

- 4x less data transferred between CPU and GPU (u8 vs f32)
- No CPU-side format conversion before upload — `PixelBuffer.data` goes direct
- Simpler `GpuBuffer::upload/download` — just copy bytes

### Negative

- Shaders are slightly more complex (pack/unpack overhead per pixel)
- Less numeric precision during computation (but 8-bit output doesn't benefit from f32 intermediates anyway)
