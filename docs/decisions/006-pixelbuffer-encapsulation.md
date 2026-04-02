# ADR 006: Encapsulate PixelBuffer Fields

**Status**: Accepted
**Date**: 2026-03-28

## Context

`PixelBuffer` had public fields: `data: Vec<u8>`, `width: u32`, `height: u32`,
`format: PixelFormat`. External consumers could:

1. Set `data` to an arbitrary length that does not match `width * height * bpp`
2. Change `width` or `height` without resizing `data`
3. Change `format` without re-encoding the pixel data

All of these break the internal invariant:
`data.len() == format.buffer_size(width, height)`.

Violating this invariant causes out-of-bounds access in filter, blend, and
conversion functions that compute byte offsets from dimensions and format.

## Decision

Fields changed from `pub` to `pub(crate)`. Public accessor methods added:

- `data() -> &[u8]`
- `data_mut() -> &mut [u8]`
- `into_data() -> Vec<u8>`
- `width() -> u32`
- `height() -> u32`
- `format() -> PixelFormat`

Construction still goes through `PixelBuffer::new()`, `PixelBuffer::zeroed()`,
and `PixelBuffer::from_raw()`, all of which validate the invariant. Internal
crate code (`pub(crate)`) continues to access fields directly.

## Consequences

### Positive

- External consumers cannot break the `data.len() == buffer_size(w, h)` invariant
- `data_mut()` allows in-place pixel manipulation without exposing dimensions
- Internal crate code (filters, blend, convert, GPU) is unchanged — direct field
  access within the crate has zero overhead

### Negative

- Breaking API change for external consumers: `buf.data` becomes `buf.data()`,
  `buf.width` becomes `buf.width()`, etc.
- Downstream crates (rasa, tazama, aethersafta) required a one-time migration

### Mitigation

- The migration is mechanical (field access to method call) and caught at compile
  time — no silent breakage
- `data_mut()` preserves the ability to write pixels directly for performance
