# Security Policy

## Scope

ranga is an image processing library focused on pixel processing, color math,
and buffer handling. It is a pure Rust library with no C FFI in its core, no
network access, and no file I/O.

The primary security-relevant surface areas are:

- **Buffer overflows in pixel processing** -- incorrect length calculations or
  indexing when operating on pixel buffers.
- **Malformed input handling** -- untrusted pixel dimensions, formats, or buffer
  sizes passed by downstream consumers.
- **Integer overflow in color math** -- fixed-point arithmetic used in color
  space conversions.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.20.x  | Yes       |
| < 0.20  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in ranga, please report it
responsibly:

1. **Email** [security@agnos.dev](mailto:security@agnos.dev) with a description
   of the issue, steps to reproduce, and any relevant context.
2. **Do not** open a public issue for security vulnerabilities.
3. You will receive an acknowledgment within **48 hours**.
4. We follow a **90-day disclosure timeline**. We will work with you to
   coordinate public disclosure after a fix is available.

## Security Design

- No C FFI in the core library -- eliminates an entire class of memory safety
  issues.
- All pixel buffer operations validate dimensions and expected data lengths
  before processing.
- Fixed-point BT.601 arithmetic uses explicit clamping to prevent overflow.
- Fuzz testing (`make fuzz`) targets buffer construction and color conversion
  paths.
