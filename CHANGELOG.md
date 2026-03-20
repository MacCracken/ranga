# Changelog

## [0.20.3] — 2026-03-20

First release. Core image processing primitives extracted from rasa and aethersafta.

### Added

- **Color module** — `LinRgba`, `Srgba`, `Hsl` types with sRGB↔linear gamma conversion, HSL conversion
- **Pixel buffer** — `PixelBuffer` type with format validation, 6 pixel formats (RGBA8, ARGB8, RGB8, YUV420p, NV12, RgbaF32)
- **Blend modes** — 12 Porter-Duff blend modes (Normal, Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, SoftLight, HardLight, Difference, Exclusion)
- **Color conversion** — RGBA↔YUV420p (BT.601 fixed-point), ARGB→NV12
- **Filters** — brightness, contrast, saturation, levels, curves, grayscale, invert (all in-place on RGBA8)
- **Histogram** — luminance histogram, per-channel RGB histograms, chi-squared distance
- 40+ unit tests covering all modules
