# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## Backlog (medium priority)

### Geometry & Transforms
- [x] Bicubic resize (higher quality than bilinear) — v0.21.3
- [x] Perspective / projective transform (4-corner mapping) — v0.21.3

### Compositing
- [x] Gradient radial fill — v0.21.3
- [x] Gradient along arbitrary angle — v0.21.3

### Filters
- [x] Median filter (noise reduction, non-linear) — v0.20.4
- [x] Bilateral filter (edge-preserving smoothing) — v0.20.4
- [x] Vibrance (smart saturation that protects skin tones) — v0.20.4
- [x] Channel mixer (arbitrary RGB matrix) — v0.20.4
- [x] Threshold (binary, Otsu) — v0.20.4
- [x] Flood fill with tolerance (replaces rasa flood_fill) — v0.20.4

### Color Science
- [x] Oklab/Oklch color space (modern perceptually uniform alternative to Lab) — v0.21.3
- [x] Wider-gamut BT.2020 support (HDR video) — v0.21.3
- [x] ICC profile: LUT-based profiles (not just matrix) — v0.21.3
- [x] Embedded sRGB v2 ICC profile bytes (like rasa) — v0.21.3
- [x] Auto white balance (gray-world or similar) — v0.21.3
- [x] Histogram equalization / auto-levels — v0.21.3

### Performance
- [ ] GPU batched dispatch — chain filter+blend on GPU without CPU readback
- [x] Rayon parallelism for in-place filters (brightness, contrast, saturation, invert) — v0.20.4
- [x] Cache-aware tiling for blur (process in L2-friendly 64px tiles) — v0.21.3
- [x] SIMD color conversion (vectorize Y-plane with SSE2/NEON) — v0.21.3
- [x] SIMD filter paths (brightness, grayscale via SSE2/NEON) — v0.21.3
- [ ] GPU crop / resize / transform shaders

### API & Usability
- [x] `Display` impl for `PixelFormat`, `BlendMode`, `ColorSpace` — v0.20.4
- [ ] Consistent error messages (standardize GPU format validation strings)
- [x] `PixelBuffer::from_view()` — convert a `PixelView` back to owned — v0.20.4
- [x] `PixelBuffer::rows()` / `rows_mut()` — row iterator for ergonomic processing — v0.20.4
- [x] Typed pixel accessors (`get_rgba(x, y) -> [u8; 4]`, `set_rgba(x, y, pixel)`) — v0.20.4

### GPU Compute
- [x] GPU all 12 blend modes in compositor pipeline (batch layers) — v0.20.3
- [x] GPU 3D LUT shader (trilinear interpolation on GPU) — v0.21.3
- [x] GPU hue shift / color balance shaders — v0.21.3
- [ ] GPU noise generation (compute shader RNG)
- [ ] GPU dissolve / fade / wipe transition shaders
- [x] Automatic CPU/GPU selection based on image size + ai-hwaccel detection — v0.20.3

### Testing & Quality
- [x] Photoshop reference test suite (compare blend output against golden values) — v0.21.3
- [ ] Visual regression tests (render → compare against golden images)
- [ ] Extended fuzz campaigns (add blur, LUT, ICC targets)
- [ ] 90%+ code coverage
- [ ] Benchmark regression tracking in CI (compare against baseline)

### Consumer Integration
- [ ] rasa adopts ranga (replace rasa-core color + blend + filters + transform)
- [ ] tazama adopts ranga (replace GPU shaders + BT.601/709 + histogram)
- [ ] aethersafta adopts ranga (replace custom blend + color conversion)
- [ ] secureyeoman: evaluate ranga for screenshot region redaction/crop

---

## v1.0.0 Criteria

- [ ] All color spaces stable (sRGB, linear, P3, BT.601, BT.709, BT.2020)
- [x] All blend modes match Photoshop reference output — v0.21.3
- [x] SIMD on x86_64 (SSE2+AVX2) and aarch64 (NEON) — v0.21.3
- [ ] GPU compute path functional and benchmarked
- [ ] At least 3 downstream consumers (rasa, tazama, aethersafta)
- [ ] 90%+ test coverage
- [ ] docs.rs documentation complete
- [ ] No `unsafe` without `// SAFETY:` comments
- [ ] Benchmarks establish golden numbers

---

## Non-goals

- **Image I/O** — ranga does not load/save files. That's the consumer's job (image crate, tarang, etc.)
- **Video decode/encode** — that's tarang's domain
- **Scene graph / compositing engine** — that's aethersafta
- **AI/ML** — that's hoosh/tarang-ai
- **Text rendering** — that's the consumer's responsibility (cosmic-text, etc.)
