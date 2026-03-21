# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## Backlog (medium priority)

### Geometry & Transforms
- [ ] Bicubic resize (higher quality than bilinear)
- [ ] Perspective / projective transform (4-corner mapping)

### Compositing
- [ ] Gradient radial fill
- [ ] Gradient along arbitrary angle

### Filters
- [ ] Median filter (noise reduction, non-linear)
- [ ] Bilateral filter (edge-preserving smoothing)
- [ ] Vibrance (smart saturation that protects skin tones)
- [ ] Channel mixer (arbitrary RGB matrix)
- [ ] Threshold (binary, Otsu)
- [ ] Flood fill with tolerance (replaces rasa flood_fill)

### Color Science
- [ ] Oklab/Oklch color space (modern perceptually uniform alternative to Lab)
- [ ] Wider-gamut BT.2020 support (HDR video)
- [ ] ICC profile: LUT-based profiles (not just matrix)
- [ ] Embedded sRGB v2 ICC profile bytes (like rasa)
- [ ] Auto white balance (gray-world or similar)
- [ ] Histogram equalization / auto-levels

### Performance
- [ ] GPU batched dispatch — chain filter+blend on GPU without CPU readback
- [ ] Rayon parallelism for in-place filters (brightness, contrast, saturation, invert)
- [ ] Cache-aware tiling for blur (process in L2-friendly 64x64 tiles)
- [ ] SIMD color conversion (vectorize Y-plane with SSSE3 shuffle)
- [ ] SIMD filter paths (brightness, grayscale via SSE2/NEON)
- [ ] GPU crop / resize / transform shaders

### API & Usability
- [ ] `Display` impl for `PixelFormat`, `BlendMode`, `ColorSpace`
- [ ] Consistent error messages (standardize GPU format validation strings)
- [ ] `PixelBuffer::from_view()` — convert a `PixelView` back to owned
- [ ] `PixelBuffer::rows()` / `rows_mut()` — row iterator for ergonomic processing
- [ ] Typed pixel accessors (`get_rgba(x, y) -> [u8; 4]`, `set_rgba(x, y, pixel)`)

### GPU Compute
- [ ] GPU all 12 blend modes in compositor pipeline (batch layers)
- [ ] GPU 3D LUT shader (trilinear interpolation on GPU)
- [ ] GPU hue shift / color balance shaders
- [ ] GPU noise generation (compute shader RNG)
- [ ] GPU dissolve / fade / wipe transition shaders
- [ ] Automatic CPU/GPU selection based on image size + ai-hwaccel detection

### Testing & Quality
- [ ] Photoshop reference test suite (compare blend output against golden values)
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

- [ ] All color spaces stable (sRGB, linear, P3, BT.601, BT.709)
- [ ] All blend modes match Photoshop reference output
- [ ] SIMD on x86_64 (SSE2+AVX2) and aarch64 (NEON)
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
