# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## v0.20.3 — Foundation + Color Science & SIMD (current)

### Color science
- [ ] BT.709 conversion (HD video standard)
- [ ] Display P3 gamut support
- [ ] ICC profile parsing and application
- [ ] CMYK ↔ RGB conversion
- [ ] Color temperature (Kelvin to RGB)
- [ ] Delta-E color distance (CIE76, CIE94, CIEDE2000)

### SIMD acceleration
- [ ] SSE2 alpha blending (2 pixels/iter, from aethersafta)
- [ ] AVX2 alpha blending (4 pixels/iter)
- [ ] NEON alpha blending (aarch64)
- [ ] SIMD color conversion (RGBA↔YUV)
- [ ] Benchmarks: SIMD vs scalar per operation

### Additional conversions
- [ ] NV12 → RGBA
- [ ] RGB8 ↔ RGBA8 (strip/add alpha)
- [ ] ARGB8 ↔ RGBA8 (channel reorder)

---

## v0.21.3 — Advanced Filters & GPU

### Filters
- [ ] Gaussian blur (separable kernel, configurable radius)
- [ ] Unsharp mask (sharpen)
- [ ] Box blur (fast approximation)
- [ ] Hue shift
- [ ] Color balance (shadows/midtones/highlights)
- [ ] 3D LUT application (from .cube files)
- [ ] Vignette
- [ ] Noise (Gaussian, salt-and-pepper)

### GPU compute (requires `gpu` feature)
- [ ] wgpu device/queue management
- [ ] GPU blend (WGSL compute shader)
- [ ] GPU color conversion
- [ ] GPU filter pipeline
- [ ] Async GPU readback

---

## v0.22.3 — Integration & Performance

### Consumer integration
- [ ] rasa adopts ranga (replace rasa-core color math + rasa-engine filters)
- [ ] tazama adopts ranga (replace manual BT.601 + histogram analysis)
- [ ] aethersafta adopts ranga (replace custom blend + color conversion)

### Performance
- [ ] Zero-copy buffer views (borrow instead of clone for read-only ops)
- [ ] Buffer pool (reuse allocations across frames)
- [ ] Parallel filter application (rayon)
- [ ] Cache-aware tiling for large images

### Quality
- [ ] 85%+ code coverage
- [ ] Benchmark regression tracking in CI

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
