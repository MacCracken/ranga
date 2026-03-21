# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## Backlog (medium priority)

### Performance
- [ ] GPU batched dispatch — chain filter+blend on GPU without CPU readback
- [ ] GPU crop / resize / transform shaders

### API & Usability
- [ ] Review ai-hwaccel 0.21.3 for new functionality (updated from 0.20 in v0.21.4)
- [ ] Consistent error messages (standardize GPU format validation strings)

### GPU Compute
- [ ] GPU noise generation (compute shader RNG)
- [ ] GPU dissolve / fade / wipe transition shaders

### Testing & Quality
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
