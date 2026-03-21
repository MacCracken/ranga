# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

---

## v0.21.3 — Integration & Performance

### Consumer integration
- [ ] rasa adopts ranga (replace rasa-core color math + rasa-engine filters)
- [ ] tazama adopts ranga (replace manual BT.601 + histogram analysis)
- [ ] aethersafta adopts ranga (replace custom blend + color conversion)

### GPU performance
- [ ] Pipeline caching (avoid recreating pipelines per call)
- [ ] Batched on-GPU dispatch (chain operations without CPU readback)
- [ ] GPU blur kernels (horizontal + vertical WGSL compute shaders)
- [ ] Async GPU readback (non-blocking download)
- [ ] GPU/CPU crossover benchmarks at 4K+

### CPU performance
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
