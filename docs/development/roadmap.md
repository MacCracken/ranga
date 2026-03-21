# Ranga Development Roadmap

> Ranga (रंग, Sanskrit: color/hue) — Core image processing library for the AGNOS creative suite.

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
