# Roadmap

The plan lives in [docs/development/roadmap.md](docs/development/roadmap.md). This file
is a summary. ranga continues in Cyrius; the Rust line closed at 1.0.1 and is
preserved at `rust-old/`, as a checklist of what the port is still missing and an oracle
for porting mistakes, until the port is complete.

| Milestone | Theme |
| --- | --- |
| ~~2.2.x~~ | **Done in 2.2.0.** Bit-exact again: `scripts/oracle` against `rust-old/`, with every CPU divergence fixed or ADR'd |
| **2.3.x** | Verification parity, done in 2.3.0–2.3.2: 8 fuzz targets in CI, 31 properties, Rust's integration and edge-case tests, golden vectors from the oracle, the oracle in CI, all 134 benchmarks (GPU on the AMD dev host). Rust's blur and bilateral darkening fixed, not kept (ADR 003, 004). Open: guard NVIDIA contexts, which can't compute until mabda compiles SPIR-V for SM75 (checked on a GTX 1660 SUPER) |
| **2.4.x** | CPU surface completion: the serde decision, clone/eq for heap records, `Lut3d::from_cube`, writable records |
| **2.5.x** | GPU: the wgpu decision, `GpuChain`, native blur/blend/bilinear, rounding and size fixes |
| **2.6.x** | Performance: vectorised blur, histogram ops, contrast/saturation, the YUV inverse |
| **2.7.x** | Docs, examples, security audit, a first Cyrius consumer |
| **2.x** | Close the port: retire `rust-old/` (its outputs are captured in `tests/oracle_golden.tcyr` since 2.3.0) |

The per-item inventory behind these milestones:
[docs/development/parity-rust-v-cyrius.md](docs/development/parity-rust-v-cyrius.md).
