# Roadmap

The plan lives in [docs/development/roadmap.md](docs/development/roadmap.md). This file
is a summary. ranga continues in Cyrius; the Rust line closed at 1.0.1 and is
preserved at `rust-old/`, as a checklist of what the port is still missing and an oracle
for porting mistakes, until the port is complete.

| Milestone | Theme |
| --- | --- |
| ~~2.2.x~~ | **Done in 2.2.0.** Bit-exact again: `scripts/oracle` against `rust-old/`, with every CPU divergence fixed or ADR'd |
| ~~2.3.x~~ | **Done in 2.3.0–2.3.3.** Verification parity: 8 fuzz targets in CI, 31 properties, Rust's integration and edge-case tests, golden vectors from the oracle, the oracle in CI, all 134 benchmarks (GPU on the AMD dev host). Rust's blur and bilateral darkening fixed, not kept (ADR 003, 004). NVIDIA contexts guarded until mabda compiles SPIR-V for SM75, and tested on a GTX 1660 SUPER (ADR 005) |
| ~~2.4.x~~ | **Done in 2.4.0.** CPU surface completion: serde as JSON in serde_json's form (ADR 008), `Lut3d::from_cube` with the `.cube` domain honoured (ADR 007), clone/eq and writable fields on every record, ICC constructors, `into_data`, the Rust spectral names, the `rg_bv_` prefix (ADR 006) |
| **2.5.x** | GPU: the wgpu decision, `GpuChain`, native blur/blend/bilinear, rounding and size fixes |
| **2.6.x** | Performance: vectorised blur, histogram ops, contrast/saturation, the YUV inverse |
| **2.7.x** | Docs, examples, security audit, a first Cyrius consumer |
| **2.x** | Close the port: retire `rust-old/` (its outputs are captured in `tests/oracle_golden.tcyr` since 2.3.0) |

The per-item inventory behind these milestones:
[docs/development/parity-rust-v-cyrius.md](docs/development/parity-rust-v-cyrius.md).
