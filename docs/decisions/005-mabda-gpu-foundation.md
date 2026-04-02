# ADR 005: Use mabda as GPU Foundation Layer

**Status**: Accepted
**Date**: 2026-03-28

## Context

ranga had a custom GPU context wrapping wgpu directly: a `RefCell<PipelineCache>`
for interior mutability, raw pointers for buffer mapping, and a custom `block_on`
spin-loop for async device creation. This worked but carried safety hazards —
`RefCell` panics on overlapping borrows, raw pointers bypass borrow checking, and
the spin-loop was not portable.

Multiple AGNOS crates (ranga, rasa, tazama, aethersafta) each depended on wgpu
independently, causing duplicate dependency trees, inconsistent GPU context
management, and divergent error handling across the suite.

## Decision

Depend on **mabda** (AGNOS GPU foundation crate) instead of raw wgpu for device
context, pipeline cache, shader cache, and buffer management. wgpu remains as a
transitive dependency through mabda for type re-exports (e.g. `wgpu::TextureFormat`).

ranga's `GpuContext` becomes a thin wrapper around `mabda::GpuContext`, adding
ranga-specific pipeline and shader caches on top.

## Consequences

### Positive

- Eliminated `RefCell` and raw pointer hazards — mabda's context is `Send + Sync`
- Shared pipeline and shader caching across all AGNOS crates
- Consistent error handling (mabda error types map cleanly to `RangaError`)
- All AGNOS crates share one wgpu version through mabda, avoiding duplicate trees
- Device creation uses mabda's async runtime integration instead of a custom spin-loop

### Negative

- `gpu_*` functions now require `&mut GpuContext` instead of `&GpuContext`, since
  mabda enforces exclusive access for pipeline cache mutation
- ranga's GPU layer is coupled to mabda's release cadence
- Contributors must understand the mabda abstraction layer in addition to wgpu

### Mitigation

- The `&mut` requirement is a net positive — it prevents concurrent GPU submissions
  that could cause undefined behavior
- mabda is an AGNOS crate under the same release process, so version coordination
  is straightforward
