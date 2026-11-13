# 005 — GPU compute runs only on the native AMD backend

**Status**: Accepted
**Date**: 2026-11-13

## Context

Every compute operation in `gpu_pipeline.cyr` emits a SPIR-V kernel, hands it to
mabda's `gpu_shader_module_create_spirv`, uploads its buffers and dispatches. Rust
1.0.1 had no equivalent decision to make: wgpu compiled WGSL on every adapter it
found.

mabda 4.2.0 has three backends behind one API, and they treat that SPIR-V
differently:

| Backend | What its shader-module create does with SPIR-V | Dispatch |
|---|---|---|
| native AMD | compiles it to GFX9 | runs |
| native NVIDIA | **uploads the words verbatim as SM75 SASS** | runs the words as machine code |
| wgpu | needs a launcher built with SPIR-V support | context-level slot returns `GPU_ERR_OTHER` unconditionally |

There is a reserved fourth kind, `BACKEND_KIND_INTEL`, with no backend behind it.

The NVIDIA row was measured in 2.3.2 on `chew`, a GeForce GTX 1660 SUPER (TU116, SM75)
on nouveau and NVK:
- the context opens, and buffers upload and download;
- every compute operation is executed as garbage machine code;
- nouveau kills the channel (`Xid 13 … Out Of Range Register`).

The driver recovered each time, but a library that faults the user's GPU on its
first `gpu_invert` is worse than one that refuses the call. mabda's SPIR-V→SASS
compiler is its unshipped phase N9, and its NVIDIA create ignoring the source kind
is filed upstream as `2026-11-02-nvidia-shader-module-create-ignores-source-kind.md`.
Neither has a release date.

## Decision

ranga runs compute only on a backend known to compile its SPIR-V, which in mabda
4.2.0 is the native AMD backend alone.

- **`gpu_context_supports_compute(ctx)`** is public. It returns 1 when the context's
  backend kind is `BACKEND_KIND_AMD`, and 0 for every other kind and for a null
  context.
- **`_gp_kernel` checks it first**, before it emits SPIR-V or calls
  `gpu_shader_module_create_spirv`. Otherwise it reports `RG_GPU_ERR_UNSUPPORTED`,
  the same code the four operations without a native kernel already return. Every
  operation reaches the GPU through `_gp_kernel`, and it runs before any buffer is
  uploaded, so no SPIR-V and no pixel data reaches a backend that cannot run them.
- **It is an allowlist.** A backend is added when mabda compiles ranga's kernels on
  it and a hardware suite passes there, not removed when it is found to fault.

The guard covers compute only. The buffer calls (`gpu_buffer_upload`,
`gpu_buffer_download`) work on NVIDIA and are not guarded.

## Consequences

- **Positive**
  - ranga cannot fault an NVIDIA card. `tests/gpu_nvidia.tcyr` opens an NVIDIA
    context, checks that the buffer round trip works and that all eleven native
    operations are refused, and asserts that the context's shader cache is still
    empty afterwards, so no module was ever created. It passes 53 of 53 on `chew`, and
    the kernel log stays clean.
  - A consumer can ask before choosing the GPU, instead of learning it from the
    first operation's error.
- **Negative**
  - NVIDIA gets no GPU compute from ranga until mabda ships N9 and ranga lifts
    the guard. It got none before either; it got a channel fault instead.
  - A Rust caller on an NVIDIA adapter got working compute through wgpu. A ranga
    caller gets UNSUPPORTED and must take the CPU path.
- **Neutral**
  - On wgpu the result was already a failure, but a later one: `BUFFER_OP` from the
    dispatch, after the upload. It is now `UNSUPPORTED`, before anything is uploaded.
    Whether wgpu dispatches WGSL at all is the open 2.5.x decision, and lifting the
    guard for wgpu belongs to that decision.
  - A degenerate `gpu_crop` (empty rectangle) still succeeds on every backend. It
    needs no GPU work and never reaches the guard.
  - When N9 lands, lifting the guard means adding `BACKEND_KIND_NVIDIA` to the
    predicate and flipping `tests/gpu_nvidia.tcyr`'s refused rows to the AMD
    suites' expected values. mabda's NVIDIA dispatch also reads one output buffer
    where AMD reads a binding array
    (`2026-11-02-compute-dispatch-bindings-differ-amd-nvidia.md`). That must be
    settled first, or the kernels will bind the wrong buffers.

## Alternatives considered

- **Deny NVIDIA only.** Rejected. It protects the one backend measured to fault and
  passes SPIR-V to any backend mabda adds next, including the reserved Intel kind,
  until someone measures that one too. Failing closed costs nothing on the backends
  that work.
- **Wait for mabda to refuse SPIR-V on NVIDIA.** That fix is filed upstream, and it
  would turn the fault into a 0 from the create, which `_gp_kernel` already reports
  as UNSUPPORTED. But ranga would still depend on another library's unreleased fix to
  avoid faulting hardware, and it would still upload kernels to a backend that
  cannot run them.
- **Detect a failed create and fall back to the CPU inside `gpu_*`.** Rejected. Rust's
  `gpu_*` functions never silently ran on the CPU, and a caller that asked for the GPU
  should learn that it did not get it. The CPU operations exist under their own names.
