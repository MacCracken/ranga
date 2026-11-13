# 001 — What mabda's native GPU path accepts (as of mabda 4.2.0)

ranga's GPU layer (`src/gpu_*.cyr`) runs on mabda. ranga emits SPIR-V itself
(`gpu_spirv.cyr`), and mabda lowers it to GFX9 on the native AMD backend. No other
backend compiles it (below, and ADR 005). The rules
below cannot be derived from ranga's code. Most of them fail silently or with a bare
0 when broken. Symbols are named rather than given line numbers, because the
vendored `lib/mabda.cyr` moves with every bump. Re-check this note whenever the
mabda pin moves.

## Backends and entry points

- **The wgpu path does not dispatch anything.** mabda's context-aware wgpu dispatch,
  `_backend_wgpu_compute_dispatch`, returns `GPU_ERR_OTHER` unconditionally, and
  `MABDA_WGPU_COMPUTE = 0`. ranga carries all 21 WGSL sources from the Rust line
  (`gpu_shaders.cyr`), but nothing in `src/` dispatches them. On a wgpu context every
  `gpu_*` operation fails. The only wgpu compute route in mabda is the older
  `compute_dispatch(device, queue, …)`, which takes the device and queue directly.
- **The native NVIDIA backend runs SASS, not SPIR-V, and does not refuse SPIR-V.**
  Its shader-module create takes precompiled SM75 SASS (`SHADER_SRC_SASS`). The
  SPIR-V→SASS compiler is mabda phase N9, unshipped. Handed SPIR-V through
  `gpu_shader_module_create_spirv`, it uploads the words verbatim, and the dispatch
  runs them as machine code. On a GTX 1660 SUPER, nouveau then kills the channel
  (`Xid 13 … Out Of Range Register`). Its dispatch also reads `bp` as one output
  buffer, not the count-prefixed binding array AMD reads. Context creation and the
  buffer calls do work. ranga refuses every compute operation on any backend but
  native AMD before a module is created (`gpu_context_supports_compute`,
  [ADR 005](../adr/005-gpu-compute-runs-only-on-native-amd.md)). Filed in mabda's
  `docs/development/issues/`, all still open at mabda 4.2.0:
  - `2026-11-02-n9-spirv-to-sass-compiler-request.md`;
  - `2026-11-02-compute-dispatch-bindings-differ-amd-nvidia.md`;
  - `2026-11-02-nvidia-shader-module-create-ignores-source-kind.md`.
- **Only the context-taking surface is backend-agnostic:**
  - `gpu_buffer_create`, `_write`, `_read` and `_release`;
  - `gpu_compute_dispatch`.

  `gpu_ctx_device` / `gpu_ctx_queue` and every helper that takes a `device` are
  wgpu-only. On a native context the same struct slot holds a GEM buffer-object handle.
- **Every context constructor returns a tagged Result.** Dereferencing it as a
  context yields plausible garbage rather than a fault. Call `payload()` only after
  `is_err_result()`.
- **The two error spaces overlap.** `RG_GPU_ERR_FORMAT` and
  `RG_ERR_UNSUPPORTED_CONVERSION` are both -4, and `RG_GPU_ERR_BUFFER_OP` and
  `RG_ERR_BUFFER_TOO_SMALL` are both -3. Anything returned in an `RG_ERR_*` slot goes
  through `gpu_error_to_ranga`.
- **mabda exposes neither async readback nor an adapter name.** Readback is
  synchronous only. `gpu_context_adapter_name` is a pinned stub. The pieces for a
  real adapter name exist but are not wired:
  - `wgpu_adapter_get_info` on wgpu;
  - `native_drm_version` and `native_amdgpu_query_dev_info` natively, which give a
    driver name and device id but no marketing name.

## SPIR-V the native lowerer accepts

- **The id cap is 256 (`NATIVE_SHADER_CAP_IDS`), and the ids mabda adds itself come
  out of the same range.** They are allocated from `[id_bound, 256)`, so the usable
  budget for ranga's own ids is well under 256. Bilinear resize, at an id bound of
  201, fails with -25, "MIR: value id outside cap_ids". The fix is a smaller module
  or separable passes. Register pressure is not the cause.
- **Nested selections are rejected** (`GISEL_ERR_NESTED_SELECTION`). Two sequential
  selections are fine. A two-condition guard is written without nesting:
  `_gk_guard_open2`, used by `wipe`.
- **Branch shape:** the body goes in the THEN block, which ends with `OpBranch` to
  the merge, and the merge block holds the function's only `OpReturn`. An early
  return inside an `if` now fails to compile with a named error. Before mabda 4.1.0
  it compiled and silently wrote nothing.
- **`OpSelect` lowers only when its condition is an integer compare in the
  immediately preceding instruction.** Other shapes fail with codes 33/34.
- **GLSL.std.450 ext-inst coverage:**
  - lowered: FAbs, Floor, FMA, FClamp, Sqrt, FMin, FMax and Ldexp (Ldexp is f64-only);
  - not lowered: Log, Sin, Cos and Round.

  A kernel that needs Log/Sin/Cos (Gaussian noise) cannot run natively.
  `floor(x + 0.5)` is the rounding idiom, and it rounds half UP, whereas WGSL
  `round()` rounds half to EVEN.
- **Fixed in mabda 4.1.0 (the filing is archived as resolved in mabda's
  `docs/development/issues/archived/2026-08-19-native-spirv-compile-limits.md`):**
  - loops (`OpLoopMerge`) and `OpPhi`;
  - correctly rounded f32 `OpFDiv`;
  - `ULessThan`, which is mapped.

  ranga's kernels still carry workarounds written before then: host-side
  reciprocals, the `count > idx` operand swap and the underflow-conjunction trick.
  ranga's emitter has no loop or phi helpers yet.
- **f64 is enabled natively** (`MABDA_NATIVE_F64 = 1`). ranga's kernels are f32
  throughout.
- **mabda's parser does not enforce SPIR-V section order.** A module with its
  sections swapped validates, compiles and runs. `tests/gpu_spirv.tcyr` asserts the
  section boundaries itself, because a stricter consumer would reject such a module.
- **Diagnostics:** when a module fails to compile, `gpu_last_shader_error()` and
  `gpu_shader_error_name(code)` give the reason. Read them before bisecting.
