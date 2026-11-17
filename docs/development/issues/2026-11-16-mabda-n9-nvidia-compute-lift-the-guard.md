# mabda 4.4.0 compiles ranga's SPIR-V on NVIDIA (N9): lift ADR 005's guard, and the test-side changes

**Status:** 🔴 **OPEN**, the follow-up to ranga's own request, which mabda has resolved:
`mabda/docs/development/issues/archived/2026-11-02-n9-spirv-to-sass-compiler-request.md`.
**Filed:** 2026-11-16, from mabda's N9.6 (the done-when of its SPIR-V → SASS compiler). mabda's
CHANGELOG `[4.4.0]` has sections N9.1–N9.6, and its roadmap's v4.4.0 "The consumer check" lists this
issue's items.
**Updated:** 2026-11-16, later the same day. Item 5 was added: the bilinear resize now compiles
on both backends (see "Found since filing").
**Placement:**
- `src/gpu_context.cyr`: `gpu_context_supports_compute` at line 164, and its comment at 146–162;
- `src/gpu_pipeline.cyr`: the guard comments at 30–34 and 99–106;
- `tests/gpu.tcyr`, `tests/gpu_kernels.tcyr`, `tests/gpu_pipeline.tcyr`, `tests/gpu_spirv.tcyr` and
  `tests/gpu_nvidia.tcyr`;
- `docs/adr/005-gpu-compute-runs-only-on-native-amd.md`;
- `cyrius.cyml` `[deps.mabda]` (line 176–179, `tag = "4.2.0"`);
- low priority: `src/gpu_spirv.cyr` `_SV_VERSION` (line 34) and `spv_preamble`'s interface list
  (line 393 onward).

**Severity:** none for ranga's library as pinned today. Item 1 is a **test build break** as soon as
ranga's mabda pin moves to 4.4.0, on every backend. The rest unblocks a feature: GPU compute on NVIDIA.
**Affects:** ranga 2.4.0 against mabda 4.4.0. That release is **not tagged yet**: the work is
uncommitted in mabda's tree. Everything below was measured against that tree's `dist/mabda.cyr`.

## Summary

mabda N9 compiles SPIR-V to SM75 SASS in-tree, so `gpu_shader_module_create_spirv` on a native NVIDIA
context now does what it does on AMD. mabda checked ranga's kernels and suites against that tree
without editing ranga's repository. It used a scratch clone of **ranga@de54123** plus the four
changes listed under "What this asks of ranga". Results:

| Suite (built static, run on `chew`: GTX 1660 SUPER, TU116, nouveau) | native AMD (Cezanne) | NVIDIA (TU116) |
|---|---|---|
| `tests/gpu.tcyr` | 48 / 48 | **48 / 48** |
| `tests/gpu_kernels.tcyr` | 53 / 53 | **53 / 53** |
| `tests/gpu_pipeline.tcyr` | 102 / 102 | **102 / 102** |
| `tests/gpu_spirv.tcyr` | 54 / 54 | **54 / 54** |
| `tests/gpu_nvidia.tcyr` | SKIPPED | 23 / 54. By design: it pins the refusals (item 3) |

The kernel log on chew stayed clean: no Xid, no channel kill.

The logs are on the mabda dev box, not in a repository (`mabda/build/` is gitignored):
- `build/logs/ranga-gpu-suites-amd-n9.6-20261116-142307.log`;
- `build/logs/ranga-gpu-suites-nvidia-n9.6-20261116-142509.log`, the passing NVIDIA run;
- `build/logs/ranga-gpu-suites-nvidia-n9.6-20261116-142345.log`, the run before mabda's fix described
  next.

**The first NVIDIA run found a mabda bug, now fixed there.** mabda's NVIDIA compiler counted only the
bindings a kernel *reads*. Its AMD compiler counts the bindings a kernel *declares*. ranga's one-buffer
kernels declare three (pixels, then the params block's u32 and f32 views), and invert and grayscale never
read the f32 view. Their NVIDIA modules therefore wanted 2 bindings, and ranga's 3-entry binding array
was refused (`gpu_compute_dispatch` → `GPU_ERR_OTHER` → `RG_GPU_ERR_BUFFER_OP`). The flips' 4-entry
arrays were refused the same way. mabda now counts declared bindings on both backends, so ranga's
binding arrays need no change.

mabda also now **vendors ranga's 11 kernels**: `mabda/tests/spirv/ranga/*.spv` holds the bytes from
de54123. Its gates `test-native-ranga-kernels-e2e` and `test-nvidia-ranga-kernels-e2e` run them with
ranga's binding layout and params on 4087 seeded pixels. The outputs match goldens that mabda's fixture
generator computes with its own SPIR-V interpreter, rounding every f32 operation, on both the Cezanne and
the TU116. When ranga changes a kernel, mabda can re-dump them with
`scripts/gen-spirv-ranga-fixtures.py --from-ranga <ranga checkout>`.

## What this asks of ranga

Items 1–4 go together, **with the `[deps.mabda]` pin moving to the 4.4.0 tag** once mabda cuts it
(item 5 came later and rides with them; see "Found since filing").
Order matters:
- On mabda **4.2.0**, the pinned version, NVIDIA runs SPIR-V as machine code and nouveau kills the
  channel. **Do not lift the guard on 4.2.0.**
- On 4.3.x, NVIDIA refuses SPIR-V at create (`SHMOD_ERR_SRC_KIND`). That is safe: `_gp_kernel` reports
  UNSUPPORTED.
- Only 4.4.0 compiles it.

1. **`tests/gpu_spirv.tcyr` lines 189–195: the table builders' 4-argument form.** mabda 4.4.0's
   scratch-buffer contract (its CHANGELOG `[4.4.0]`, "Breaking") dropped the `cap` / `id_bound` arguments.
   The old 5-argument calls are a compile error against 4.4.0 (*expects 4 arguments, got 5*) on every
   backend:
   ```
   -assert_eq(spirv_build_type_table(p, n, idb, idb, types) >= 0, 1,
   +assert_eq(spirv_build_type_table(p, n, types, idb * SPIRV_TYPE_REC) >= 0, 1,
   -assert_eq(spirv_build_const_table(p, n, idb, idb, consts) >= 0, 1,
   +assert_eq(spirv_build_const_table(p, n, consts, idb * SPIRV_CONST_REC) >= 0, 1,
   -assert_eq(spirv_build_decoration_table(p, n, idb, idb, deco) >= 0, 1,
   +assert_eq(spirv_build_decoration_table(p, n, deco, idb * SPIRV_DECO_REC) >= 0, 1,
   ```
   The third argument is the buffer, the fourth its length in bytes. With this change the suite passes
   54 / 54 on both backends.

2. **`gpu_context_supports_compute`: add `BACKEND_KIND_NVIDIA` to the allowlist** (ADR 005 says a
   backend joins "when mabda compiles ranga's kernels on it and a hardware suite passes there"; both now
   hold). The one-line change measured above:
   ```
        if (backend_kind(gpu_ctx_backend(load64(c))) == BACKEND_KIND_AMD) { return 1; }
   +    if (backend_kind(gpu_ctx_backend(load64(c))) == BACKEND_KIND_NVIDIA) { return 1; }
   ```
   Update the comments that describe N9 as unshipped: `src/gpu_context.cyr` 146–162 and
   `src/gpu_pipeline.cyr` 30–34. Also amend ADR 005, whose "Consequences / Neutral" bullet already
   describes this step. That bullet's caveat about the binding array was settled in mabda 4.3.0
   (`mabda/docs/development/issues/archived/2026-11-02-compute-dispatch-bindings-differ-amd-nvidia.md`).
   ranga's `_gp_run_1buf` / `_gp_run_2buf` arrays and `_gp_bind` work unchanged on NVIDIA:
   `NV_BUF_HANDLE` / `NV_BUF_VA` sit at the same offsets (+0 / +8) as AMD's
   `NATIVE_BUF_FIELD_HANDLE` / `_VA`.

3. **`tests/gpu_nvidia.tcyr`: flip the refused rows**, as its own header says to ("When mabda ships N9
   and ranga lifts the guard, this suite is what fails first, by design"). With the guard lifted it
   fails 31 rows, because all eleven operations now succeed:
   - "the NVIDIA backend does not support ranga's compute";
   - each operation's "is refused" / "with RG_GPU_ERR_UNSUPPORTED through err_out" rows, plus "and the
     buffer is untouched" for the in-place ones;
   - the shader-cache check "and it is EMPTY: no SPIR-V reached the NVIDIA backend", which got 22: two
     cache words for each of the 11 modules now built.

   They become the AMD suites' expected values.

4. **A way to run the four GPU suites on an NVIDIA context.** They hard-code `gpu_context_new_amd(err)`
   (`tests/gpu.tcyr:85`, `tests/gpu_kernels.tcyr:123`, `tests/gpu_pipeline.tcyr:127` and `:378`,
   `tests/gpu_spirv.tcyr:206`). On chew, which has no AMD GPU, they SKIP. mabda's run swapped in
   `gpu_context_new_nvidia(err)`, and in `tests/gpu.tcyr:96` the expected backend name became
   `"native-nvidia"`. Whether to try both, pick by an environment variable, or keep separate NVIDIA
   copies is ranga's choice.

## Found since filing: the bilinear resize compiles on both backends

Two later pieces of mabda 4.4.0 work moved the MIR id space from 256 ids to 1024:
- its fourth N9 follow-on (`OpFDiv` and f64 on NVIDIA) gave SM75 the wider space;
- GFX9 then took the same space, so there is one ceiling, `MIR_MAX_CAP_IDS` = 1024.

The SPIR-V id bound stays at 256. `gk_resize_bilinear` uses 201 SPIR-V ids, and its lowering's
temporaries took it past the old 256. It now compiles on both backends; on GFX9 it uses 16 VGPRs.

**The cause was the id space, not register pressure.** The refusal was `MIR_ERR_ID_OOR`, which
ranga's `src/gpu_pipeline.cyr:21` records as "reason -25". mabda's register allocator header said
otherwise and is corrected. So is ranga's own note:
- `src/gpu_kernels.cyr:854–861`, "⚠ THE FOUR TEXELS ARE HELD PACKED … It is liveness, not size",
  says mabda's GFX9 register allocator "gives up";
- what actually happened is that the module ran out of MIR ids. Packing the texels is harmless and
  can stay.

mabda now vendors the kernel as its twelfth ranga fixture: `mabda/tests/spirv/ranga/resize_bilinear.spv`,
the bytes from de54123.
- The gate runs it on 4087 seeded pixels, resizing 61 × 67 to 97 × 53.
- Its params are `[count, src_w, src_h, dst_w, dst_h, f32(1/max(dst_w-1,1)), f32(1/max(dst_h-1,1))]`,
  the layout the emitter's header gives.
- The output hashes to the interpreter's golden `0x784d2538b784a2e0` on the Cezanne and on the TU116
  (`test-native-ranga-kernels-e2e` / `test-nvidia-ranga-kernels-e2e`, 12 of 12).

5. **The bilinear row and `gpu_resize`.** This also needs mabda 4.4.0, like the others.
   - **`tests/gpu_kernels.tcyr:507–508`.** The pinned "but mabda cannot yet lower it to GFX9 —
     flip this when it can" assertion now fails on both backends: `gpu_kernels` passes 52 / 53.
     The module is created (non-zero), so flip the row. It would be worth dispatching the module
     and checking pixels there too.
   - **`gpu_resize` (`src/gpu_pipeline.cyr:468–476`)** could dispatch `gk_resize_bilinear` for
     `SF_BILINEAR` instead of returning `RG_GPU_ERR_UNSUPPORTED`. That needs a kernel id beside
     `GP_K_RESIZE_N`, and the two host-side reciprocals packed into params 5–6. The comments at
     `src/gpu_pipeline.cyr:14–22` and `:468–471` describe the old cap.
   - **`SF_BICUBIC`**, which Rust ran through the bilinear shader, is ranga's call.

   The logs are on the mabda dev box:
   - NVIDIA: `build/logs/nvidia-ranga-fdiv-20261116-162252.log`;
   - AMD: `build/logs/ranga-gpu-suites-amd-gfx9ids-20261116-173352.log`.

   The other three suites pass unchanged on both: `gpu` 48, `gpu_pipeline` 102, `gpu_spirv` 54.
   On AMD the scratch clone opened `gpu_context_new_amd`, with item 1 applied.

## Also, low priority: the SPIR-V version and the entry-point interface

ranga's modules say SPIR-V **1.3** (`_SV_VERSION = 0x00010300`) but list the storage-buffer variables
in `OpEntryPoint`'s interface. Before 1.4 the interface may hold only Input / Output variables, so
`spirv-val --target-env vulkan1.1` rejects every kernel ("In SPIR-V 1.3 or earlier, OpEntryPoint
interfaces must be OpVariables with Storage Class of Input(1) or Output(3)"). mabda accepts them, but a
validator, naga or a Vulkan driver would not. That matters if ranga ever feeds them to wgpu.

Either fix validates. mabda tested both on invert's module: both compile for GFX9 and SM75 with the same
binding count, but they were not run on hardware.
- **Keep 1.3 and list only the GlobalInvocationId input** in `spv_preamble`'s interface. It validates
  with `spirv-val --target-env vulkan1.1`.
- **Or declare 1.4** (`0x00010400`) and keep the list as it is. It validates with
  `--target-env vulkan1.1spv1.4`.

The `_SV_VERSION` comment ("what mabda's parser expects") is also not accurate. mabda reads the header
but never checks the version word (`spirv_hdr_version` in `mabda/src/spirv_parse.cyr` has no caller).

## Not affected

- ranga's kernels themselves. All 11 compile for both backends unchanged, with LocalSize 256 and
  binding counts of 3 (one-buffer kernels) or 4 (two-buffer and geometry kernels). The bilinear
  resize, the twelfth, compiles unchanged too (item 5), with 4 bindings.
- `gpu_buffer_upload` / `gpu_buffer_download`, which never had a guard.
- The wgpu backend, which stays outside the allowlist; that decision is ranga's 2.5.x item.
