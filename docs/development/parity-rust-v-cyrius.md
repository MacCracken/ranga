# Surface parity — ranga Rust 1.0.1 vs the Cyrius port

> M7. Audited module by module against `rust-old/`, then each claimed gap
> adversarially re-checked against the files. This is a **surface** audit: it
> asks whether every public Rust item has a counterpart, not whether the
> counterpart computes the same numbers. Behavioural parity is what the 1,700+
> assertions and the per-module mutation testing cover.

## Totals

| | Count |
| --- | ---: |
| Rust public items (fns incl. impl methods, structs, enums, consts, trait impls) | **318** |
| Ported — same name or the expected mechanical rename | **222** |
| Deliberate omission — absent with a documented reason | **22** |
| **Missing** | **51** |
| Cyrius public fns | 370 |

The Cyrius surface is larger than the Rust one because Cyrius has no methods,
no `Drop`, and no `std`: every `X::y()` becomes a free `x_y()`, every owning
type needs an explicit `*_free`, and `bytevec.cyr` exists because `Vec<u8>` does
not.

## Where the 51 gaps are

| Group | Rust items | Missing | What is missing |
| --- | ---: | ---: | --- |
| gpu | 75 | **28** | **the entirety of `pipeline.rs`** |
| icc / spectral / hwaccel | 49 | 9 | derive impls, `IccLutProfile` accessors |
| error / pixel | 61 | 4 | `RangaError` Display/Error, `into_data` |
| transform / filter | 53 | 6 | derive impls, two `ScaleFilter` niceties |
| color / convert | 52 | 2 | `ColorSpace::Bt601`, one `FromStr` |
| composite / histogram / blend | 28 | 2 | two `BlendMode` derive impls |

**Half the parity debt is one file.** `gpu/pipeline.rs` holds the fourteen
public `gpu_*` operations and `GpuChain`, and none of it is ported. What IS
ported is everything underneath: the context, the buffers, the SPIR-V emitter,
eleven kernels and all twenty-one WGSL sources. So the gap is **wiring, not
maths** — each missing operation is a function that uploads, picks a kernel,
dispatches and downloads.

The rest is mostly `#[derive]`. Rust's `Debug`, `Clone`, `Copy`, `PartialEq`,
`Hash`, `Serialize` and `FromStr` on plain enums have no Cyrius analogue and
mostly need none — an `i64` is already copyable and comparable. The ones that
genuinely cost something are the string-facing ones: no `FromStr` means a
consumer cannot parse `"Multiply"` back into a `BlendMode`, and there is no
inverse of `blend_mode_name`.

## Fixed by this audit

The audit's value was not the inventory — it was four things it found wrong in
code that was already passing its tests:

1. **`pixel_view_new` accepted oversized buffers.** It tested `len < want`; Rust
   tests `data.len() != expected` and returns `DimensionMismatch`. The port
   silently widened the contract — a caller passing a whole frame where a tile
   was meant got a view reading the wrong rows instead of a diagnostic. Fixed to
   exact equality, with the error code corrected to match Rust, and both
   directions now asserted.

2. **`RangaError`'s entire Display surface was absent.** Eight `RG_ERR_*` codes
   with no way to render them — while `gpu_error_message` in the GPU module
   already established the idiom. Added `ranga_error_message`, with a test that
   no two codes share a string.

3. **`gpu_context_adapter_name` was a stub reporting itself as ported.** It
   returns the literal `"unknown"` unconditionally; nothing ever assigns the
   field, because mabda exposes no live adapter query. Now documented as a stub,
   given a `gpu_context_has_adapter_name` companion, and PINNED by a test that
   asserts the stub state — so the day mabda can report a name, the suite fails
   and the contract gets updated rather than a consumer branching on "unknown"
   forever.

4. **Two comments asserted things that were not true.** `pixel.cyr` claimed the
   `PixelView`/`PixelViewMut` split "survives as naming plus the mutating
   accessors being defined only for the Mut form" — there are no such accessors,
   and both forms read through a writable pointer. `gpu_spirv.cyr` claimed
   mabda's lowerer handles "exactly five" GLSL.std.450 ops; it handles eight,
   and the three unlisted ones include `FClamp`, which several kernels here
   spell the long way as FMax-then-FMin.

## Divergences that are deliberate, recorded so they are not re-litigated

- **Error payloads are gone.** `BufferTooSmall { need, have }` and friends carry
  data an `i64` cannot. `ranga_error_message` returns the format string with the
  placeholders removed rather than a message implying detail it lacks. Rust's own
  tests assert on the interpolated values and have no analogue here.
- **`PixelView` and `PixelViewMut` share one representation.** Cyrius cannot
  express borrow mutability. The distinction is naming only, and now says so.
- **Panicking variants collapse into checked ones.** `PixelFormat::buffer_size`
  merges into `pixel_format_checked_buffer_size` because there is no panic idiom;
  callers who ignore the negative return are where `.expect()` used to be.
- **`GpuBuffer::download_async` is not ported.** It existed to work around
  `mabda::PendingReadback::finish()` needing a non-`Send` `&Device` — a Rust
  borrow-checker artifact. The Cyrius mabda exposes no async readback to build
  on. The consumer cost is real: no way to overlap readback with CPU work.
- **`data`/`data_mut` and `rows`/`rows_mut` merge.** Cyrius pointers carry no
  mutability, so the pairs are one function each.
- **GPU grayscale is BT.709, CPU grayscale is BT.601.** Inherited from Rust,
  which had the same split. Reconciling either side would change output the Rust
  line produced.

## Remaining behavioural divergences worth an ADR

Found while checking surface parity, not counted as gaps:

- `PixelBuffer::new` maps a size overflow to `BufferTooSmall { need: usize::MAX }`
  in Rust; the port returns `RG_ERR_ALLOCATION`.
- Rust's `PixelView::new` calls the *panicking* `buffer_size`, so `u32::MAX`
  dimensions abort the process; the port returns an error code. Likewise
  `rows()` on a planar format `debug_assert`s in Rust and yields wrong slices in
  release, where `pixel_buffer_row_offset` returns `RG_ERR_INVALID_FORMAT`. Both
  are hardening, and both mean a program that aborted under Rust now continues.
