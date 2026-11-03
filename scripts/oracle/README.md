# Differential oracle: Rust 1.0.1 vs the Cyrius port

The port's bar is "matches what Rust did". This tool measures that directly. It runs
the frozen Rust crate at `rust-old/` and the Cyrius port on **identical inputs** and
diffs every output value.

```sh
./scripts/oracle/run.sh                     # all groups
./scripts/oracle/run.sh levels affine       # only these groups
./scripts/oracle/run.sh hsl --detail srgba_to_hsl.h --limit 20
```

`run.sh` does the following:
1. Builds `rust/` (a small driver that depends on `rust-old/` by path; it never builds
   inside it and never modifies it).
2. Builds `probe.cyr` against `src/`. The group code lives in `probe_core.cyr`, which
   `probe.cyr` wraps in a `main`; the golden test includes the same file.
3. Runs both, writing `build/oracle/{rust,cyrius}.out`.
4. Runs `compare.py`.

It needs `cargo` (rust-old builds in seconds, offline once its dependencies are
cached) and the pinned `cyrius`. Build products go to `build/oracle/`, which is
gitignored.

## Output and report

Both sides print `group<TAB>key<TAB>type<TAB>value`. The type is one of:
- `u8`: bytes, as hex;
- `f32` / `f64`: bit patterns;
- `code`: the error code, using the Cyrius `RG_ERR_*` numbering;
- `x`: a transition point.

For each group, `compare.py` reports:
- how many values it compared and how many differ;
- the worst distance: ULPs for floats and transition points, counts for bytes;
- the largest absolute float difference.

A sign flip around zero shows as a huge ULP count with a tiny absolute difference.
Add `--detail GROUP` to list the differing keys.

The script exits 1 if any group differs without an accepted reason. Accepted
divergences are listed in `ACCEPTED` in `compare.py`, each with the ADR that decided
it. An accepted group is still reported, but does not fail the run.

## Transition groups: exact over every f32 in [0, 1]

Byte outputs of one float input, such as `linear_to_srgb`, `rgbaf32_to_rgba8`, or the
grey and alpha paths, can differ from Rust only on a few hundred inputs out of
~1.07 × 10⁹. Each one sits next to a rounding boundary, so sampled sweeps miss them
(the pre-2.0.0 audit called these paths byte-exact).

These functions are monotone in their input. So for each output level k = 1…255,
both sides binary-search the smallest f32 at which the output reaches k. The 255
boundaries determine the output on **every** f32 in [0, 1], at a cost of about 7,700
evaluations per function. Two implementations agree everywhere exactly when their
boundaries are identical. A boundary 1 ULP apart means exactly one input value rounds
differently.

## Golden vectors: the evidence that outlives `rust-old/`

`tests/oracle_golden.tcyr` runs every group in **golden mode**: instead of printing,
`probe_core.cyr` hashes each group's lines (FNV-1a 64 over the exact text, newline
included) and counts them. `golden_rust.cyr` holds Rust 1.0.1's count and digest for
every exact group, so `cyrius test` checks the port against Rust's output with no
`cargo` and no `rust-old/`. The groups accepted within a bound (three by ADR 002, and
`bilateral` by ADR 004) are not exact by design; they are listed as skipped and stay
with the live oracle.

A digest says THAT a group moved, not where. While `rust-old/` exists, `run.sh` with
`--detail` says where.

Regenerate the digests only after a deliberate, reviewed change, or after adding a
group:

```sh
./scripts/oracle/run.sh
python3 -I scripts/oracle/make_golden.py build/oracle/rust.out > scripts/oracle/golden_rust.cyr
cyrius test tests/oracle_golden.tcyr
```

`make_golden.py` digests `rust.out`, never `cyrius.out`: the golden values are
Rust's, so a regenerated file cannot bless a port regression.

## Adding a group

Add the group to both `rust/src/main.rs` and `probe_core.cyr` (and wire it into
`run_groups`), keeping:
- the same group name;
- the same keys;
- the same generator calls in the same order. Both sides use the same xorshift64.

Pass floats as exact bit patterns wherever Rust writes a literal, or derive them with
the same f32 operation on both sides (`_o_fdiv` / `_o_fmul` in the probe). Otherwise
a literal or a widened intermediate can make the two disagree for reasons that have
nothing to do with the code under test.

Then regenerate `golden_rust.cyr` as above, so the new group is captured.

`rust-old/` is due to be deleted once the port is complete (roadmap 2.x). CI runs
`run.sh` until then, and `tests/oracle_golden.tcyr` after.
