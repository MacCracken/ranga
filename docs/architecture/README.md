# Architecture notes

Non-obvious constraints, quirks, and invariants that a reader cannot derive from the code alone. Numbered chronologically — never renumber.

Not decisions (those live in [`../adr/`](../adr/)) and not guides (those live in [`../guides/`](../guides/)). An item here describes *how the world is*, not *what we chose* or *how to do something*.

## Items

- [001 — What mabda's native GPU path accepts](001-mabda-native-gpu-constraints.md): backends, entry points, error spaces, and the SPIR-V the native lowerer takes (as of mabda 4.2.0).
