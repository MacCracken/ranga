# Contributing to ranga

Thank you for your interest in contributing to ranga. This document covers the
development workflow, code standards, and project conventions.

## Development Workflow

1. **Fork** the repository on GitHub.
2. **Create a branch** from `main` for your work.
3. **Make your changes**, ensuring all checks pass.
4. **Open a pull request** against `main`.

## Prerequisites

- Rust toolchain (MSRV: **1.89**)
- cargo-deny, cargo-fuzz, and cargo-tarpaulin (for the full Makefile workflow)

## Makefile Targets

| Target          | Description                                      |
| --------------- | ------------------------------------------------ |
| `make check`    | Run fmt + clippy + test + audit (the full suite) |
| `make fmt`      | Format code with `cargo fmt`                     |
| `make clippy`   | Lint with `cargo clippy`                         |
| `make test`     | Run the test suite                               |
| `make deny`     | Audit dependencies with `cargo deny`             |
| `make vet`      | Run `cargo vet` checks                           |
| `make fuzz`     | Run fuzz targets                                 |
| `make coverage` | Generate code coverage report                    |

Before opening a PR, run `make check` to verify everything passes.

## Adding a New Module

1. Create `src/module.rs` with your implementation.
2. Add `pub mod module;` to `src/lib.rs`.
3. Add unit tests in the module file and integration tests in `tests/` as
   appropriate.

## Code Style

- Run `cargo fmt` before committing. All code must be formatted.
- `cargo clippy -D warnings` must pass with no warnings.
- All public items (functions, structs, enums, traits, type aliases) must have
  doc comments.
- Keep functions focused and testable.

## Testing

- Unit tests go in the module file under `#[cfg(test)]`.
- Integration tests go in the `tests/` directory.
- Property-based and fuzz tests are encouraged for buffer and arithmetic code.

## License

ranga is licensed under **GPL-3.0-only**. All contributions must be compatible
with this license. By submitting a pull request, you agree that your
contribution is licensed under the same terms.
