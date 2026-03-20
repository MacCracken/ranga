.PHONY: check fmt clippy test audit deny vet fuzz coverage build doc clean

# Run all CI checks locally
check: fmt clippy test audit

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-targets -- -D warnings

# Run test suite
test:
	cargo test

# Security audit
audit:
	cargo audit

# Supply-chain checks (cargo-deny)
deny:
	cargo deny check

# Supply-chain verification (cargo-vet)
vet:
	cargo vet

# Run fuzz targets (30 seconds each)
fuzz:
	cargo fuzz run fuzz_blend -- -max_total_time=30
	cargo fuzz run fuzz_convert -- -max_total_time=30
	cargo fuzz run fuzz_filter -- -max_total_time=30

# Generate coverage report
coverage:
	cargo llvm-cov --html --output-dir coverage/
	@echo "Coverage report: coverage/html/index.html"

# Build release
build:
	cargo build --release

# Generate documentation
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/
