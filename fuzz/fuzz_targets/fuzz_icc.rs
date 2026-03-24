#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::icc::{IccProfile, ToneCurve};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Strategy 1: Feed raw bytes to IccProfile::from_bytes
    if let Ok(profile) = IccProfile::from_bytes(data) {
        // If it parses, call apply with fuzz-driven RGB values
        if data.len() >= 6 {
            let tail = &data[data.len() - 6..];
            let r = (u16::from_le_bytes([tail[0], tail[1]]) as f64) / 65535.0;
            let g = (u16::from_le_bytes([tail[2], tail[3]]) as f64) / 65535.0;
            let b = (u16::from_le_bytes([tail[4], tail[5]]) as f64) / 65535.0;
            let _ = profile.apply(r, g, b);
        }
        // Also try boundary values
        let _ = profile.apply(0.0, 0.0, 0.0);
        let _ = profile.apply(1.0, 1.0, 1.0);
        let _ = profile.apply(0.5, 0.5, 0.5);
    }

    // Strategy 2: Fuzz ToneCurve::Table with fuzz-driven entries
    if data.len() >= 4 {
        // Build a table from pairs of bytes → f64 values in 0.0..1.0
        let table: Vec<f64> = data
            .chunks(2)
            .map(|chunk| {
                let val = if chunk.len() == 2 {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                } else {
                    chunk[0] as u16
                };
                val as f64 / 65535.0
            })
            .collect();

        let curve = ToneCurve::Table(table);

        // Apply with various values — should never panic
        let _ = curve.apply(0.0);
        let _ = curve.apply(0.5);
        let _ = curve.apply(1.0);

        // Fuzz-driven input value
        let input = data[0] as f64 / 255.0;
        let _ = curve.apply(input);

        // Out-of-range values — should be clamped gracefully
        let _ = curve.apply(-1.0);
        let _ = curve.apply(2.0);
    }

    // Strategy 3: Fuzz ToneCurve::Gamma with arbitrary gamma values
    if data.len() >= 8 {
        let gamma = f64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        if gamma.is_finite() {
            let curve = ToneCurve::Gamma(gamma);
            let _ = curve.apply(0.0);
            let _ = curve.apply(0.5);
            let _ = curve.apply(1.0);
        }
    }
});
