//! Tests for the spectral module (prakash integration).

#![cfg(feature = "spectral")]

use ranga::color::CieXyz;
use ranga::spectral::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// ---------------------------------------------------------------------------
// White point constants
// ---------------------------------------------------------------------------

#[test]
fn white_point_constants_match_functions() {
    let d65_const = CieXyz::D65_WHITE;
    let d65_fn = d65_white();
    assert!(
        approx_eq(d65_const.x, d65_fn.x, 1e-4),
        "D65 x mismatch: const={}, fn={}",
        d65_const.x,
        d65_fn.x,
    );
    assert!(
        approx_eq(d65_const.y, d65_fn.y, 1e-4),
        "D65 y mismatch: const={}, fn={}",
        d65_const.y,
        d65_fn.y,
    );
    assert!(
        approx_eq(d65_const.z, d65_fn.z, 1e-4),
        "D65 z mismatch: const={}, fn={}",
        d65_const.z,
        d65_fn.z,
    );

    let d50_const = CieXyz::D50_WHITE;
    let d50_fn = d50_white();
    assert!(
        approx_eq(d50_const.x, d50_fn.x, 1e-4),
        "D50 x mismatch: const={}, fn={}",
        d50_const.x,
        d50_fn.x,
    );
    assert!(
        approx_eq(d50_const.y, d50_fn.y, 1e-4),
        "D50 y mismatch: const={}, fn={}",
        d50_const.y,
        d50_fn.y,
    );
    assert!(
        approx_eq(d50_const.z, d50_fn.z, 1e-4),
        "D50 z mismatch: const={}, fn={}",
        d50_const.z,
        d50_fn.z,
    );
}

#[test]
fn d65_white_point_values() {
    let d65 = d65_white();
    assert!(
        approx_eq(d65.x, 0.95047, 1e-3),
        "D65 x expected ~0.95047, got {}",
        d65.x,
    );
    assert!(
        approx_eq(d65.y, 1.0, 1e-3),
        "D65 y expected ~1.0, got {}",
        d65.y,
    );
    assert!(
        approx_eq(d65.z, 1.08883, 1e-3),
        "D65 z expected ~1.08883, got {}",
        d65.z,
    );
}

// ---------------------------------------------------------------------------
// XYZ conversion roundtrip
// ---------------------------------------------------------------------------

#[test]
fn xyz_conversion_roundtrip() {
    let original = CieXyz {
        x: 0.4124,
        y: 0.2126,
        z: 0.0193,
    };
    let prakash: PrakashXyz = original.into();
    let back: CieXyz = prakash.into();
    assert!(
        approx_eq(original.x, back.x, 1e-10),
        "x roundtrip failed: {} != {}",
        original.x,
        back.x,
    );
    assert!(
        approx_eq(original.y, back.y, 1e-10),
        "y roundtrip failed: {} != {}",
        original.y,
        back.y,
    );
    assert!(
        approx_eq(original.z, back.z, 1e-10),
        "z roundtrip failed: {} != {}",
        original.z,
        back.z,
    );
}

// ---------------------------------------------------------------------------
// SPD to XYZ
// ---------------------------------------------------------------------------

#[test]
fn spd_to_xyz_d65_illuminant() {
    let d65_spd = illuminant_d65();
    let xyz = spd_to_xyz(&d65_spd);

    // Y should be positive (it represents luminance)
    assert!(xyz.y > 0.0, "D65 SPD should have positive Y, got {}", xyz.y);

    // Check chromaticity is approximately the D65 white point
    let sum = xyz.x + xyz.y + xyz.z;
    let cx = xyz.x / sum;
    let cy = xyz.y / sum;
    // D65 chromaticity: x≈0.3127, y≈0.3290
    assert!(
        approx_eq(cx, 0.3127, 0.005),
        "D65 chromaticity x expected ~0.3127, got {}",
        cx,
    );
    assert!(
        approx_eq(cy, 0.3290, 0.005),
        "D65 chromaticity y expected ~0.3290, got {}",
        cy,
    );
}

// ---------------------------------------------------------------------------
// Wavelength to XYZ
// ---------------------------------------------------------------------------

#[test]
fn wavelength_to_xyz_peak_sensitivity() {
    let xyz_555 = wavelength_to_xyz(555.0);
    // 555nm is peak luminous efficiency — Y should be highest
    assert!(
        xyz_555.y > 0.9,
        "555nm should have high Y (peak sensitivity), got {}",
        xyz_555.y,
    );
}

#[test]
fn wavelength_to_xyz_outside_visible() {
    let xyz_low = wavelength_to_xyz(300.0);
    assert!(
        approx_eq(xyz_low.x, 0.0, 1e-6)
            && approx_eq(xyz_low.y, 0.0, 1e-6)
            && approx_eq(xyz_low.z, 0.0, 1e-6),
        "300nm should return ~zero XYZ, got ({}, {}, {})",
        xyz_low.x,
        xyz_low.y,
        xyz_low.z,
    );

    let xyz_high = wavelength_to_xyz(900.0);
    assert!(
        approx_eq(xyz_high.x, 0.0, 1e-6)
            && approx_eq(xyz_high.y, 0.0, 1e-6)
            && approx_eq(xyz_high.z, 0.0, 1e-6),
        "900nm should return ~zero XYZ, got ({}, {}, {})",
        xyz_high.x,
        xyz_high.y,
        xyz_high.z,
    );
}

// ---------------------------------------------------------------------------
// CCT from XYZ
// ---------------------------------------------------------------------------

#[test]
fn cct_from_d65_white_point() {
    let d65 = d65_white();
    let cct = xyz_to_cct(&d65);
    assert!(
        approx_eq(cct, 6504.0, 100.0),
        "D65 white CCT expected ~6504K, got {}K",
        cct,
    );
}

#[test]
fn cct_from_warm_blackbody() {
    let spd = blackbody_spd(3000.0);
    let xyz = spd_to_xyz(&spd);
    let cct = xyz_to_cct(&xyz);
    assert!(
        approx_eq(cct, 3000.0, 100.0),
        "3000K blackbody CCT expected ~3000K, got {}K",
        cct,
    );
}

// ---------------------------------------------------------------------------
// Blackbody SPD
// ---------------------------------------------------------------------------

#[test]
fn blackbody_spd_properties() {
    let spd = blackbody_spd(5500.0);
    assert!(
        !spd.values.is_empty(),
        "Blackbody SPD should have non-empty values",
    );
    assert!(
        approx_eq(spd.start_nm, 380.0, 1e-6),
        "Blackbody SPD should start at 380nm, got {}",
        spd.start_nm,
    );
    assert!(
        approx_eq(spd.step_nm, 5.0, 1e-6),
        "Blackbody SPD should have 5nm step, got {}",
        spd.step_nm,
    );
}

// ---------------------------------------------------------------------------
// CIE CMF data
// ---------------------------------------------------------------------------

#[test]
fn cie_cmf_data_structure() {
    // 380nm to 780nm at 5nm steps = 81 entries
    assert_eq!(
        CIE_1931_2DEG.len(),
        81,
        "CIE 1931 2-degree observer should have 81 entries, got {}",
        CIE_1931_2DEG.len(),
    );
}

#[test]
fn cie_cmf_peak_y_bar() {
    // Index 35 corresponds to 555nm (380 + 35*5 = 555)
    // This should have the highest y_bar (luminous efficiency)
    let (_, y_bar_555, _) = CIE_1931_2DEG[35];
    for (i, &(_, y_bar, _)) in CIE_1931_2DEG.iter().enumerate() {
        assert!(
            y_bar <= y_bar_555 + 1e-10,
            "y_bar at index {} ({}) exceeds y_bar at 555nm (index 35): {} > {}",
            i,
            380.0 + i as f64 * 5.0,
            y_bar,
            y_bar_555,
        );
    }
}

// ---------------------------------------------------------------------------
// Illuminant SPDs
// ---------------------------------------------------------------------------

#[test]
fn illuminant_spds_valid() {
    let illuminants: Vec<(&str, Spd)> = vec![
        ("D65", illuminant_d65()),
        ("D50", illuminant_d50()),
        ("A", illuminant_a()),
        ("F2", illuminant_f2()),
        ("F11", illuminant_f11()),
    ];

    for (name, spd) in &illuminants {
        assert_eq!(
            spd.values.len(),
            81,
            "{} illuminant should have 81 values, got {}",
            name,
            spd.values.len(),
        );
        assert!(
            approx_eq(spd.start_nm, 380.0, 1e-6),
            "{} illuminant should start at 380nm, got {}",
            name,
            spd.start_nm,
        );
    }
}

// ---------------------------------------------------------------------------
// Color temperature to RGB
// ---------------------------------------------------------------------------

#[test]
fn color_temperature_to_rgb_neutral() {
    let rgb = color_temperature_to_rgb(6500.0);
    // At 6500K (D65), RGB should be near-neutral
    assert!(
        approx_eq(rgb.r, rgb.g, 0.1) && approx_eq(rgb.g, rgb.b, 0.1),
        "6500K should be near-neutral RGB, got ({}, {}, {})",
        rgb.r,
        rgb.g,
        rgb.b,
    );
}

#[test]
fn color_temperature_to_rgb_warm() {
    let rgb = color_temperature_to_rgb(2000.0);
    assert!(
        rgb.r > rgb.b,
        "2000K should be warm (r > b), got r={}, b={}",
        rgb.r,
        rgb.b,
    );
}

// ---------------------------------------------------------------------------
// sRGB gamma roundtrip
// ---------------------------------------------------------------------------

#[test]
fn srgb_gamma_roundtrip() {
    let test_values = [0.0, 0.01, 0.04045, 0.1, 0.5, 0.73, 0.99, 1.0];
    for &x in &test_values {
        let roundtrip = linear_to_srgb_gamma(srgb_gamma_to_linear(x));
        assert!(
            approx_eq(roundtrip, x, 1e-6),
            "sRGB gamma roundtrip failed for {}: got {}",
            x,
            roundtrip,
        );
    }
}

// ---------------------------------------------------------------------------
// Wien peak
// ---------------------------------------------------------------------------

#[test]
fn wien_peak_sun_temperature() {
    // Sun surface temperature ~5778K, peak should be ~501nm
    // wien_peak returns meters, so convert to nm for comparison
    let peak_m = wien_peak(5778.0);
    let peak_nm = peak_m * 1e9;
    assert!(
        approx_eq(peak_nm, 501.0, 10.0),
        "Wien peak for 5778K expected ~501nm, got {}nm",
        peak_nm,
    );
}

// ---------------------------------------------------------------------------
// Color Rendering Index
// ---------------------------------------------------------------------------

#[test]
fn cri_d65_illuminant() {
    let d65_spd = illuminant_d65();
    let cri = color_rendering_index(&d65_spd);
    assert!(
        cri > 95.0,
        "D65 illuminant CRI expected near 100, got {}",
        cri,
    );
}
