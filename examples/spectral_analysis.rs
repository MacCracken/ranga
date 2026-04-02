//! Spectral color science — SPD analysis, illuminants, CRI.
//!
//! Run with: `cargo run --example spectral_analysis --features spectral`

use ranga::spectral;

fn main() {
    // ── White points ────────────────────────────────────────────────────
    // D65 represents average daylight (~6504K), D50 represents horizon
    // daylight (~5003K). These are reference illuminants for color science.
    let d65_wp = spectral::d65_white();
    let d50_wp = spectral::d50_white();
    println!(
        "D65 white point: X={:.4} Y={:.4} Z={:.4}",
        d65_wp.x, d65_wp.y, d65_wp.z
    );
    println!(
        "D50 white point: X={:.4} Y={:.4} Z={:.4}",
        d50_wp.x, d50_wp.y, d50_wp.z
    );

    // ── Blackbody SPD at 5000K ──────────────────────────────────────────
    // A blackbody radiator at 5000K produces a warm daylight-like spectrum.
    // We convert it to XYZ and then recover the correlated color temperature.
    let bb_spd = spectral::blackbody_spd(5000.0);
    let bb_xyz = spectral::spd_to_xyz(&bb_spd);
    println!(
        "\nBlackbody 5000K -> XYZ: X={:.4} Y={:.4} Z={:.4}",
        bb_xyz.x, bb_xyz.y, bb_xyz.z
    );

    // ── Correlated Color Temperature from XYZ ──────────────────────────
    // McCamy's approximation recovers ~5000K from the blackbody XYZ.
    let cct = spectral::xyz_to_cct(&bb_xyz);
    println!("CCT recovered from blackbody XYZ: {cct:.0} K (expected ~5000 K)");

    // ── CIE 1931 2-degree CMF data at selected wavelengths ─────────────
    // The color matching functions define how the human eye responds to
    // monochromatic light at each wavelength.
    println!("\nCIE 1931 2-degree CMFs at selected wavelengths:");
    for nm in [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0] {
        let (x, y, z) = spectral::cie_cmf_at(nm);
        println!("  {nm:>5.0} nm: x={x:.4} y={y:.4} z={z:.4}");
    }

    // ── Color Rendering Index for D65 illuminant ────────────────────────
    // CRI measures how well a light source renders colors compared to a
    // reference illuminant. D65 (daylight) should score very high (~100).
    let d65_spd = spectral::illuminant_d65();
    let cri = spectral::color_rendering_index(&d65_spd);
    println!("\nCRI for D65 illuminant: {cri:.1} (daylight, expected ~100)");

    // Compare with fluorescent F2 (expected lower CRI, ~60-70).
    let f2_spd = spectral::illuminant_f2();
    let cri_f2 = spectral::color_rendering_index(&f2_spd);
    println!("CRI for F2 illuminant:  {cri_f2:.1} (fluorescent, expected ~60-70)");

    // ── Color temperature to RGB ────────────────────────────────────────
    // Approximate the visual appearance of light at different temperatures.
    println!("\nColor temperature -> sRGB:");
    for kelvin in [2700.0, 4000.0, 5500.0, 6500.0, 10000.0] {
        let rgb = spectral::color_temperature_to_rgb(kelvin);
        let [r, g, b] = rgb.to_u8();
        println!("  {kelvin:>7.0} K: R={r:>3} G={g:>3} B={b:>3}");
    }
}
