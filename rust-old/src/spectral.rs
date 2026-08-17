//! Spectral color science bridge via [`prakash`].
//!
//! This module re-exports key spectral types and functions from the prakash
//! crate and provides convenience conversions between prakash's `Xyz` type
//! and ranga's [`CieXyz`]. Use it for spectral power distribution analysis,
//! illuminant calculations, correlated color temperature, and color rendering
//! index computation.

pub use prakash::spectral::{
    CIE_1931_2DEG, Spd, VISIBLE_MAX_NM, VISIBLE_MIN_NM, cct_from_xy, cie_cmf_at,
    color_rendering_index, color_temperature_to_rgb, illuminant_a, illuminant_d50, illuminant_d65,
    illuminant_f2, illuminant_f11, linear_to_srgb_gamma, planck_radiance, srgb_gamma_to_linear,
    wavelength_to_rgb, wien_peak,
};

/// Prakash RGB type, aliased to avoid conflicts with ranga color types.
pub use prakash::spectral::Rgb as PrakashRgb;
/// Prakash XYZ type, aliased to avoid conflicts with [`CieXyz`].
pub use prakash::spectral::Xyz as PrakashXyz;

use crate::color::CieXyz;

// ---------------------------------------------------------------------------
// From conversions: prakash Xyz <-> ranga CieXyz
// ---------------------------------------------------------------------------

impl From<prakash::spectral::Xyz> for CieXyz {
    #[inline]
    fn from(xyz: prakash::spectral::Xyz) -> Self {
        CieXyz {
            x: xyz.x,
            y: xyz.y,
            z: xyz.z,
        }
    }
}

impl From<CieXyz> for prakash::spectral::Xyz {
    #[inline]
    fn from(xyz: CieXyz) -> Self {
        prakash::spectral::Xyz::new(xyz.x, xyz.y, xyz.z)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Convert an SPD to a ranga CieXyz color.
///
/// Integrates the spectral power distribution against the CIE 1931
/// standard observer to produce XYZ tristimulus values.
#[must_use]
#[inline]
pub fn spd_to_xyz(spd: &Spd) -> CieXyz {
    spd.to_xyz().into()
}

/// Compute correlated color temperature from a ranga CieXyz value.
///
/// Uses McCamy's approximation. Valid for ~3000-50000 K.
#[must_use]
#[inline]
pub fn xyz_to_cct(xyz: &CieXyz) -> f64 {
    let pk_xyz = prakash::spectral::Xyz::new(xyz.x, xyz.y, xyz.z);
    let (cx, cy, _) = pk_xyz.to_xyy();
    cct_from_xy(cx, cy)
}

/// Convert a wavelength (nm) to a ranga CieXyz color via CIE 1931 CMFs.
///
/// Returns the XYZ tristimulus values for a monochromatic light source
/// at the given wavelength. Returns zero outside 380-780 nm.
#[must_use]
#[inline]
pub fn wavelength_to_xyz(wavelength_nm: f64) -> CieXyz {
    let (x, y, z) = cie_cmf_at(wavelength_nm);
    CieXyz { x, y, z }
}

/// Get the D65 illuminant white point as a ranga CieXyz.
#[must_use]
#[inline]
pub fn d65_white() -> CieXyz {
    prakash::spectral::Xyz::D65_WHITE.into()
}

/// Get the D50 illuminant white point as a ranga CieXyz.
#[must_use]
#[inline]
pub fn d50_white() -> CieXyz {
    prakash::spectral::Xyz::D50_WHITE.into()
}

/// Generate a blackbody spectral power distribution at the given temperature.
///
/// Wraps [`Spd::blackbody`] for convenience.
#[must_use]
#[inline]
pub fn blackbody_spd(temperature_k: f64) -> Spd {
    Spd::blackbody(temperature_k)
}
