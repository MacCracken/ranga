//! ICC color profile parsing — matrix-based profiles only.
//!
//! Parses ICC v2/v4 profiles to extract the 3x3 color transformation matrix
//! and tone response curves (TRC). Only matrix/TRC-based profiles are supported;
//! LUT-based profiles will return an error.
//!
//! # Examples
//!
//! ```
//! use ranga::icc::{IccProfile, ToneCurve};
//!
//! let tc = ToneCurve::Gamma(2.2);
//! assert!((tc.apply(0.5) - 0.5_f64.powf(2.2)).abs() < 1e-10);
//! ```

use crate::RangaError;
use serde::{Deserialize, Serialize};

/// The `acsp` magic signature that must appear at bytes 36..40 in every ICC profile.
const ACSP_SIGNATURE: &[u8; 4] = b"acsp";

/// Minimum size for a valid ICC profile (128-byte header + 4-byte tag count).
const MIN_PROFILE_SIZE: usize = 132;

/// A parsed ICC color profile (matrix-based only).
///
/// Contains the 3x3 color matrix (rXYZ, gXYZ, bXYZ columns) and the three
/// tone response curves needed to convert linearized RGB to CIE XYZ.
///
/// # Examples
///
/// ```no_run
/// use ranga::icc::IccProfile;
///
/// # fn example(profile_bytes: &[u8]) {
/// let profile = IccProfile::from_bytes(profile_bytes).unwrap();
/// let (x, y, z) = profile.apply(0.5, 0.5, 0.5);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct IccProfile {
    /// Profile version (major, minor).
    pub version: (u8, u8),
    /// Color space of the profile (e.g., `b"RGB "`, `b"CMYK"`).
    pub color_space: [u8; 4],
    /// 3x3 color matrix (column-major: columns are rXYZ, gXYZ, bXYZ).
    pub matrix: [[f64; 3]; 3],
    /// Red tone response curve.
    pub red_trc: ToneCurve,
    /// Green tone response curve.
    pub green_trc: ToneCurve,
    /// Blue tone response curve.
    pub blue_trc: ToneCurve,
}

/// Tone response curve — either a simple gamma value or a lookup table.
///
/// # Examples
///
/// ```
/// use ranga::icc::ToneCurve;
///
/// let gamma = ToneCurve::Gamma(2.2);
/// assert!((gamma.apply(1.0) - 1.0).abs() < 1e-10);
///
/// let table = ToneCurve::Table(vec![0.0, 0.5, 1.0]);
/// assert!((table.apply(0.5) - 0.5).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToneCurve {
    /// Simple gamma curve: `output = input.powf(gamma)`.
    Gamma(f64),
    /// Lookup table mapping `0.0..=1.0` input to `0.0..=1.0` output.
    Table(Vec<f64>),
}

impl ToneCurve {
    /// Apply the tone curve to a single value.
    ///
    /// The input should be in the range `0.0..=1.0`. Values outside that range
    /// are clamped before evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::icc::ToneCurve;
    ///
    /// let tc = ToneCurve::Gamma(2.2);
    /// assert!((tc.apply(0.0) - 0.0).abs() < 1e-10);
    /// assert!((tc.apply(1.0) - 1.0).abs() < 1e-10);
    /// assert!((tc.apply(0.5) - 0.5_f64.powf(2.2)).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn apply(&self, v: f64) -> f64 {
        let v = v.clamp(0.0, 1.0);
        match self {
            ToneCurve::Gamma(gamma) => v.powf(*gamma),
            ToneCurve::Table(table) => {
                if table.is_empty() {
                    return v;
                }
                if table.len() == 1 {
                    return table[0];
                }
                let max_idx = (table.len() - 1) as f64;
                let pos = v * max_idx;
                let lo = pos.floor() as usize;
                let hi = lo.min(table.len() - 2) + 1;
                let lo = lo.min(table.len() - 1);
                let frac = pos - pos.floor();
                table[lo] * (1.0 - frac) + table[hi] * frac
            }
        }
    }
}

impl IccProfile {
    /// Parse an ICC profile from raw bytes.
    ///
    /// Only matrix/TRC-based profiles are supported. Returns an error if the
    /// profile is malformed, truncated, or uses LUT-based tables instead of
    /// matrix + TRC tags.
    ///
    /// # Errors
    ///
    /// Returns [`RangaError::InvalidFormat`] if:
    /// - The data is too short for a valid ICC header
    /// - The `acsp` signature is missing
    /// - Required tags (`rXYZ`, `gXYZ`, `bXYZ`, `rTRC`, `gTRC`, `bTRC`) are missing
    /// - Tag data is malformed or truncated
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::icc::IccProfile;
    ///
    /// # fn example(bytes: &[u8]) {
    /// let profile = IccProfile::from_bytes(bytes).unwrap();
    /// println!("version: {}.{}", profile.version.0, profile.version.1);
    /// # }
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, RangaError> {
        if data.len() < MIN_PROFILE_SIZE {
            return Err(RangaError::InvalidFormat(
                "ICC profile too short".to_string(),
            ));
        }

        // Validate acsp signature at offset 36.
        if &data[36..40] != ACSP_SIGNATURE {
            return Err(RangaError::InvalidFormat(
                "missing acsp signature in ICC profile".to_string(),
            ));
        }

        // Parse version.
        let major = data[8];
        let minor = data[9] >> 4;

        // Parse color space.
        let mut color_space = [0u8; 4];
        color_space.copy_from_slice(&data[16..20]);

        // Read tag table.
        let tag_count = read_u32_be(data, 128) as usize;
        let tag_table_end = 132 + tag_count * 12;
        if data.len() < tag_table_end {
            return Err(RangaError::InvalidFormat(
                "ICC tag table extends beyond profile data".to_string(),
            ));
        }

        let tags = parse_tag_table(data, tag_count)?;

        // Parse XYZ tags for the matrix columns.
        let r_xyz = parse_xyz_tag(data, find_tag(&tags, b"rXYZ")?)?;
        let g_xyz = parse_xyz_tag(data, find_tag(&tags, b"gXYZ")?)?;
        let b_xyz = parse_xyz_tag(data, find_tag(&tags, b"bXYZ")?)?;

        // Matrix is column-major: column 0 = rXYZ, column 1 = gXYZ, column 2 = bXYZ.
        let matrix = [
            [r_xyz[0], r_xyz[1], r_xyz[2]],
            [g_xyz[0], g_xyz[1], g_xyz[2]],
            [b_xyz[0], b_xyz[1], b_xyz[2]],
        ];

        // Parse TRC tags.
        let red_trc = parse_trc_tag(data, find_tag(&tags, b"rTRC")?)?;
        let green_trc = parse_trc_tag(data, find_tag(&tags, b"gTRC")?)?;
        let blue_trc = parse_trc_tag(data, find_tag(&tags, b"bTRC")?)?;

        Ok(IccProfile {
            version: (major, minor),
            color_space,
            matrix,
            red_trc,
            green_trc,
            blue_trc,
        })
    }

    /// Apply the profile to an RGB triplet, producing CIE XYZ values.
    ///
    /// First linearizes the input through the tone response curves, then
    /// multiplies by the 3x3 color matrix.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::icc::IccProfile;
    ///
    /// # fn example(profile: &IccProfile) {
    /// let (x, y, z) = profile.apply(0.5, 0.5, 0.5);
    /// assert!(x >= 0.0);
    /// # }
    /// ```
    #[must_use]
    pub fn apply(&self, r: f64, g: f64, b: f64) -> (f64, f64, f64) {
        let rl = self.red_trc.apply(r);
        let gl = self.green_trc.apply(g);
        let bl = self.blue_trc.apply(b);

        let x = self.matrix[0][0] * rl + self.matrix[1][0] * gl + self.matrix[2][0] * bl;
        let y = self.matrix[0][1] * rl + self.matrix[1][1] * gl + self.matrix[2][1] * bl;
        let z = self.matrix[0][2] * rl + self.matrix[1][2] * gl + self.matrix[2][2] * bl;

        (x, y, z)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A raw tag entry from the ICC tag table.
struct TagEntry {
    signature: [u8; 4],
    offset: usize,
    size: usize,
}

fn read_u32_be(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_u16_be(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([data[offset], data[offset + 1]])
}

fn read_i32_be(data: &[u8], offset: usize) -> i32 {
    i32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_s15fixed16(data: &[u8], offset: usize) -> f64 {
    read_i32_be(data, offset) as f64 / 65536.0
}

fn parse_tag_table(data: &[u8], count: usize) -> Result<Vec<TagEntry>, RangaError> {
    let mut tags = Vec::with_capacity(count);
    for i in 0..count {
        let base = 132 + i * 12;
        if base + 12 > data.len() {
            return Err(RangaError::InvalidFormat(
                "ICC tag table entry out of bounds".to_string(),
            ));
        }
        let mut signature = [0u8; 4];
        signature.copy_from_slice(&data[base..base + 4]);
        let offset = read_u32_be(data, base + 4) as usize;
        let size = read_u32_be(data, base + 8) as usize;
        tags.push(TagEntry {
            signature,
            offset,
            size,
        });
    }
    Ok(tags)
}

fn find_tag<'a>(tags: &'a [TagEntry], sig: &[u8; 4]) -> Result<&'a TagEntry, RangaError> {
    tags.iter().find(|t| &t.signature == sig).ok_or_else(|| {
        RangaError::InvalidFormat(format!(
            "missing required ICC tag: {}",
            String::from_utf8_lossy(sig)
        ))
    })
}

fn parse_xyz_tag(data: &[u8], tag: &TagEntry) -> Result<[f64; 3], RangaError> {
    let off = tag.offset;
    if tag.size < 20 || off + 20 > data.len() {
        return Err(RangaError::InvalidFormat(
            "XYZ tag data too short".to_string(),
        ));
    }
    if &data[off..off + 4] != b"XYZ " {
        return Err(RangaError::InvalidFormat(
            "XYZ tag has wrong type signature".to_string(),
        ));
    }
    Ok([
        read_s15fixed16(data, off + 8),
        read_s15fixed16(data, off + 12),
        read_s15fixed16(data, off + 16),
    ])
}

fn parse_trc_tag(data: &[u8], tag: &TagEntry) -> Result<ToneCurve, RangaError> {
    let off = tag.offset;
    if tag.size < 8 || off + 8 > data.len() {
        return Err(RangaError::InvalidFormat(
            "TRC tag data too short".to_string(),
        ));
    }

    let type_sig = &data[off..off + 4];
    match type_sig {
        b"curv" => parse_curv(data, off, tag.size),
        b"para" => parse_para(data, off, tag.size),
        _ => Err(RangaError::InvalidFormat(format!(
            "unsupported TRC type: {}",
            String::from_utf8_lossy(type_sig)
        ))),
    }
}

fn parse_curv(data: &[u8], off: usize, size: usize) -> Result<ToneCurve, RangaError> {
    if size < 12 || off + 12 > data.len() {
        return Err(RangaError::InvalidFormat(
            "curv tag too short for entry count".to_string(),
        ));
    }
    let count = read_u32_be(data, off + 8) as usize;

    if count == 0 {
        // Linear (gamma 1.0).
        return Ok(ToneCurve::Gamma(1.0));
    }

    if count == 1 {
        // u8Fixed8Number gamma.
        if off + 14 > data.len() {
            return Err(RangaError::InvalidFormat(
                "curv tag too short for gamma value".to_string(),
            ));
        }
        let gamma = read_u16_be(data, off + 12) as f64 / 256.0;
        return Ok(ToneCurve::Gamma(gamma));
    }

    // Table of u16 values.
    let table_bytes = count * 2;
    if off + 12 + table_bytes > data.len() {
        return Err(RangaError::InvalidFormat(
            "curv tag table extends beyond data".to_string(),
        ));
    }
    let mut table = Vec::with_capacity(count);
    for i in 0..count {
        let v = read_u16_be(data, off + 12 + i * 2);
        table.push(v as f64 / 65535.0);
    }
    Ok(ToneCurve::Table(table))
}

fn parse_para(data: &[u8], off: usize, size: usize) -> Result<ToneCurve, RangaError> {
    if size < 12 || off + 12 > data.len() {
        return Err(RangaError::InvalidFormat("para tag too short".to_string()));
    }
    let func_type = read_u16_be(data, off + 8);

    match func_type {
        0 => {
            // Type 0: single gamma parameter.
            if off + 16 > data.len() {
                return Err(RangaError::InvalidFormat(
                    "para type 0 too short for gamma parameter".to_string(),
                ));
            }
            let g = read_s15fixed16(data, off + 12);
            Ok(ToneCurve::Gamma(g))
        }
        3 => {
            // Type 3: 7 parameters — g, a, b, c, d, e, f.
            // Requires 12 + 7*4 = 40 bytes.
            if off + 40 > data.len() {
                return Err(RangaError::InvalidFormat(
                    "para type 3 too short for parameters".to_string(),
                ));
            }
            let g = read_s15fixed16(data, off + 12);
            let a = read_s15fixed16(data, off + 16);
            let b = read_s15fixed16(data, off + 20);
            let c = read_s15fixed16(data, off + 24);
            let d = read_s15fixed16(data, off + 28);
            let e = read_s15fixed16(data, off + 32);
            let f = read_s15fixed16(data, off + 36);

            // Build a 4096-entry lookup table for the parametric curve.
            let n = 4096;
            let mut table = Vec::with_capacity(n);
            for i in 0..n {
                let input = i as f64 / (n - 1) as f64;
                let output = if input >= d {
                    (a * input + b).max(0.0).powf(g) + e
                } else {
                    c * input + f
                };
                table.push(output.clamp(0.0, 1.0));
            }
            Ok(ToneCurve::Table(table))
        }
        _ => Err(RangaError::InvalidFormat(format!(
            "unsupported parametricCurveType function type: {func_type}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_profile() {
        let profile_data = build_test_profile();
        let profile = IccProfile::from_bytes(&profile_data).unwrap();
        assert_eq!(profile.version.0, 2);
        assert_eq!(&profile.color_space, b"RGB ");
        // Check matrix is approximately sRGB.
        assert!(
            (profile.matrix[0][0] - 0.4124564).abs() < 0.001,
            "rXYZ.X mismatch: {}",
            profile.matrix[0][0]
        );
    }

    #[test]
    fn invalid_signature_rejected() {
        let data = vec![0u8; 256];
        // No acsp signature.
        assert!(IccProfile::from_bytes(&data).is_err());
    }

    #[test]
    fn too_short_rejected() {
        let data = vec![0u8; 64];
        assert!(IccProfile::from_bytes(&data).is_err());
    }

    #[test]
    fn tone_curve_gamma() {
        let tc = ToneCurve::Gamma(2.2);
        assert!((tc.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((tc.apply(1.0) - 1.0).abs() < 1e-10);
        assert!((tc.apply(0.5) - 0.5f64.powf(2.2)).abs() < 1e-10);
    }

    #[test]
    fn tone_curve_table() {
        let table: Vec<f64> = (0..256).map(|i| (i as f64 / 255.0).powf(2.2)).collect();
        let tc = ToneCurve::Table(table);
        let v = tc.apply(0.5);
        assert!(v > 0.0 && v < 1.0);
    }

    #[test]
    fn tone_curve_clamps_input() {
        let tc = ToneCurve::Gamma(2.0);
        assert!((tc.apply(-0.5) - 0.0).abs() < 1e-10);
        assert!((tc.apply(1.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tone_curve_table_single_entry() {
        let tc = ToneCurve::Table(vec![0.42]);
        assert!((tc.apply(0.0) - 0.42).abs() < 1e-10);
        assert!((tc.apply(1.0) - 0.42).abs() < 1e-10);
    }

    #[test]
    fn tone_curve_table_empty() {
        let tc = ToneCurve::Table(vec![]);
        // Passthrough when empty.
        assert!((tc.apply(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn apply_white() {
        let profile_data = build_test_profile();
        let profile = IccProfile::from_bytes(&profile_data).unwrap();
        let (x, y, z) = profile.apply(1.0, 1.0, 1.0);
        // Sum of columns should approximate D65 white point.
        assert!(x > 0.9 && x < 1.0, "X={x}");
        assert!(y > 0.9 && y < 1.1, "Y={y}");
        assert!(z > 0.8 && z < 1.2, "Z={z}");
    }

    #[test]
    fn apply_black() {
        let profile_data = build_test_profile();
        let profile = IccProfile::from_bytes(&profile_data).unwrap();
        let (x, y, z) = profile.apply(0.0, 0.0, 0.0);
        assert!((x).abs() < 1e-10);
        assert!((y).abs() < 1e-10);
        assert!((z).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Test profile builder
    // -----------------------------------------------------------------------

    /// Build a minimal ICC v2 profile with sRGB-like matrix and gamma 2.2 TRC.
    fn build_test_profile() -> Vec<u8> {
        // sRGB matrix (D65) XYZ values (s15Fixed16):
        let r_xyz: [f64; 3] = [0.4124564, 0.2126729, 0.0193339];
        let g_xyz: [f64; 3] = [0.3575761, 0.7151522, 0.1191920];
        let b_xyz: [f64; 3] = [0.1804375, 0.0721750, 0.9503041];

        // We need 6 tags: rXYZ, gXYZ, bXYZ, rTRC, gTRC, bTRC.
        let tag_count: u32 = 6;
        let tag_table_size = 4 + tag_count as usize * 12; // 4 for count + 6*12
        let header_size = 128;
        let tag_table_start = header_size;

        // Tag data starts after header + tag table.
        let data_start = tag_table_start + tag_table_size;

        // Each XYZ tag is 20 bytes. Each curv tag (gamma 2.2) is 14 bytes.
        let xyz_tag_size = 20usize;
        let curv_tag_size = 14usize;

        // Layout tag data sequentially, with 4-byte alignment padding.
        fn align4(v: usize) -> usize {
            (v + 3) & !3
        }

        let r_xyz_off = data_start;
        let g_xyz_off = r_xyz_off + align4(xyz_tag_size);
        let b_xyz_off = g_xyz_off + align4(xyz_tag_size);
        let r_trc_off = b_xyz_off + align4(xyz_tag_size);
        let g_trc_off = r_trc_off + align4(curv_tag_size);
        let b_trc_off = g_trc_off + align4(curv_tag_size);
        let total_size = b_trc_off + align4(curv_tag_size);

        let mut buf = vec![0u8; total_size];

        // -- Header --
        // Profile size.
        write_u32_be(&mut buf, 0, total_size as u32);
        // Version: 2.4.
        buf[8] = 2;
        buf[9] = 4 << 4;
        // Color space: "RGB ".
        buf[16..20].copy_from_slice(b"RGB ");
        // acsp signature.
        buf[36..40].copy_from_slice(b"acsp");

        // -- Tag table --
        write_u32_be(&mut buf, 128, tag_count);

        let tags: [(&[u8; 4], usize, usize); 6] = [
            (b"rXYZ", r_xyz_off, xyz_tag_size),
            (b"gXYZ", g_xyz_off, xyz_tag_size),
            (b"bXYZ", b_xyz_off, xyz_tag_size),
            (b"rTRC", r_trc_off, curv_tag_size),
            (b"gTRC", g_trc_off, curv_tag_size),
            (b"bTRC", b_trc_off, curv_tag_size),
        ];

        for (i, (sig, offset, size)) in tags.iter().enumerate() {
            let base = 132 + i * 12;
            buf[base..base + 4].copy_from_slice(*sig);
            write_u32_be(&mut buf, base + 4, *offset as u32);
            write_u32_be(&mut buf, base + 8, *size as u32);
        }

        // -- XYZ tag data --
        fn write_xyz(buf: &mut [u8], off: usize, xyz: [f64; 3]) {
            buf[off..off + 4].copy_from_slice(b"XYZ ");
            // Reserved bytes 4..8 are zero.
            write_s15fixed16(buf, off + 8, xyz[0]);
            write_s15fixed16(buf, off + 12, xyz[1]);
            write_s15fixed16(buf, off + 16, xyz[2]);
        }

        write_xyz(&mut buf, r_xyz_off, r_xyz);
        write_xyz(&mut buf, g_xyz_off, g_xyz);
        write_xyz(&mut buf, b_xyz_off, b_xyz);

        // -- curv tag data (gamma 2.2) --
        fn write_curv_gamma(buf: &mut [u8], off: usize, gamma: f64) {
            buf[off..off + 4].copy_from_slice(b"curv");
            // Reserved bytes 4..8 are zero.
            write_u32_be(buf, off + 8, 1); // count = 1
            let fixed = (gamma * 256.0).round() as u16;
            write_u16_be(buf, off + 12, fixed);
        }

        write_curv_gamma(&mut buf, r_trc_off, 2.2);
        write_curv_gamma(&mut buf, g_trc_off, 2.2);
        write_curv_gamma(&mut buf, b_trc_off, 2.2);

        buf
    }

    fn write_u32_be(buf: &mut [u8], off: usize, val: u32) {
        buf[off..off + 4].copy_from_slice(&val.to_be_bytes());
    }

    fn write_u16_be(buf: &mut [u8], off: usize, val: u16) {
        buf[off..off + 2].copy_from_slice(&val.to_be_bytes());
    }

    fn write_s15fixed16(buf: &mut [u8], off: usize, val: f64) {
        let fixed = (val * 65536.0).round() as i32;
        buf[off..off + 4].copy_from_slice(&fixed.to_be_bytes());
    }
}
