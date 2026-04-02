//! Blend modes — Porter-Duff and Photoshop-style compositing.
//!
//! 12 blend modes with optional SIMD acceleration.
//!
//! # Examples
//!
//! ```
//! use ranga::blend::{BlendMode, blend_pixel};
//!
//! let result = blend_pixel([255, 0, 0, 255], [0, 0, 255, 255], BlendMode::Normal, 255);
//! assert!(result[0] > 200); // mostly red
//! ```

use serde::{Deserialize, Serialize};

/// Fast, accurate division by 255: `(val + 128 + ((val + 128) >> 8)) >> 8`.
#[inline(always)]
fn div255(val: u16) -> u16 {
    let tmp = val + 128;
    (tmp + (tmp >> 8)) >> 8
}

/// Supported blend modes.
///
/// Implements standard Photoshop-style compositing modes. All modes operate
/// on sRGB byte values (0–255) with straight (non-premultiplied) alpha.
///
/// # Examples
///
/// ```
/// use ranga::blend::BlendMode;
///
/// let mode = BlendMode::Multiply;
/// assert_ne!(mode, BlendMode::Screen);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
}

impl std::fmt::Display for BlendMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Multiply => write!(f, "Multiply"),
            Self::Screen => write!(f, "Screen"),
            Self::Overlay => write!(f, "Overlay"),
            Self::Darken => write!(f, "Darken"),
            Self::Lighten => write!(f, "Lighten"),
            Self::ColorDodge => write!(f, "Color Dodge"),
            Self::ColorBurn => write!(f, "Color Burn"),
            Self::SoftLight => write!(f, "Soft Light"),
            Self::HardLight => write!(f, "Hard Light"),
            Self::Difference => write!(f, "Difference"),
            Self::Exclusion => write!(f, "Exclusion"),
        }
    }
}

/// Blend a source pixel over a destination pixel using the given mode and opacity.
///
/// All values are 0–255 (sRGB byte space). Alpha is premultiplied in the blend.
/// Returns the composited pixel as `[R, G, B, A]`.
///
/// # Examples
///
/// ```
/// use ranga::blend::{BlendMode, blend_pixel};
///
/// // Opaque red over opaque blue → mostly red
/// let result = blend_pixel([255, 0, 0, 255], [0, 0, 255, 255], BlendMode::Normal, 255);
/// assert!(result[0] > 200);
///
/// // Transparent source leaves destination unchanged
/// let result = blend_pixel([255, 0, 0, 0], [0, 0, 255, 255], BlendMode::Normal, 255);
/// assert_eq!(result, [0, 0, 255, 255]);
///
/// // Multiply darkens
/// let result = blend_pixel([128, 128, 128, 255], [200, 200, 200, 255], BlendMode::Multiply, 255);
/// assert!(result[0] < 200);
/// ```
#[inline]
#[must_use]
pub fn blend_pixel(src: [u8; 4], dst: [u8; 4], mode: BlendMode, opacity: u8) -> [u8; 4] {
    let sa = div255(src[3] as u16 * opacity as u16) as u8;
    if sa == 0 {
        return dst;
    }

    let blend_channel = |s: u8, d: u8| -> u8 {
        let s = s as f32 / 255.0;
        let d = d as f32 / 255.0;
        let result = match mode {
            BlendMode::Normal => s,
            BlendMode::Multiply => s * d,
            BlendMode::Screen => 1.0 - (1.0 - s) * (1.0 - d),
            BlendMode::Overlay => {
                if d < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
            BlendMode::Darken => s.min(d),
            BlendMode::Lighten => s.max(d),
            BlendMode::ColorDodge => {
                if s >= 1.0 {
                    1.0
                } else {
                    (d / (1.0 - s)).min(1.0)
                }
            }
            BlendMode::ColorBurn => {
                if s <= 0.0 {
                    0.0
                } else {
                    1.0 - ((1.0 - d) / s).min(1.0)
                }
            }
            BlendMode::SoftLight => {
                if s <= 0.5 {
                    d - (1.0 - 2.0 * s) * d * (1.0 - d)
                } else {
                    let g = if d <= 0.25 {
                        ((16.0 * d - 12.0) * d + 4.0) * d
                    } else {
                        d.sqrt()
                    };
                    d + (2.0 * s - 1.0) * (g - d)
                }
            }
            BlendMode::HardLight => {
                if s < 0.5 {
                    2.0 * s * d
                } else {
                    1.0 - 2.0 * (1.0 - s) * (1.0 - d)
                }
            }
            BlendMode::Difference => (s - d).abs(),
            BlendMode::Exclusion => s + d - 2.0 * s * d,
        };
        (result * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };

    // Porter-Duff source-over with blended color
    let br = blend_channel(src[0], dst[0]);
    let bg = blend_channel(src[1], dst[1]);
    let bb = blend_channel(src[2], dst[2]);

    let sa16 = sa as u16;
    let inv_sa = 255u16 - sa16;

    [
        div255(br as u16 * sa16 + dst[0] as u16 * inv_sa) as u8,
        div255(bg as u16 * sa16 + dst[1] as u16 * inv_sa) as u8,
        div255(bb as u16 * sa16 + dst[2] as u16 * inv_sa) as u8,
        (div255(sa16 * 255 + dst[3] as u16 * inv_sa)).min(255) as u8,
    ]
}

/// Scalar fallback for Normal-mode row blending.
fn blend_row_normal_scalar(src: &[u8], dst: &mut [u8], opacity: u8) {
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let sa = div255(s[3] as u16 * opacity as u16);
        if sa == 0 {
            continue;
        }
        let inv_sa = 255u16 - sa;
        d[0] = div255(s[0] as u16 * sa + d[0] as u16 * inv_sa) as u8;
        d[1] = div255(s[1] as u16 * sa + d[1] as u16 * inv_sa) as u8;
        d[2] = div255(s[2] as u16 * sa + d[2] as u16 * inv_sa) as u8;
        d[3] = div255(sa * 255 + d[3] as u16 * inv_sa).min(255) as u8;
    }
}

// ---------------------------------------------------------------------------
// SSE2 implementation (x86_64)
// ---------------------------------------------------------------------------
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn blend_row_normal_sse2(src: &[u8], dst: &mut [u8], opacity: u8) {
    use std::arch::x86_64::*;

    let pixel_count = src.len() / 4;
    let simd_pixels = pixel_count / 2 * 2; // round down to multiple of 2
    let byte_count = simd_pixels * 4;

    // SAFETY: All intrinsics below require SSE2, which is enabled via
    // #[target_feature(enable = "sse2")] on this function.
    unsafe {
        let zero = _mm_setzero_si128();
        let opacity_val = opacity as u16;
        let ones_255 = _mm_set1_epi16(255);

        let mut i = 0usize;
        while i < byte_count {
            // SAFETY: We process 2 pixels (8 bytes) per iteration. `i + 8 <= byte_count`
            // is guaranteed because `byte_count` is aligned to 8-byte (2-pixel) boundaries
            // and i advances by 8 each iteration.
            let s_bytes = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
            let d_bytes = _mm_loadl_epi64(dst.as_ptr().add(i) as *const __m128i);

            // Unpack u8 -> u16 (lower 8 bytes)
            let s16 = _mm_unpacklo_epi8(s_bytes, zero);
            let d16 = _mm_unpacklo_epi8(d_bytes, zero);

            // Broadcast alpha for each pixel:
            // Layout is [R0 G0 B0 A0 R1 G1 B1 A1] as u16
            // Mask 0xFF = 11_11_11_11 broadcasts element 3 across low 4 u16s
            let alpha_lo = _mm_shufflelo_epi16(s16, 0xFF);
            let alpha = _mm_shufflehi_epi16(alpha_lo, 0xFF);

            // sa = div255(alpha * opacity)
            let opacity_vec = _mm_set1_epi16(opacity_val as i16);
            let round_128 = _mm_set1_epi16(128);
            let raw = _mm_mullo_epi16(alpha, opacity_vec);
            let tmp = _mm_add_epi16(raw, round_128);
            let sa = _mm_srli_epi16(_mm_add_epi16(tmp, _mm_srli_epi16(tmp, 8)), 8);
            let inv_sa = _mm_sub_epi16(ones_255, sa);

            // result_rgb = div255(src * sa + dst * inv_sa)
            let src_term = _mm_mullo_epi16(s16, sa);
            let dst_term = _mm_mullo_epi16(d16, inv_sa);
            let sum = _mm_add_epi16(src_term, dst_term);
            let tmp2 = _mm_add_epi16(sum, round_128);
            let blended = _mm_srli_epi16(_mm_add_epi16(tmp2, _mm_srli_epi16(tmp2, 8)), 8);

            // Save original dst alpha before overwriting
            let orig_da = [dst[i + 3], dst[i + 7]];

            // Pack u16 -> u8
            let packed = _mm_packus_epi16(blended, zero);

            // SAFETY: Store 8 bytes back. Same bounds reasoning as the load above.
            _mm_storel_epi64(dst.as_mut_ptr().add(i) as *mut __m128i, packed);

            // Fix up alpha channel using Porter-Duff formula with div255.
            for (px, &da) in orig_da.iter().enumerate() {
                let px_off = i + px * 4;
                let sa_px = div255(src[px_off + 3] as u16 * opacity_val);
                let inv_sa_px = 255u16 - sa_px;
                dst[px_off + 3] = div255(sa_px * 255 + da as u16 * inv_sa_px).min(255) as u8;
            }

            i += 8;
        }
    }

    // Scalar fallback for remainder pixels
    if simd_pixels < pixel_count {
        blend_row_normal_scalar(&src[byte_count..], &mut dst[byte_count..], opacity);
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64, runtime detected)
// ---------------------------------------------------------------------------
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn blend_row_normal_avx2(src: &[u8], dst: &mut [u8], opacity: u8) {
    use std::arch::x86_64::*;

    let pixel_count = src.len() / 4;
    let simd_pixels = pixel_count / 4 * 4; // round down to multiple of 4
    let byte_count = simd_pixels * 4;

    // SAFETY: All intrinsics below require AVX2, which is enabled via
    // #[target_feature(enable = "avx2")] on this function, and the caller
    // verifies AVX2 support at runtime with is_x86_feature_detected!.
    unsafe {
        let zero = _mm256_setzero_si256();
        let opacity_val = opacity as u16;
        let ones_255 = _mm256_set1_epi16(255);
        let opacity_vec = _mm256_set1_epi16(opacity_val as i16);
        let round_128 = _mm256_set1_epi16(128);

        let mut i = 0usize;
        while i < byte_count {
            // SAFETY: We process 4 pixels (16 bytes) per iteration. i + 16 <= byte_count
            // is guaranteed by the rounding.
            let s_bytes_128 = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            let d_bytes_128 = _mm_loadu_si128(dst.as_ptr().add(i) as *const __m128i);

            // Unpack u8 -> u16 into 256-bit register
            let s16 = _mm256_cvtepu8_epi16(s_bytes_128);
            let d16 = _mm256_cvtepu8_epi16(d_bytes_128);

            // Broadcast alpha for each pixel within each 128-bit lane.
            let alpha_lo = _mm256_shufflelo_epi16(s16, 0xFF);
            let alpha = _mm256_shufflehi_epi16(alpha_lo, 0xFF);

            // sa = div255(alpha * opacity)
            let raw = _mm256_mullo_epi16(alpha, opacity_vec);
            let tmp = _mm256_add_epi16(raw, round_128);
            let sa = _mm256_srli_epi16(_mm256_add_epi16(tmp, _mm256_srli_epi16(tmp, 8)), 8);
            let inv_sa = _mm256_sub_epi16(ones_255, sa);

            // result = div255(src * sa + dst * inv_sa)
            let src_term = _mm256_mullo_epi16(s16, sa);
            let dst_term = _mm256_mullo_epi16(d16, inv_sa);
            let sum = _mm256_add_epi16(src_term, dst_term);
            let tmp2 = _mm256_add_epi16(sum, round_128);
            let blended = _mm256_srli_epi16(_mm256_add_epi16(tmp2, _mm256_srli_epi16(tmp2, 8)), 8);

            // Save original dst alpha before overwriting
            let orig_da = [dst[i + 3], dst[i + 7], dst[i + 11], dst[i + 15]];

            // Pack u16 -> u8 (256-bit packus works within 128-bit lanes, need permute)
            let packed = _mm256_packus_epi16(blended, zero);
            let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);

            // SAFETY: Store 16 bytes back. Same bounds reasoning as the load above.
            _mm_storeu_si128(
                dst.as_mut_ptr().add(i) as *mut __m128i,
                _mm256_castsi256_si128(result),
            );

            // Fix up alpha channel using Porter-Duff formula with div255.
            for (px, &da) in orig_da.iter().enumerate() {
                let px_off = i + px * 4;
                let sa_px = div255(src[px_off + 3] as u16 * opacity_val);
                let inv_sa_px = 255u16 - sa_px;
                dst[px_off + 3] = div255(sa_px * 255 + da as u16 * inv_sa_px).min(255) as u8;
            }

            i += 16;
        }
    }

    // Handle remainder with SSE2
    if simd_pixels < pixel_count {
        // SAFETY: SSE2 is baseline on x86_64, and the remainder slice is valid.
        unsafe {
            blend_row_normal_sse2(&src[byte_count..], &mut dst[byte_count..], opacity);
        }
    }
}

// ---------------------------------------------------------------------------
// NEON implementation (aarch64)
// ---------------------------------------------------------------------------
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
unsafe fn blend_row_normal_neon(src: &[u8], dst: &mut [u8], opacity: u8) {
    use std::arch::aarch64::*;

    let pixel_count = src.len() / 4;
    let simd_pixels = pixel_count / 8 * 8; // vld4_u8 loads 8 bytes per channel = 8 pixels
    let byte_count = simd_pixels * 4;

    // SAFETY: All NEON intrinsics below are available because NEON is baseline
    // on aarch64 and this function is only called on aarch64.
    unsafe {
        let mut i = 0usize;
        while i < byte_count {
            // SAFETY: We process 8 pixels (32 bytes) per iteration. vld4_u8 reads
            // 8x4 = 32 bytes. i + 32 <= byte_count is guaranteed by rounding.
            let s = vld4_u8(src.as_ptr().add(i));
            let d = vld4_u8(dst.as_ptr().add(i));

            // Compute sa = div255(src_alpha * opacity)  (8 pixels at once)
            let round_128 = vdupq_n_u16(128);
            let raw_sa = vmull_u8(s.3, vdup_n_u8(opacity));
            let tmp_sa = vaddq_u16(raw_sa, round_128);
            let sa_wide = vshrq_n_u16(vaddq_u16(tmp_sa, vshrq_n_u16(tmp_sa, 8)), 8);
            let sa = vmovn_u16(sa_wide);

            // inv_sa = 255 - sa
            let inv_sa = vsub_u8(vdup_n_u8(255), sa);

            // For each channel: result = div255(src_c * sa + dst_c * inv_sa)
            let div255_neon = |val: uint16x8_t| -> uint8x8_t {
                let tmp = vaddq_u16(val, round_128);
                vmovn_u16(vshrq_n_u16(vaddq_u16(tmp, vshrq_n_u16(tmp, 8)), 8))
            };
            let r = div255_neon(vaddq_u16(vmull_u8(s.0, sa), vmull_u8(d.0, inv_sa)));
            let g = div255_neon(vaddq_u16(vmull_u8(s.1, sa), vmull_u8(d.1, inv_sa)));
            let b = div255_neon(vaddq_u16(vmull_u8(s.2, sa), vmull_u8(d.2, inv_sa)));

            // Alpha: div255(sa * 255 + dst_a * inv_sa), saturating narrow
            let sa255 = vmull_u8(sa, vdup_n_u8(255));
            let da_term = vmull_u8(d.3, inv_sa);
            let a_sum = vaddq_u16(sa255, da_term);
            let a_tmp = vaddq_u16(a_sum, round_128);
            let a_wide = vshrq_n_u16(vaddq_u16(a_tmp, vshrq_n_u16(a_tmp, 8)), 8);
            let a = vqmovn_u16(a_wide);

            let result = uint8x8x4_t(r, g, b, a);
            // SAFETY: Store 32 bytes back, same bounds as load.
            vst4_u8(dst.as_mut_ptr().add(i), result);

            i += 32;
        }
    }

    // Scalar fallback for remainder
    if simd_pixels < pixel_count {
        blend_row_normal_scalar(&src[byte_count..], &mut dst[byte_count..], opacity);
    }
}

/// Blend an entire source row over a destination row (RGBA8, Normal mode).
///
/// Both slices must have equal length and be a multiple of 4 bytes.
/// When the `simd` feature is enabled, uses SSE2 on x86_64 for ~4x throughput.
///
/// # Examples
///
/// ```
/// use ranga::blend::blend_row_normal;
///
/// let src = vec![255, 0, 0, 255, 0, 255, 0, 255]; // red, green
/// let mut dst = vec![0, 0, 255, 255, 0, 0, 255, 255]; // blue, blue
/// blend_row_normal(&src, &mut dst, 128);
/// assert_eq!(dst.len(), 8);
/// ```
pub fn blend_row_normal(src: &[u8], dst: &mut [u8], opacity: u8) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 4, 0);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime, src/dst alignment not required for loadu/storeu
            unsafe { blend_row_normal_avx2(src, dst, opacity) };
            return;
        }
        // SAFETY: SSE2 is baseline on x86_64
        unsafe { blend_row_normal_sse2(src, dst, opacity) };
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline on aarch64
        unsafe { blend_row_normal_neon(src, dst, opacity) };
        return;
    }

    // Scalar fallback for non-SIMD targets or when simd feature is disabled
    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    blend_row_normal_scalar(src, dst, opacity);
}

// ---------------------------------------------------------------------------
// ARGB8 blend (alpha-first layout used by aethersafta)
// ---------------------------------------------------------------------------

/// Blend a single ARGB8 pixel: `[A, R, G, B]` layout.
///
/// Same Porter-Duff compositing as [`blend_pixel`] but for ARGB8 channel
/// order (used by aethersafta). Alpha is at index 0.
///
/// # Examples
///
/// ```
/// use ranga::blend::{BlendMode, blend_pixel_argb};
///
/// let result = blend_pixel_argb([255, 255, 0, 0], [255, 0, 0, 255], BlendMode::Normal, 255);
/// assert!(result[1] > 200); // red channel dominant
/// ```
#[inline]
#[must_use]
pub fn blend_pixel_argb(src: [u8; 4], dst: [u8; 4], mode: BlendMode, opacity: u8) -> [u8; 4] {
    // Convert ARGB → RGBA, blend, convert back
    let rgba_src = [src[1], src[2], src[3], src[0]];
    let rgba_dst = [dst[1], dst[2], dst[3], dst[0]];
    let result = blend_pixel(rgba_src, rgba_dst, mode, opacity);
    [result[3], result[0], result[1], result[2]]
}

/// Blend an entire source row over a destination row (ARGB8, Normal mode).
///
/// ARGB layout: `[A, R, G, B]` per pixel. Used by aethersafta's compositor.
/// The opacity parameter is a fixed-point Q8 value (0–255).
///
/// # Examples
///
/// ```
/// use ranga::blend::blend_row_normal_argb;
///
/// let src = vec![200, 128, 64, 32, 200, 64, 128, 96]; // 2 ARGB pixels
/// let mut dst = vec![255, 0, 0, 255, 255, 0, 0, 255];
/// blend_row_normal_argb(&src, &mut dst, 255);
/// assert_eq!(dst.len(), 8);
/// ```
pub fn blend_row_normal_argb(src: &[u8], dst: &mut [u8], opacity: u8) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % 4, 0);

    let opacity_fp = opacity as u16;
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let sa = div255(s[0] as u16 * opacity_fp);
        if sa == 0 {
            continue;
        }
        if sa >= 255 {
            d.copy_from_slice(s);
            continue;
        }
        let inv_sa = 255u16 - sa;
        d[0] = div255(sa * 255 + d[0] as u16 * inv_sa).min(255) as u8; // A
        d[1] = div255(s[1] as u16 * sa + d[1] as u16 * inv_sa) as u8; // R
        d[2] = div255(s[2] as u16 * sa + d[2] as u16 * inv_sa) as u8; // G
        d[3] = div255(s[3] as u16 * sa + d[3] as u16 * inv_sa) as u8; // B
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_blend_opaque() {
        let src = [255, 0, 0, 255]; // red
        let dst = [0, 0, 255, 255]; // blue
        let result = blend_pixel(src, dst, BlendMode::Normal, 255);
        assert!(result[0] > 200); // mostly red
        assert!(result[2] < 55); // very little blue
    }

    #[test]
    fn normal_blend_transparent() {
        let src = [255, 0, 0, 0]; // transparent red
        let dst = [0, 0, 255, 255]; // blue
        let result = blend_pixel(src, dst, BlendMode::Normal, 255);
        assert_eq!(result, dst); // unchanged
    }

    #[test]
    fn multiply_darkens() {
        let src = [128, 128, 128, 255];
        let dst = [200, 200, 200, 255];
        let result = blend_pixel(src, dst, BlendMode::Multiply, 255);
        assert!(result[0] < dst[0]);
    }

    #[test]
    fn screen_lightens() {
        let src = [128, 128, 128, 255];
        let dst = [50, 50, 50, 255];
        let result = blend_pixel(src, dst, BlendMode::Screen, 255);
        assert!(result[0] > dst[0]);
    }

    #[test]
    fn difference_of_same_is_black() {
        let src = [100, 100, 100, 255];
        let dst = [100, 100, 100, 255];
        let result = blend_pixel(src, dst, BlendMode::Difference, 255);
        assert!(result[0] < 5);
    }

    #[test]
    fn blend_row_preserves_length() {
        let src = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let mut dst = vec![0, 0, 255, 255, 0, 0, 255, 255];
        blend_row_normal(&src, &mut dst, 128);
        assert_eq!(dst.len(), 8);
    }

    #[test]
    fn all_modes_dont_panic() {
        let modes = [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::SoftLight,
            BlendMode::HardLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
        ];
        for mode in modes {
            let _ = blend_pixel([128, 64, 200, 200], [50, 100, 150, 255], mode, 180);
        }
    }

    fn assert_pixels_near(a: &[u8], b: &[u8], tolerance: u8, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av as i16 - bv as i16).unsigned_abs() <= tolerance as u16,
                "{label}: byte {i} differs: {av} vs {bv} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn simd_scalar_equivalence_blend() {
        let src: Vec<u8> = (0..256 * 4).map(|i| (i % 256) as u8).collect();
        let mut dst_scalar = vec![128u8; 256 * 4];
        let mut dst_simd = dst_scalar.clone();
        blend_row_normal_scalar(&src, &mut dst_scalar, 180);
        blend_row_normal(&src, &mut dst_simd, 180);
        // SIMD alpha fixup uses integer division in a different order than scalar,
        // which can produce +/-1 rounding differences on the alpha channel.
        assert_pixels_near(&dst_scalar, &dst_simd, 1, "blend_256px");
    }

    #[test]
    fn simd_scalar_equivalence_blend_odd_count() {
        let src: Vec<u8> = (0..13 * 4).map(|i| (i % 256) as u8).collect();
        let mut dst_scalar = vec![200u8; 13 * 4];
        let mut dst_simd = dst_scalar.clone();
        blend_row_normal_scalar(&src, &mut dst_scalar, 220);
        blend_row_normal(&src, &mut dst_simd, 220);
        assert_pixels_near(&dst_scalar, &dst_simd, 1, "blend_13px");
    }
}
