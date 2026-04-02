//! Pixel format conversion — RGB↔YUV, ARGB↔NV12, format interop.
//!
//! Uses fixed-point BT.601 and BT.709 coefficients for integer-only fast paths.
//! Replaces inline conversion code from rasa, tazama, and tarang with a single
//! shared implementation.

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};

// =========================================================================
// Y-plane computation helper
// =========================================================================

/// Compute BT.601 luminance for a row of RGBA8 pixels: Y = (77*R + 150*G + 29*B) >> 8
fn compute_y_row(rgba: &[u8], y_out: &mut [u8]) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // SAFETY: SSE2 is baseline for x86_64.
        unsafe { compute_y_row_simd_sse2(rgba, y_out, 77, 150, 29) };
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline for aarch64.
        unsafe { compute_y_row_simd_neon(rgba, y_out, 77, 150, 29) };
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    compute_y_row_scalar(rgba, y_out, 77, 150, 29);
}

/// Compute BT.709 luminance for a row: Y = (54*R + 183*G + 18*B) >> 8
fn compute_y_row_bt709(rgba: &[u8], y_out: &mut [u8]) {
    // BT.709: 0.2126*R + 0.7152*G + 0.0722*B, scaled by 256 and rounded
    // (54, 183, 19) sums to 256 so white (255,255,255) correctly maps to Y=255
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        // SAFETY: SSE2 is baseline for x86_64.
        unsafe { compute_y_row_simd_sse2(rgba, y_out, 54, 183, 19) };
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is baseline for aarch64.
        unsafe { compute_y_row_simd_neon(rgba, y_out, 54, 183, 19) };
    }

    #[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    compute_y_row_scalar(rgba, y_out, 54, 183, 19);
}

fn compute_y_row_scalar(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    for (pixel, y) in rgba.chunks_exact(4).zip(y_out.iter_mut()) {
        *y = ((cr * pixel[0] as u16 + cg * pixel[1] as u16 + cb * pixel[2] as u16) >> 8) as u8;
    }
}

// x86_64: SSE2 Y-row using _mm_madd_epi16 for pair-wise multiply-accumulate.
// Processes 2 pixels at a time.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn compute_y_row_simd_sse2(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    use std::arch::x86_64::*;
    let pixel_count = rgba.len() / 4;
    let simd_pixels = pixel_count / 2 * 2;

    unsafe {
        let zero = _mm_setzero_si128();
        let coeffs = _mm_setr_epi16(
            cr as i16, cg as i16, cb as i16, 0, cr as i16, cg as i16, cb as i16, 0,
        );

        let mut i = 0usize;
        let mut oi = 0usize;
        while oi + 2 <= simd_pixels {
            // Load 2 pixels (8 bytes)
            let px = _mm_loadl_epi64(rgba.as_ptr().add(i * 4) as *const __m128i);
            let px16 = _mm_unpacklo_epi8(px, zero);

            // _mm_madd_epi16: multiply pairs and add adjacent
            // [R0*cr+G0*cg, B0*cb+0, R1*cr+G1*cg, B1*cb+0] as i32
            let products = _mm_madd_epi16(px16, coeffs);

            // Horizontal add adjacent i32 pairs to get Y per pixel
            // products = [RG0, B0, RG1, B1]
            // Shuffle to [B0, RG0, B1, RG1] and add
            let shuffled = _mm_shuffle_epi32(products, 0b10_11_00_01);
            let sums = _mm_add_epi32(products, shuffled);
            // sums = [RG0+B0, ?, RG1+B1, ?] — Y values in slots 0 and 2

            // Shift right by 8
            let y_vals = _mm_srli_epi32(sums, 8);

            // Extract from slots 0 and 2
            y_out[oi] = _mm_extract_epi16(y_vals, 0) as u8;
            y_out[oi + 1] = _mm_extract_epi16(y_vals, 4) as u8;

            i += 2;
            oi += 2;
        }
    }

    if simd_pixels < pixel_count {
        compute_y_row_scalar(
            &rgba[simd_pixels * 4..],
            &mut y_out[simd_pixels..],
            cr,
            cg,
            cb,
        );
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
unsafe fn compute_y_row_simd_neon(rgba: &[u8], y_out: &mut [u8], cr: u16, cg: u16, cb: u16) {
    use std::arch::aarch64::*;
    let pixel_count = rgba.len() / 4;
    let simd_pixels = pixel_count / 8 * 8;

    // SAFETY: NEON is baseline for aarch64. vld4_u8 loads 8 pixels.
    unsafe {
        debug_assert!(cr <= 255, "NEON Y coefficient cr={cr} exceeds u8");
        debug_assert!(cg <= 255, "NEON Y coefficient cg={cg} exceeds u8");
        debug_assert!(cb <= 255, "NEON Y coefficient cb={cb} exceeds u8");
        let vcr = vdup_n_u8(cr as u8);
        let vcg = vdup_n_u8(cg as u8);
        let vcb = vdup_n_u8(cb as u8);

        let mut i = 0usize;
        let mut oi = 0usize;
        while oi + 8 <= simd_pixels {
            let px = vld4_u8(rgba.as_ptr().add(i));
            let mut acc = vmull_u8(px.0, vcr);
            acc = vmlal_u8(acc, px.1, vcg);
            acc = vmlal_u8(acc, px.2, vcb);
            let y = vshrn_n_u16(acc, 8);
            vst1_u8(y_out.as_mut_ptr().add(oi), y);
            i += 32;
            oi += 8;
        }
    }

    if simd_pixels < pixel_count {
        compute_y_row_scalar(
            &rgba[simd_pixels * 4..],
            &mut y_out[simd_pixels..],
            cr,
            cg,
            cb,
        );
    }
}

// =========================================================================
// YUV→RGBA SSE2 row helpers (shared by BT.601, BT.709, BT.2020)
// =========================================================================

/// Convert a row of YUV420p data to RGBA using SSE2 intrinsics.
///
/// Processes 8 pixels at a time. U/V are half-width (4:2:0 subsampling), so
/// each U/V value covers 2 horizontal pixels.
///
/// Coefficients:
/// - `cr_v`: R = Y + (cr_v * V) >> 8
/// - `cg_u`, `cg_v`: G = Y - (cg_u * U + cg_v * V) >> 8
/// - `cb_u`: B = Y + (cb_u * U) >> 8
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn yuv_row_to_rgba_sse2(
    y_row: &[u8],
    u_row: &[u8],
    v_row: &[u8],
    rgba_out: &mut [u8],
    cr_v: i16,
    cg_u: i16,
    cg_v: i16,
    cb_u: i16,
) {
    use std::arch::x86_64::*;

    let width = y_row.len();
    let simd_width = width / 8 * 8; // process 8 pixels at a time

    unsafe {
        let zero = _mm_setzero_si128();
        let alpha = _mm_set1_epi16(255);
        let v_cr_v = _mm_set1_epi16(cr_v);
        let v_cg_u = _mm_set1_epi16(cg_u);
        let v_cg_v = _mm_set1_epi16(cg_v);
        let v_cb_u = _mm_set1_epi16(cb_u);
        let bias = _mm_set1_epi16(128);

        let mut x = 0usize;
        while x + 8 <= simd_width {
            // 1. Load 8 Y values, zero-extend to i16
            let y8 = _mm_loadl_epi64(y_row.as_ptr().add(x) as *const __m128i);
            let y16 = _mm_unpacklo_epi8(y8, zero);

            // 2. Load 4 U and 4 V values, zero-extend to i16, subtract 128
            let cx = x / 2;
            let u4 = _mm_cvtsi32_si128(i32::from_ne_bytes([
                u_row[cx],
                u_row[cx + 1],
                u_row[cx + 2],
                u_row[cx + 3],
            ]));
            let v4 = _mm_cvtsi32_si128(i32::from_ne_bytes([
                v_row[cx],
                v_row[cx + 1],
                v_row[cx + 2],
                v_row[cx + 3],
            ]));

            // 3. Duplicate each U/V to cover 2 pixels: [u0,u0,u1,u1,u2,u2,u3,u3]
            let u_dup = _mm_unpacklo_epi8(u4, u4); // byte-level interleave with self
            let v_dup = _mm_unpacklo_epi8(v4, v4);

            // Zero-extend to i16 and subtract 128
            let u16 = _mm_sub_epi16(_mm_unpacklo_epi8(u_dup, zero), bias);
            let v16 = _mm_sub_epi16(_mm_unpacklo_epi8(v_dup, zero), bias);

            // 4. R = Y + (cr_v * V) >> 8
            let r16 = _mm_add_epi16(y16, _mm_srai_epi16(_mm_mullo_epi16(v_cr_v, v16), 8));
            // Clamp [0, 255]: pack to u8 with saturation, then unpack back
            let r_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(r16, zero), zero);

            // 5. G = Y - (cg_u * U + cg_v * V) >> 8
            let gu = _mm_mullo_epi16(v_cg_u, u16);
            let gv = _mm_mullo_epi16(v_cg_v, v16);
            let g16 = _mm_sub_epi16(y16, _mm_srai_epi16(_mm_add_epi16(gu, gv), 8));
            let g_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(g16, zero), zero);

            // 6. B = Y + (cb_u * U) >> 8
            let b16 = _mm_add_epi16(y16, _mm_srai_epi16(_mm_mullo_epi16(v_cb_u, u16), 8));
            let b_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(b16, zero), zero);

            // 7. Interleave R,G,B,A=255 and store 32 bytes
            // Pack clamped i16 back to u8 for byte-level interleaving
            let r8 = _mm_packus_epi16(r_clamped, zero);
            let g8 = _mm_packus_epi16(g_clamped, zero);
            let b8 = _mm_packus_epi16(b_clamped, zero);
            let a8 = _mm_packus_epi16(alpha, zero);

            // Interleave: RG low, RG high, BA low, BA high
            let rg_lo = _mm_unpacklo_epi8(r8, g8); // [r0,g0,r1,g1,r2,g2,r3,g3,...]
            let ba_lo = _mm_unpacklo_epi8(b8, a8); // [b0,a0,b1,a1,b2,a2,b3,a3,...]

            // Now interleave 16-bit pairs: [r0,g0,b0,a0, r1,g1,b1,a1, ...]
            let rgba_0 = _mm_unpacklo_epi16(rg_lo, ba_lo); // pixels 0-3
            let rgba_1 = _mm_unpackhi_epi16(rg_lo, ba_lo); // pixels 4-7

            let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
            _mm_storeu_si128(out_ptr as *mut __m128i, rgba_0);
            _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, rgba_1);

            x += 8;
        }
    }

    // Scalar tail for remaining pixels
    for x in simd_width..width {
        let cx = (x / 2).min(u_row.len().saturating_sub(1));
        let yi = y_row[x] as i16;
        let u = u_row[cx] as i16 - 128;
        let v = v_row[cx] as i16 - 128;
        let oi = x * 4;
        rgba_out[oi] = (yi + ((cr_v * v) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 1] = (yi - ((cg_u * u + cg_v * v) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 2] = (yi + ((cb_u * u) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 3] = 255;
    }
}

/// NV12 variant: U/V are interleaved as [U0,V0,U1,V1,...] instead of separate planes.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(clippy::needless_range_loop)]
unsafe fn yuv_row_to_rgba_nv12_sse2(
    y_row: &[u8],
    uv_row: &[u8],
    rgba_out: &mut [u8],
    cr_v: i16,
    cg_u: i16,
    cg_v: i16,
    cb_u: i16,
) {
    use std::arch::x86_64::*;

    let width = y_row.len();
    let simd_width = width / 8 * 8;

    unsafe {
        let zero = _mm_setzero_si128();
        let alpha = _mm_set1_epi16(255);
        let v_cr_v = _mm_set1_epi16(cr_v);
        let v_cg_u = _mm_set1_epi16(cg_u);
        let v_cg_v = _mm_set1_epi16(cg_v);
        let v_cb_u = _mm_set1_epi16(cb_u);
        let bias = _mm_set1_epi16(128);
        // Mask to extract even bytes (U values from interleaved UV)
        let even_mask = _mm_set1_epi16(0x00FF);

        let mut x = 0usize;
        while x + 8 <= simd_width {
            // 1. Load 8 Y values
            let y8 = _mm_loadl_epi64(y_row.as_ptr().add(x) as *const __m128i);
            let y16 = _mm_unpacklo_epi8(y8, zero);

            // 2. Load 8 bytes of interleaved UV: [U0,V0,U1,V1,U2,V2,U3,V3]
            let cx = x / 2;
            let uv8 = _mm_loadl_epi64(uv_row.as_ptr().add(cx * 2) as *const __m128i);

            // Deinterleave: extract U (even bytes) and V (odd bytes)
            let u_bytes = _mm_and_si128(uv8, even_mask); // U in low byte of each 16-bit word
            let v_bytes = _mm_srli_epi16(uv8, 8); // V shifted down

            // u_bytes/v_bytes are already i16 with values in [0,255]
            // We need to duplicate each for 2 pixels. The values are in 16-bit lanes:
            // u_bytes = [u0, u1, u2, u3, 0, 0, 0, 0] as i16
            // Pack to bytes, duplicate, unpack back
            let u4 = _mm_packus_epi16(u_bytes, zero); // [u0,u1,u2,u3,0,0,...] as bytes
            let v4 = _mm_packus_epi16(v_bytes, zero);

            let u_dup = _mm_unpacklo_epi8(u4, u4); // [u0,u0,u1,u1,u2,u2,u3,u3,...]
            let v_dup = _mm_unpacklo_epi8(v4, v4);

            let u16 = _mm_sub_epi16(_mm_unpacklo_epi8(u_dup, zero), bias);
            let v16 = _mm_sub_epi16(_mm_unpacklo_epi8(v_dup, zero), bias);

            // Same math as planar version
            let r16 = _mm_add_epi16(y16, _mm_srai_epi16(_mm_mullo_epi16(v_cr_v, v16), 8));
            let r_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(r16, zero), zero);

            let gu = _mm_mullo_epi16(v_cg_u, u16);
            let gv = _mm_mullo_epi16(v_cg_v, v16);
            let g16 = _mm_sub_epi16(y16, _mm_srai_epi16(_mm_add_epi16(gu, gv), 8));
            let g_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(g16, zero), zero);

            let b16 = _mm_add_epi16(y16, _mm_srai_epi16(_mm_mullo_epi16(v_cb_u, u16), 8));
            let b_clamped = _mm_unpacklo_epi8(_mm_packus_epi16(b16, zero), zero);

            let r8 = _mm_packus_epi16(r_clamped, zero);
            let g8 = _mm_packus_epi16(g_clamped, zero);
            let b8 = _mm_packus_epi16(b_clamped, zero);
            let a8 = _mm_packus_epi16(alpha, zero);

            let rg_lo = _mm_unpacklo_epi8(r8, g8);
            let ba_lo = _mm_unpacklo_epi8(b8, a8);

            let rgba_0 = _mm_unpacklo_epi16(rg_lo, ba_lo);
            let rgba_1 = _mm_unpackhi_epi16(rg_lo, ba_lo);

            let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
            _mm_storeu_si128(out_ptr as *mut __m128i, rgba_0);
            _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, rgba_1);

            x += 8;
        }
    }

    // Scalar tail
    let cw = width.div_ceil(2);
    for x in simd_width..width {
        let cx = (x / 2).min(cw.saturating_sub(1));
        let yi = y_row[x] as i16;
        let u = uv_row[cx * 2] as i16 - 128;
        let v = uv_row[cx * 2 + 1] as i16 - 128;
        let oi = x * 4;
        rgba_out[oi] = (yi + ((cr_v * v) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 1] = (yi - ((cg_u * u + cg_v * v) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 2] = (yi + ((cb_u * u) >> 8)).clamp(0, 255) as u8;
        rgba_out[oi + 3] = 255;
    }
}

// =========================================================================
// BT.601 conversions
// =========================================================================

/// Convert RGBA8 buffer to YUV420p (BT.601).
///
/// Uses BT.601 fixed-point coefficients. The chroma planes (U, V) are
/// subsampled 2x2 from the top-left pixel of each block.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();
/// assert_eq!(yuv.format(), PixelFormat::Yuv420p);
/// assert!(yuv.data()[0] > 250);
/// ```
#[must_use = "returns a new YUV420p buffer"]
pub fn rgba_to_yuv420p(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    // Chroma loops must stay within cw*ch bounds — clamp to even pixel pairs
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 (BT.601).
///
/// Alpha is set to 255 for all pixels.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba(&yuv).unwrap();
/// assert!((back.data()[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA buffer"]
pub fn yuv420p_to_rgba(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * (h.div_ceil(2));
    let ch = h.div_ceil(2);
    let mut rgba = vec![0u8; w * h * 4];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            let y_start = y * w;
            let u_start = u_off + cy * cw;
            let v_start = v_off + cy * cw;
            let rgba_start = y * w * 4;
            // SAFETY: SSE2 is baseline for x86_64.
            unsafe {
                yuv_row_to_rgba_sse2(
                    &buf.data[y_start..y_start + w],
                    &buf.data[u_start..u_start + cw],
                    &buf.data[v_start..v_start + cw],
                    &mut rgba[rgba_start..rgba_start + w * 4],
                    359,
                    88,
                    183,
                    454,
                );
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            for x in 0..w {
                let cx = (x / 2).min(cw.saturating_sub(1));
                let yi = buf.data[y * w + x] as i16;
                let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
                let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
                let oi = (y * w + x) * 4;
                rgba[oi] = (yi + ((359 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 1] = (yi - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 2] = (yi + ((454 * u) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 3] = 255;
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }
}

/// Convert ARGB8 buffer to NV12 (semi-planar YUV 4:2:0, BT.601).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Argb8).unwrap();
/// let nv12 = convert::argb_to_nv12(&argb).unwrap();
/// assert_eq!(nv12.format(), PixelFormat::Nv12);
/// ```
#[must_use = "returns a new NV12 buffer"]
pub fn argb_to_nv12(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!(
            "argb_to_nv12: expected Argb8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let mut nv12 = vec![0u8; w * h + (w.div_ceil(2)) * (h.div_ceil(2)) * 2];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as u16;
            let g = buf.data[i + 2] as u16;
            let b = buf.data[i + 3] as u16;
            nv12[y * w + x] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        }
    }
    let uv_off = w * h;
    let nv12_cw = w.div_ceil(2);
    let nv12_ch = h.div_ceil(2);
    for y in (0..nv12_ch * 2).step_by(2) {
        for x in (0..nv12_cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i + 1] as i32;
            let g = buf.data[i + 2] as i32;
            let b = buf.data[i + 3] as i32;
            let ci = (y / 2) * nv12_cw * 2 + x;
            nv12[uv_off + ci] = ((-43 * r - 85 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            nv12[uv_off + ci + 1] =
                ((128 * r - 107 * g - 21 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(nv12, buf.width, buf.height, PixelFormat::Nv12)
}

// =========================================================================
// BT.709 conversions (HD video — replaces tazama/tarang inline BT.709)
// =========================================================================

/// Convert RGBA8 buffer to YUV420p using BT.709 coefficients.
///
/// BT.709 is the standard for HD video. Fixed-point:
/// Y = (54*R + 183*G + 18*B) >> 8
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt709(&rgba).unwrap();
/// assert!(yuv.data()[0] > 250);
/// ```
#[must_use = "returns a new YUV420p BT.709 buffer"]
pub fn rgba_to_yuv420p_bt709(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p_bt709: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row_bt709(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-29 * r - 99 * g + 128 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((128 * r - 116 * g - 12 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 using BT.709 coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt709(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba_bt709(&yuv).unwrap();
/// assert!((back.data()[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA BT.709 buffer"]
pub fn yuv420p_to_rgba_bt709(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba_bt709: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let mut rgba = vec![0u8; w * h * 4];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            let y_start = y * w;
            let u_start = u_off + cy * cw;
            let v_start = v_off + cy * cw;
            let rgba_start = y * w * 4;
            // SAFETY: SSE2 is baseline for x86_64.
            unsafe {
                yuv_row_to_rgba_sse2(
                    &buf.data[y_start..y_start + w],
                    &buf.data[u_start..u_start + cw],
                    &buf.data[v_start..v_start + cw],
                    &mut rgba[rgba_start..rgba_start + w * 4],
                    403,
                    48,
                    120,
                    475,
                );
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            for x in 0..w {
                let cx = (x / 2).min(cw.saturating_sub(1));
                let yi = buf.data[y * w + x] as i16;
                let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
                let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
                let oi = (y * w + x) * 4;
                rgba[oi] = (yi + ((403 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 1] = (yi - ((48 * u + 120 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 2] = (yi + ((475 * u) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 3] = 255;
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }
}

// =========================================================================
// BT.2020 conversions (UHD/HDR video — wide-gamut)
// =========================================================================

/// Compute BT.2020 luminance for a row: Y = (67*R + 174*G + 15*B) >> 8
fn compute_y_row_bt2020(rgba: &[u8], y_out: &mut [u8]) {
    for (pixel, y) in rgba.chunks_exact(4).zip(y_out.iter_mut()) {
        *y = ((67 * pixel[0] as u16 + 174 * pixel[1] as u16 + 15 * pixel[2] as u16) >> 8) as u8;
    }
}

/// Convert RGBA8 buffer to YUV420p using BT.2020 coefficients.
///
/// BT.2020 is the standard for UHD/HDR video with a wider gamut than BT.709.
/// Fixed-point Y coefficients: Y = (67*R + 174*G + 15*B) >> 8
/// (derived from BT.2020 non-constant luminance: 0.2627 R + 0.6780 G + 0.0593 B)
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt2020(&rgba).unwrap();
/// assert!(yuv.data()[0] > 250);
/// ```
#[must_use = "returns a new YUV420p BT.2020 buffer"]
pub fn rgba_to_yuv420p_bt2020(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba_to_yuv420p_bt2020: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut yuv = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        let row_start = y * w * 4;
        compute_y_row_bt2020(
            &buf.data[row_start..row_start + w * 4],
            &mut yuv[y * w..y * w + w],
        );
    }
    // BT.2020 Cb/Cr fixed-point coefficients (8-bit approximation)
    // Cb = (-18*R - 46*G + 64*B + 128*256) >> 8 (scaled from -0.0693*R - 0.1786*G + 0.2480*B)
    // Cr = (64*R - 58*G - 6*B + 128*256) >> 8   (scaled from 0.2480*R - 0.2252*G - 0.0228*B)
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    for y in (0..ch * 2).step_by(2) {
        for x in (0..cw * 2).step_by(2) {
            let i = (y * w + x) * 4;
            let r = buf.data[i] as i32;
            let g = buf.data[i + 1] as i32;
            let b = buf.data[i + 2] as i32;
            let ci = (y / 2) * cw + (x / 2);
            yuv[u_off + ci] = ((-18 * r - 46 * g + 64 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
            yuv[v_off + ci] = ((64 * r - 58 * g - 6 * b + 128 * 256) >> 8).clamp(0, 255) as u8;
        }
    }
    PixelBuffer::new(yuv, buf.width, buf.height, PixelFormat::Yuv420p)
}

/// Convert YUV420p buffer to RGBA8 using BT.2020 coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let yuv = convert::rgba_to_yuv420p_bt2020(&rgba).unwrap();
/// let back = convert::yuv420p_to_rgba_bt2020(&yuv).unwrap();
/// assert!((back.data()[0] as i16 - 128).unsigned_abs() < 10);
/// ```
#[must_use = "returns a new RGBA BT.2020 buffer"]
pub fn yuv420p_to_rgba_bt2020(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Yuv420p {
        return Err(RangaError::InvalidFormat(format!(
            "yuv420p_to_rgba_bt2020: expected Yuv420p, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let mut rgba = vec![0u8; w * h * 4];
    // BT.2020 inverse:
    // R = Y + 1.4746 * Cr → Y + (377 * V) >> 8
    // G = Y - 0.1646 * Cb - 0.5714 * Cr → Y - (42 * U + 146 * V) >> 8
    // B = Y + 1.8814 * Cb → Y + (481 * U) >> 8

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            let y_start = y * w;
            let u_start = u_off + cy * cw;
            let v_start = v_off + cy * cw;
            let rgba_start = y * w * 4;
            // SAFETY: SSE2 is baseline for x86_64.
            unsafe {
                yuv_row_to_rgba_sse2(
                    &buf.data[y_start..y_start + w],
                    &buf.data[u_start..u_start + cw],
                    &buf.data[v_start..v_start + cw],
                    &mut rgba[rgba_start..rgba_start + w * 4],
                    377,
                    42,
                    146,
                    481,
                );
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            for x in 0..w {
                let cx = (x / 2).min(cw.saturating_sub(1));
                let yi = buf.data[y * w + x] as i16;
                let u = buf.data[u_off + cy * cw + cx] as i16 - 128;
                let v = buf.data[v_off + cy * cw + cx] as i16 - 128;
                let oi = (y * w + x) * 4;
                rgba[oi] = (yi + ((377 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 1] = (yi - ((42 * u + 146 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 2] = (yi + ((481 * u) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 3] = 255;
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }
}

// =========================================================================
// NV12 → RGBA (replaces tazama/tarang inline NV12 handling)
// =========================================================================

/// Convert NV12 buffer to RGBA8 (BT.601).
///
/// NV12 has a Y plane followed by interleaved UV.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255, 128, 128, 128].repeat(16), 4, 4, PixelFormat::Argb8).unwrap();
/// let nv12 = convert::argb_to_nv12(&argb).unwrap();
/// let rgba = convert::nv12_to_rgba(&nv12).unwrap();
/// assert_eq!(rgba.format(), PixelFormat::Rgba8);
/// ```
#[must_use = "returns a new RGBA buffer"]
pub fn nv12_to_rgba(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Nv12 {
        return Err(RangaError::InvalidFormat(format!(
            "nv12_to_rgba: expected Nv12, got {:?}",
            buf.format
        )));
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_off = w * h;
    let uv_stride = cw * 2; // interleaved UV pairs per row
    let mut rgba = vec![0u8; w * h * 4];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            let y_start = y * w;
            let uv_start = uv_off + cy * uv_stride;
            let rgba_start = y * w * 4;
            // SAFETY: SSE2 is baseline for x86_64.
            unsafe {
                yuv_row_to_rgba_nv12_sse2(
                    &buf.data[y_start..y_start + w],
                    &buf.data[uv_start..uv_start + uv_stride],
                    &mut rgba[rgba_start..rgba_start + w * 4],
                    359,
                    88,
                    183,
                    454,
                );
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        for y in 0..h {
            let cy = (y / 2).min(ch.saturating_sub(1));
            for x in 0..w {
                let cx = (x / 2).min(cw.saturating_sub(1));
                let yi = buf.data[y * w + x] as i16;
                let uv_idx = uv_off + cy * uv_stride + cx * 2;
                let u = buf.data[uv_idx] as i16 - 128;
                let v = buf.data[uv_idx + 1] as i16 - 128;
                let oi = (y * w + x) * 4;
                rgba[oi] = (yi + ((359 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 1] = (yi - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 2] = (yi + ((454 * u) >> 8)).clamp(0, 255) as u8;
                rgba[oi + 3] = 255;
            }
        }
        PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
    }
}

// =========================================================================
// Format conversions: RGB8 ↔ RGBA8, ARGB8 ↔ RGBA8, RgbaF32 ↔ Rgba8
// =========================================================================

/// Convert RGB8 to RGBA8 (add alpha=255).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgb = PixelBuffer::new(vec![255, 0, 0, 0, 255, 0], 2, 1, PixelFormat::Rgb8).unwrap();
/// let rgba = convert::rgb8_to_rgba8(&rgb).unwrap();
/// assert_eq!(rgba.data(), vec![255, 0, 0, 255, 0, 255, 0, 255]);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn rgb8_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgb8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgb8_to_rgba8: expected Rgb8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        rgba[i * 4] = buf.data[i * 3];
        rgba[i * 4 + 1] = buf.data[i * 3 + 1];
        rgba[i * 4 + 2] = buf.data[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to RGB8 (strip alpha).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![255, 0, 0, 128, 0, 255, 0, 64], 2, 1, PixelFormat::Rgba8).unwrap();
/// let rgb = convert::rgba8_to_rgb8(&rgba).unwrap();
/// assert_eq!(rgb.data(), vec![255, 0, 0, 0, 255, 0]);
/// ```
#[must_use = "returns a new RGB8 buffer"]
pub fn rgba8_to_rgb8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_rgb8: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgb = vec![0u8; n * 3];
    for i in 0..n {
        rgb[i * 3] = buf.data[i * 4];
        rgb[i * 3 + 1] = buf.data[i * 4 + 1];
        rgb[i * 3 + 2] = buf.data[i * 4 + 2];
    }
    PixelBuffer::new(rgb, buf.width, buf.height, PixelFormat::Rgb8)
}

/// Convert ARGB8 to RGBA8 (channel reorder, used by aethersafta).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let argb = PixelBuffer::new(vec![255, 128, 64, 32], 1, 1, PixelFormat::Argb8).unwrap();
/// let rgba = convert::argb8_to_rgba8(&argb).unwrap();
/// assert_eq!(rgba.data(), vec![128, 64, 32, 255]);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn argb8_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Argb8 {
        return Err(RangaError::InvalidFormat(format!(
            "argb8_to_rgba8: expected Argb8, got {:?}",
            buf.format
        )));
    }
    let mut rgba = vec![0u8; buf.data.len()];
    for (src, dst) in buf.data.chunks_exact(4).zip(rgba.chunks_exact_mut(4)) {
        dst[0] = src[1];
        dst[1] = src[2];
        dst[2] = src[3];
        dst[3] = src[0];
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to ARGB8 (channel reorder).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let rgba = PixelBuffer::new(vec![128, 64, 32, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let argb = convert::rgba8_to_argb8(&rgba).unwrap();
/// assert_eq!(argb.data(), vec![255, 128, 64, 32]);
/// ```
#[must_use = "returns a new ARGB8 buffer"]
pub fn rgba8_to_argb8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_argb8: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let mut argb = vec![0u8; buf.data.len()];
    for (src, dst) in buf.data.chunks_exact(4).zip(argb.chunks_exact_mut(4)) {
        dst[0] = src[3];
        dst[1] = src[0];
        dst[2] = src[1];
        dst[3] = src[2];
    }
    PixelBuffer::new(argb, buf.width, buf.height, PixelFormat::Argb8)
}

/// Convert RgbaF32 to RGBA8 (float → byte, clamped).
///
/// Each f32 channel is in 0.0–1.0 and scaled to 0–255.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let f32_data: Vec<u8> = [0.5f32, 0.0, 1.0, 1.0]
///     .iter()
///     .flat_map(|v| v.to_ne_bytes())
///     .collect();
/// let buf = PixelBuffer::new(f32_data, 1, 1, PixelFormat::RgbaF32).unwrap();
/// let rgba = convert::rgbaf32_to_rgba8(&buf).unwrap();
/// assert_eq!(rgba.data()[0], 128);
/// assert_eq!(rgba.data()[2], 255);
/// ```
#[must_use = "returns a new RGBA8 buffer"]
pub fn rgbaf32_to_rgba8(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::RgbaF32 {
        return Err(RangaError::InvalidFormat(format!(
            "rgbaf32_to_rgba8: expected RgbaF32, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let base = i * 16;
        for c in 0..4 {
            let bytes = [
                buf.data[base + c * 4],
                buf.data[base + c * 4 + 1],
                buf.data[base + c * 4 + 2],
                buf.data[base + c * 4 + 3],
            ];
            let v = f32::from_ne_bytes(bytes);
            rgba[i * 4 + c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    PixelBuffer::new(rgba, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Convert RGBA8 to RgbaF32 (byte → float, 0.0–1.0).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::convert;
///
/// let buf = PixelBuffer::new(vec![128, 0, 255, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// let f32buf = convert::rgba8_to_rgbaf32(&buf).unwrap();
/// let r = f32::from_ne_bytes([f32buf.data()[0], f32buf.data()[1], f32buf.data()[2], f32buf.data()[3]]);
/// assert!((r - 128.0 / 255.0).abs() < 0.01);
/// ```
#[must_use = "returns a new RgbaF32 buffer"]
pub fn rgba8_to_rgbaf32(buf: &PixelBuffer) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!(
            "rgba8_to_rgbaf32: expected Rgba8, got {:?}",
            buf.format
        )));
    }
    let n = buf.pixel_count();
    let mut f32data = vec![0u8; n * 16];
    for i in 0..n {
        for c in 0..4 {
            let v = buf.data[i * 4 + c] as f32 / 255.0;
            let base = i * 16 + c * 4;
            f32data[base..base + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }
    PixelBuffer::new(f32data, buf.width, buf.height, PixelFormat::RgbaF32)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_to_yuv_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        assert_eq!(yuv.format, PixelFormat::Yuv420p);
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn rgba_to_yuv_black() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        assert_eq!(yuv.data[0], 0);
    }

    #[test]
    fn yuv_to_rgba_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&rgba).unwrap();
        let back = yuv420p_to_rgba(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn wrong_format_rejected() {
        let buf = PixelBuffer::new(vec![0; 4 * 4 * 3], 4, 4, PixelFormat::Rgb8).unwrap();
        assert!(rgba_to_yuv420p(&buf).is_err());
    }

    #[test]
    fn bt709_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&buf).unwrap();
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn bt709_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&rgba).unwrap();
        let back = yuv420p_to_rgba_bt709(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn bt709_different_from_bt601() {
        let red: Vec<u8> = [255, 0, 0, 255].repeat(16);
        let buf = PixelBuffer::new(red, 4, 4, PixelFormat::Rgba8).unwrap();
        let y601 = rgba_to_yuv420p(&buf).unwrap().data[0];
        let y709 = rgba_to_yuv420p_bt709(&buf).unwrap().data[0];
        assert_ne!(y601, y709, "BT.601 and BT.709 should differ for red");
    }

    #[test]
    fn nv12_to_rgba_roundtrip() {
        let argb_data: Vec<u8> = [255, 128, 128, 128].repeat(16);
        let argb = PixelBuffer::new(argb_data, 4, 4, PixelFormat::Argb8).unwrap();
        let nv12 = argb_to_nv12(&argb).unwrap();
        let rgba = nv12_to_rgba(&nv12).unwrap();
        assert_eq!(rgba.format, PixelFormat::Rgba8);
        assert!((rgba.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn rgb8_rgba8_roundtrip() {
        let rgb =
            PixelBuffer::new(vec![100, 150, 200, 50, 75, 25], 2, 1, PixelFormat::Rgb8).unwrap();
        let rgba = rgb8_to_rgba8(&rgb).unwrap();
        assert_eq!(rgba.data[3], 255);
        let back = rgba8_to_rgb8(&rgba).unwrap();
        assert_eq!(back.data, rgb.data);
    }

    #[test]
    fn argb8_rgba8_roundtrip() {
        let argb = PixelBuffer::new(vec![200, 100, 50, 25], 1, 1, PixelFormat::Argb8).unwrap();
        let rgba = argb8_to_rgba8(&argb).unwrap();
        assert_eq!(rgba.data, vec![100, 50, 25, 200]);
        let back = rgba8_to_argb8(&rgba).unwrap();
        assert_eq!(back.data, argb.data);
    }

    #[test]
    fn rgbaf32_rgba8_roundtrip() {
        let rgba = PixelBuffer::new(vec![128, 64, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let f32buf = rgba8_to_rgbaf32(&rgba).unwrap();
        let back = rgbaf32_to_rgba8(&f32buf).unwrap();
        for i in 0..4 {
            assert!(
                (rgba.data[i] as i16 - back.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}"
            );
        }
    }

    // Edge case: odd dimensions must not panic
    #[test]
    fn yuv420p_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 5 * 3 * 4], 5, 3, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p(&buf).unwrap();
        let _back = yuv420p_to_rgba(&yuv).unwrap();
    }

    #[test]
    fn yuv420p_bt709_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 7 * 5 * 4], 7, 5, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt709(&buf).unwrap();
        let _back = yuv420p_to_rgba_bt709(&yuv).unwrap();
    }

    #[test]
    fn nv12_odd_dimensions_no_panic() {
        let buf = PixelBuffer::new(vec![128; 5 * 3 * 4], 5, 3, PixelFormat::Argb8).unwrap();
        let nv12 = argb_to_nv12(&buf).unwrap();
        let _rgba = nv12_to_rgba(&nv12).unwrap();
    }

    #[test]
    fn bt2020_white() {
        let buf = PixelBuffer::new(vec![255; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt2020(&buf).unwrap();
        assert!(yuv.data[0] > 250);
    }

    #[test]
    fn bt2020_roundtrip() {
        let rgba = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let yuv = rgba_to_yuv420p_bt2020(&rgba).unwrap();
        let back = yuv420p_to_rgba_bt2020(&yuv).unwrap();
        assert!((back.data[0] as i16 - 128).unsigned_abs() < 10);
    }

    #[test]
    fn bt2020_different_from_bt709() {
        let red: Vec<u8> = [255, 0, 0, 255].repeat(16);
        let buf = PixelBuffer::new(red, 4, 4, PixelFormat::Rgba8).unwrap();
        let y709 = rgba_to_yuv420p_bt709(&buf).unwrap().data[0];
        let y2020 = rgba_to_yuv420p_bt2020(&buf).unwrap().data[0];
        assert_ne!(y709, y2020, "BT.709 and BT.2020 should differ for red");
    }

    #[test]
    fn yuv420p_1x1_no_panic() {
        let buf = PixelBuffer::new(vec![128; 4], 1, 1, PixelFormat::Rgba8).unwrap();
        let _yuv = rgba_to_yuv420p(&buf).unwrap();
    }
}
