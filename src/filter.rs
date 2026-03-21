//! CPU image filters — brightness, contrast, saturation, levels, curves,
//! blur, sharpen, hue shift, color balance, 3D LUT, vignette, and noise.
//!
//! All filters operate on RGBA8 pixel buffers. In-place filters take
//! `&mut PixelBuffer`; spatial filters (blur, sharpen) return a new buffer.
//!
//! When the `parallel` feature is enabled, large-buffer operations use
//! rayon for row-parallel processing.

use crate::RangaError;
use crate::pixel::{PixelBuffer, PixelFormat};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Adjust brightness of an RGBA8 buffer in-place.
///
/// `offset` is in -1.0 to 1.0 range (maps to -255 to +255).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![100, 100, 100, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::brightness(&mut buf, 0.5).unwrap();
/// assert!(buf.data[0] > 200);
/// ```
pub fn brightness(buf: &mut PixelBuffer, offset: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let shift = (offset * 255.0) as i16;
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = (pixel[0] as i16 + shift).clamp(0, 255) as u8;
        pixel[1] = (pixel[1] as i16 + shift).clamp(0, 255) as u8;
        pixel[2] = (pixel[2] as i16 + shift).clamp(0, 255) as u8;
    }
    Ok(())
}

/// Adjust contrast of an RGBA8 buffer in-place.
///
/// `factor` of 1.0 is unchanged; >1.0 increases, <1.0 decreases.
/// Contrast is centered around 128.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![128, 128, 128, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::contrast(&mut buf, 2.0).unwrap();
/// // 128 is the center point, so it stays at 128
/// assert_eq!(buf.data[0], 128);
/// ```
pub fn contrast(buf: &mut PixelBuffer, factor: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = ((((pixel[0] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
        pixel[1] = ((((pixel[1] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
        pixel[2] = ((((pixel[2] as f32 - 128.0) * factor) + 128.0).clamp(0.0, 255.0)) as u8;
    }
    Ok(())
}

/// Adjust saturation of an RGBA8 buffer in-place.
///
/// `factor` of 1.0 is unchanged; 0.0 is grayscale; >1.0 increases saturation.
/// Uses BT.601 luminance coefficients.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::saturation(&mut buf, 0.0).unwrap();
/// // At zero saturation, all channels equal (grayscale)
/// assert_eq!(buf.data[0], buf.data[1]);
/// assert_eq!(buf.data[1], buf.data[2]);
/// ```
pub fn saturation(buf: &mut PixelBuffer, factor: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        pixel[0] = (gray + factor * (r - gray)).clamp(0.0, 255.0) as u8;
        pixel[1] = (gray + factor * (g - gray)).clamp(0.0, 255.0) as u8;
        pixel[2] = (gray + factor * (b - gray)).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Apply levels adjustment (black point, white point, gamma).
///
/// All values in 0.0–1.0 range. Gamma of 1.0 is linear. Uses a pre-computed
/// lookup table for performance.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![128, 128, 128, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// // Identity transform: black=0, white=1, gamma=1
/// let original = buf.data.clone();
/// filter::levels(&mut buf, 0.0, 1.0, 1.0).unwrap();
/// assert_eq!(buf.data, original);
/// ```
pub fn levels(buf: &mut PixelBuffer, black: f32, white: f32, gamma: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let range = (white - black).max(1e-6);
    // Build LUT for speed
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = i as f32 / 255.0;
        let mapped = ((v - black) / range).clamp(0.0, 1.0);
        let corrected = mapped.powf(1.0 / gamma);
        *entry = (corrected * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    Ok(())
}

/// Apply a curves adjustment using a 256-entry lookup table.
///
/// Each input byte value is mapped through `lut` to produce the output.
/// An identity LUT (`lut[i] = i`) leaves the image unchanged.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// // Invert via curves
/// let mut lut = [0u8; 256];
/// for i in 0..256 {
///     lut[i] = (255 - i) as u8;
/// }
/// let mut buf = PixelBuffer::new(vec![100, 150, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::curves(&mut buf, &lut).unwrap();
/// assert_eq!(buf.data[0], 155);
/// ```
pub fn curves(buf: &mut PixelBuffer, lut: &[u8; 256]) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = lut[pixel[0] as usize];
        pixel[1] = lut[pixel[1] as usize];
        pixel[2] = lut[pixel[2] as usize];
    }
    Ok(())
}

/// Convert an RGBA8 buffer to grayscale in-place (BT.601 luminance).
///
/// Sets R, G, B channels to the luminance value while preserving alpha.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![200, 100, 50, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::grayscale(&mut buf).unwrap();
/// assert_eq!(buf.data[0], buf.data[1]); // all channels equal
/// assert_eq!(buf.data[1], buf.data[2]);
/// assert_eq!(buf.data[3], 255); // alpha preserved
/// ```
pub fn grayscale(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let gray =
            ((77 * pixel[0] as u16 + 150 * pixel[1] as u16 + 29 * pixel[2] as u16) >> 8) as u8;
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }
    Ok(())
}

/// Invert all color channels in-place.
///
/// Each RGB channel is replaced with `255 - value`. Alpha is preserved.
/// Applying invert twice restores the original image.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![100, 150, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::invert(&mut buf).unwrap();
/// assert_eq!(buf.data[0], 155);
/// assert_eq!(buf.data[1], 105);
/// assert_eq!(buf.data[2], 55);
/// ```
pub fn invert(buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        pixel[0] = 255 - pixel[0];
        pixel[1] = 255 - pixel[1];
        pixel[2] = 255 - pixel[2];
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Spatial filters (blur, sharpen)
// ---------------------------------------------------------------------------

fn build_gaussian_kernel(radius: u32) -> Vec<f32> {
    let r = radius as i32;
    let sigma = (radius as f32 / 3.0).max(0.5);
    let len = (2 * r + 1) as usize;
    let mut kernel = vec![0.0f32; len];
    let mut sum = 0.0;
    for i in 0..len as i32 {
        let x = (i - r) as f32;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel[i as usize] = v;
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

fn blur_pass_horizontal(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    _h: usize,
    kernel: &[f32],
    radius: i32,
) {
    let process_row = |y: usize, row: &mut [u8]| {
        for x in 0..w {
            let mut rv = 0.0f32;
            let mut gv = 0.0f32;
            let mut bv = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - radius).clamp(0, w as i32 - 1) as usize;
                let idx = (y * w + sx) * 4;
                rv += src[idx] as f32 * kv;
                gv += src[idx + 1] as f32 * kv;
                bv += src[idx + 2] as f32 * kv;
            }
            let oi = x * 4;
            row[oi] = rv.clamp(0.0, 255.0) as u8;
            row[oi + 1] = gv.clamp(0.0, 255.0) as u8;
            row[oi + 2] = bv.clamp(0.0, 255.0) as u8;
            row[oi + 3] = src[(y * w + x) * 4 + 3];
        }
    };

    let row_bytes = w * 4;
    #[cfg(feature = "parallel")]
    {
        dst.par_chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        dst.chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    }
}

fn blur_pass_vertical(src: &[u8], dst: &mut [u8], w: usize, h: usize, kernel: &[f32], radius: i32) {
    let process_row = |y: usize, row: &mut [u8]| {
        for x in 0..w {
            let mut rv = 0.0f32;
            let mut gv = 0.0f32;
            let mut bv = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - radius).clamp(0, h as i32 - 1) as usize;
                let idx = (sy * w + x) * 4;
                rv += src[idx] as f32 * kv;
                gv += src[idx + 1] as f32 * kv;
                bv += src[idx + 2] as f32 * kv;
            }
            let oi = x * 4;
            row[oi] = rv.clamp(0.0, 255.0) as u8;
            row[oi + 1] = gv.clamp(0.0, 255.0) as u8;
            row[oi + 2] = bv.clamp(0.0, 255.0) as u8;
            row[oi + 3] = src[(y * w + x) * 4 + 3];
        }
    };

    let row_bytes = w * 4;
    #[cfg(feature = "parallel")]
    {
        dst.par_chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        dst.chunks_exact_mut(row_bytes)
            .enumerate()
            .for_each(|(y, row)| process_row(y, row));
    }
}

/// Apply Gaussian blur with the given pixel radius (separable, two-pass).
///
/// Returns a new blurred buffer. The kernel sigma is `radius / 3`.
/// Replaces rasa's `gaussian_blur` with an identical separable algorithm.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let blurred = filter::gaussian_blur(&buf, 2).unwrap();
/// assert_eq!(blurred.width, 8);
/// ```
pub fn gaussian_blur(buf: &PixelBuffer, radius: u32) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    if radius == 0 {
        return Ok(buf.clone());
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let kernel = build_gaussian_kernel(radius);
    let r = radius as i32;

    let mut temp = vec![0u8; buf.data.len()];
    blur_pass_horizontal(&buf.data, &mut temp, w, h, &kernel, r);

    let mut out = vec![0u8; buf.data.len()];
    blur_pass_vertical(&temp, &mut out, w, h, &kernel, r);

    PixelBuffer::new(out, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Apply box blur with the given pixel radius (separable, two-pass).
///
/// Faster than Gaussian blur — uses a uniform kernel (all weights equal).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let blurred = filter::box_blur(&buf, 2).unwrap();
/// assert_eq!(blurred.width, 8);
/// ```
pub fn box_blur(buf: &PixelBuffer, radius: u32) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    if radius == 0 {
        return Ok(buf.clone());
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let len = 2 * radius as usize + 1;
    let kernel = vec![1.0 / len as f32; len];
    let r = radius as i32;

    let mut temp = vec![0u8; buf.data.len()];
    blur_pass_horizontal(&buf.data, &mut temp, w, h, &kernel, r);

    let mut out = vec![0u8; buf.data.len()];
    blur_pass_vertical(&temp, &mut out, w, h, &kernel, r);

    PixelBuffer::new(out, buf.width, buf.height, PixelFormat::Rgba8)
}

/// Apply unsharp mask (sharpen): `output = original + amount * (original - blurred)`.
///
/// `radius` controls the blur kernel size, `amount` controls the sharpening
/// strength (typically 0.5–2.0).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// let sharp = filter::unsharp_mask(&buf, 2, 1.0).unwrap();
/// assert_eq!(sharp.width, 8);
/// ```
pub fn unsharp_mask(
    buf: &PixelBuffer,
    radius: u32,
    amount: f32,
) -> Result<PixelBuffer, RangaError> {
    let blurred = gaussian_blur(buf, radius)?;
    let mut out = buf.data.clone();
    for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
        let bi = i * 4;
        for (c, ch) in chunk.iter_mut().enumerate().take(3) {
            let orig = buf.data[bi + c] as f32;
            let blur = blurred.data[bi + c] as f32;
            *ch = (orig + amount * (orig - blur)).clamp(0.0, 255.0) as u8;
        }
    }
    PixelBuffer::new(out, buf.width, buf.height, PixelFormat::Rgba8)
}

// ---------------------------------------------------------------------------
// Color filters
// ---------------------------------------------------------------------------

/// Shift hue of all pixels by `degrees` (in HSL space).
///
/// Positive values rotate clockwise, negative counter-clockwise.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::hue_shift(&mut buf, 120.0).unwrap();
/// // Red shifted by 120° → approximately green
/// assert!(buf.data[1] > buf.data[0]); // G > R
/// ```
pub fn hue_shift(buf: &mut PixelBuffer, degrees: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let srgb = crate::color::Srgba {
            r: pixel[0],
            g: pixel[1],
            b: pixel[2],
            a: pixel[3],
        };
        let mut hsl: crate::color::Hsl = srgb.into();
        hsl.h = (hsl.h + degrees).rem_euclid(360.0);
        let back: crate::color::Srgba = hsl.into();
        pixel[0] = back.r;
        pixel[1] = back.g;
        pixel[2] = back.b;
    }
    Ok(())
}

/// Adjust color balance for shadows, midtones, and highlights.
///
/// Each parameter is `[R, G, B]` offset in -1.0 to 1.0 range.
/// Shadows affect dark pixels, midtones affect mid-range, highlights affect
/// bright pixels, using smooth luminance-based weighting.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![128, 128, 128, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// filter::color_balance(
///     &mut buf,
///     [0.0, 0.0, 0.0],   // shadows: neutral
///     [0.1, 0.0, -0.1],   // midtones: warm shift
///     [0.0, 0.0, 0.0],   // highlights: neutral
/// ).unwrap();
/// assert!(buf.data[0] > 128); // red increased in midtones
/// ```
pub fn color_balance(
    buf: &mut PixelBuffer,
    shadows: [f32; 3],
    midtones: [f32; 3],
    highlights: [f32; 3],
) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let lum =
            (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 255.0;
        // Smooth weighting: shadows peak at lum=0, midtones at lum=0.5, highlights at lum=1
        let sw = (1.0 - lum * 2.0).clamp(0.0, 1.0);
        let hw = ((lum - 0.5) * 2.0).clamp(0.0, 1.0);
        let mw = 1.0 - sw - hw;
        for c in 0..3 {
            let adj = shadows[c] * sw + midtones[c] * mw + highlights[c] * hw;
            pixel[c] = ((pixel[c] as f32 + adj * 255.0).clamp(0.0, 255.0)) as u8;
        }
    }
    Ok(())
}

/// Apply a vignette effect — darken edges relative to the center.
///
/// `strength` controls how much darkening is applied (0.0 = none, 1.0 = full).
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![200; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// filter::vignette(&mut buf, 0.5).unwrap();
/// // Center pixel should be brighter than corner
/// let center = buf.data[(4 * 8 + 4) * 4] as u16;
/// let corner = buf.data[0] as u16;
/// assert!(center > corner);
/// ```
pub fn vignette(buf: &mut PixelBuffer, strength: f32) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let w = buf.width as f32;
    let h = buf.height as f32;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let max_dist = (cx * cx + cy * cy).sqrt();
    for y in 0..buf.height {
        for x in 0..buf.width {
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let dist = (dx * dx + dy * dy).sqrt() / max_dist;
            let factor = 1.0 - strength * dist * dist;
            let i = (y as usize * buf.width as usize + x as usize) * 4;
            buf.data[i] = (buf.data[i] as f32 * factor).clamp(0.0, 255.0) as u8;
            buf.data[i + 1] = (buf.data[i + 1] as f32 * factor).clamp(0.0, 255.0) as u8;
            buf.data[i + 2] = (buf.data[i + 2] as f32 * factor).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 3D LUT
// ---------------------------------------------------------------------------

/// A 3D color lookup table for color grading.
///
/// Loaded from `.cube` files (the standard format used by DaVinci Resolve,
/// Adobe, etc.). Trilinear interpolation is used for smooth color mapping.
///
/// # Examples
///
/// ```
/// use ranga::filter::Lut3d;
///
/// let cube = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
/// let lut = Lut3d::from_cube(cube).unwrap();
/// assert_eq!(lut.size, 2);
/// ```
#[derive(Debug, Clone)]
pub struct Lut3d {
    /// LUT dimension (size x size x size entries).
    pub size: usize,
    /// RGB triplets in R-fastest order.
    pub data: Vec<[f32; 3]>,
}

impl Lut3d {
    /// Parse a `.cube` file into a 3D LUT.
    pub fn from_cube(text: &str) -> Result<Self, RangaError> {
        let mut size = 0usize;
        let mut data = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("TITLE") {
                continue;
            }
            if trimmed.starts_with("DOMAIN_MIN") || trimmed.starts_with("DOMAIN_MAX") {
                continue;
            }
            if let Some(s) = trimmed.strip_prefix("LUT_3D_SIZE") {
                size = s
                    .trim()
                    .parse::<usize>()
                    .map_err(|e| RangaError::Other(format!("bad LUT size: {e}")))?;
                continue;
            }
            if trimmed.starts_with("LUT_1D_SIZE") {
                return Err(RangaError::Other("1D LUTs not supported".into()));
            }
            // Data line: R G B
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                let r: f32 = parts[0]
                    .parse()
                    .map_err(|e| RangaError::Other(format!("bad LUT value: {e}")))?;
                let g: f32 = parts[1]
                    .parse()
                    .map_err(|e| RangaError::Other(format!("bad LUT value: {e}")))?;
                let b: f32 = parts[2]
                    .parse()
                    .map_err(|e| RangaError::Other(format!("bad LUT value: {e}")))?;
                data.push([r, g, b]);
            }
        }
        if size < 2 {
            return Err(RangaError::Other("LUT_3D_SIZE must be >= 2".into()));
        }
        let expected = size * size * size;
        if data.len() != expected {
            return Err(RangaError::Other(format!(
                "expected {} LUT entries, got {}",
                expected,
                data.len()
            )));
        }
        Ok(Self { size, data })
    }

    /// Trilinear interpolation lookup.
    #[must_use]
    pub fn lookup(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = (self.size - 1) as f32;
        let ri = (r * n).clamp(0.0, n);
        let gi = (g * n).clamp(0.0, n);
        let bi = (b * n).clamp(0.0, n);

        let r0 = ri.floor() as usize;
        let g0 = gi.floor() as usize;
        let b0 = bi.floor() as usize;
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = ri - r0 as f32;
        let fg = gi - g0 as f32;
        let fb = bi - b0 as f32;

        let idx = |r: usize, g: usize, b: usize| r + g * self.size + b * self.size * self.size;

        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        // 8-corner trilinear interpolation
        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        let c00 = lerp(c000, c100, fr);
        let c10 = lerp(c010, c110, fr);
        let c01 = lerp(c001, c101, fr);
        let c11 = lerp(c011, c111, fr);
        let c0 = lerp(c00, c10, fg);
        let c1 = lerp(c01, c11, fg);
        lerp(c0, c1, fb)
    }
}

/// Apply a 3D LUT to an RGBA8 buffer in-place.
///
/// Uses trilinear interpolation, matching tazama's LUT pipeline.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter::{Lut3d, apply_lut3d};
///
/// // Identity LUT (size 2)
/// let cube = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
/// let lut = Lut3d::from_cube(cube).unwrap();
/// let mut buf = PixelBuffer::new(vec![128, 64, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// apply_lut3d(&mut buf, &lut).unwrap();
/// ```
pub fn apply_lut3d(buf: &mut PixelBuffer, lut: &Lut3d) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    for pixel in buf.data.chunks_exact_mut(4) {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        let [lr, lg, lb] = lut.lookup(r, g, b);
        pixel[0] = (lr * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        pixel[1] = (lg * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        pixel[2] = (lb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for noise generation.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x1234_5678_9ABC_DEF0
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Approximate Gaussian via Box-Muller-like: sum of 6 uniform - 3.
    fn next_gaussian(&mut self) -> f32 {
        let mut sum = 0.0f32;
        for _ in 0..6 {
            sum += (self.next_u64() & 0xFFFF) as f32 / 65535.0;
        }
        sum - 3.0 // approximate N(0,1) range roughly -3..3
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFF_FFFF) as f32 / 4_294_967_295.0
    }
}

/// Add Gaussian noise to an RGBA8 buffer in-place.
///
/// `amount` controls the noise standard deviation (0.0–1.0 maps to 0–255).
/// `seed` provides a deterministic PRNG seed for reproducible results.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap();
/// filter::noise_gaussian(&mut buf, 0.1, 42).unwrap();
/// // Some pixels should have changed
/// assert!(buf.data.iter().any(|&v| v != 128));
/// ```
pub fn noise_gaussian(buf: &mut PixelBuffer, amount: f32, seed: u64) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let scale = amount * 255.0;
    let mut rng = Xorshift64::new(seed);
    for pixel in buf.data.chunks_exact_mut(4) {
        for ch in pixel.iter_mut().take(3) {
            let noise = rng.next_gaussian() * scale;
            *ch = (*ch as f32 + noise).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(())
}

/// Add salt-and-pepper noise to an RGBA8 buffer in-place.
///
/// `density` is the probability that any pixel becomes salt (white) or pepper
/// (black), in 0.0–1.0 range. `seed` for reproducibility.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::filter;
///
/// let mut buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
/// filter::noise_salt_pepper(&mut buf, 0.1, 42).unwrap();
/// ```
pub fn noise_salt_pepper(buf: &mut PixelBuffer, density: f32, seed: u64) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat(format!("{:?}", buf.format)));
    }
    let mut rng = Xorshift64::new(seed);
    for pixel in buf.data.chunks_exact_mut(4) {
        let r = rng.next_f32();
        if r < density / 2.0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else if r < density {
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buf() -> PixelBuffer {
        PixelBuffer::new(vec![128; 4 * 4 * 4], 4, 4, PixelFormat::Rgba8).unwrap()
    }

    #[test]
    fn brightness_positive() {
        let mut buf = test_buf();
        brightness(&mut buf, 0.5).unwrap();
        assert!(buf.data[0] > 200);
    }

    #[test]
    fn brightness_negative() {
        let mut buf = test_buf();
        brightness(&mut buf, -0.5).unwrap();
        assert!(buf.data[0] < 10);
    }

    #[test]
    fn contrast_increase() {
        let mut buf = test_buf();
        contrast(&mut buf, 2.0).unwrap();
        // 128 is center → stays at 128
        assert_eq!(buf.data[0], 128);
    }

    #[test]
    fn saturation_zero_is_gray() {
        let mut buf = PixelBuffer::new(
            vec![255, 0, 0, 255, 0, 255, 0, 255],
            2,
            1,
            PixelFormat::Rgba8,
        )
        .unwrap();
        saturation(&mut buf, 0.0).unwrap();
        // All channels should be equal (gray)
        assert_eq!(buf.data[0], buf.data[1]);
        assert_eq!(buf.data[1], buf.data[2]);
    }

    #[test]
    fn grayscale_makes_uniform() {
        let mut buf = PixelBuffer::new(vec![200, 100, 50, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        grayscale(&mut buf).unwrap();
        assert_eq!(buf.data[0], buf.data[1]);
        assert_eq!(buf.data[1], buf.data[2]);
        assert_eq!(buf.data[3], 255); // alpha unchanged
    }

    #[test]
    fn invert_roundtrip() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        invert(&mut buf).unwrap();
        invert(&mut buf).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn levels_identity() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        levels(&mut buf, 0.0, 1.0, 1.0).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn curves_identity() {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        let mut buf = test_buf();
        let original = buf.data.clone();
        curves(&mut buf, &lut).unwrap();
        assert_eq!(buf.data, original);
    }

    // --- New filter tests ---

    #[test]
    fn gaussian_blur_uniform_unchanged() {
        // Uniform buffer should be (nearly) unchanged by blur
        let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let blurred = gaussian_blur(&buf, 2).unwrap();
        assert_eq!(blurred.data[0], 128);
    }

    #[test]
    fn gaussian_blur_radius_zero_is_identity() {
        let buf = test_buf();
        let blurred = gaussian_blur(&buf, 0).unwrap();
        assert_eq!(blurred.data, buf.data);
    }

    #[test]
    fn box_blur_uniform_unchanged() {
        let buf = PixelBuffer::new(vec![100; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let blurred = box_blur(&buf, 3).unwrap();
        assert_eq!(blurred.data[0], 100);
    }

    #[test]
    fn unsharp_mask_zero_amount_is_identity() {
        let buf = test_buf();
        let sharp = unsharp_mask(&buf, 2, 0.0).unwrap();
        assert_eq!(sharp.data, buf.data);
    }

    #[test]
    fn hue_shift_360_is_identity() {
        let mut buf = PixelBuffer::new(vec![200, 100, 50, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let original = buf.data.clone();
        hue_shift(&mut buf, 360.0).unwrap();
        for (i, (&got, &exp)) in buf.data.iter().zip(original.iter()).enumerate().take(3) {
            assert!((got as i16 - exp as i16).unsigned_abs() <= 1, "channel {i}");
        }
    }

    #[test]
    fn hue_shift_red_to_green() {
        let mut buf = PixelBuffer::new(vec![255, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        hue_shift(&mut buf, 120.0).unwrap();
        assert!(
            buf.data[1] > buf.data[0],
            "green should dominate after +120°"
        );
    }

    #[test]
    fn color_balance_neutral_is_identity() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        color_balance(&mut buf, [0.0; 3], [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(buf.data, original);
    }

    #[test]
    fn vignette_center_brighter_than_corner() {
        let mut buf = PixelBuffer::new(vec![200; 16 * 16 * 4], 16, 16, PixelFormat::Rgba8).unwrap();
        vignette(&mut buf, 1.0).unwrap();
        let center = buf.data[(8 * 16 + 8) * 4];
        let corner = buf.data[0];
        assert!(center > corner, "center={center} corner={corner}");
    }

    #[test]
    fn lut3d_identity_preserves_values() {
        // Build an identity LUT of size 2
        let cube = "LUT_3D_SIZE 2\n\
            0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n\
            0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let lut = Lut3d::from_cube(cube).unwrap();
        let mut buf = PixelBuffer::new(vec![128, 64, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let original = buf.data.clone();
        apply_lut3d(&mut buf, &lut).unwrap();
        for (i, (&got, &exp)) in buf.data.iter().zip(original.iter()).enumerate().take(3) {
            assert!(
                (got as i16 - exp as i16).unsigned_abs() <= 1,
                "channel {i}: {got} vs {exp}",
            );
        }
    }

    #[test]
    fn lut3d_bad_size_rejected() {
        let cube = "LUT_3D_SIZE 1\n0.0 0.0 0.0\n";
        assert!(Lut3d::from_cube(cube).is_err());
    }

    #[test]
    fn noise_gaussian_modifies_buffer() {
        let mut buf = test_buf();
        let original = buf.data.clone();
        noise_gaussian(&mut buf, 0.5, 42).unwrap();
        assert_ne!(buf.data, original);
    }

    #[test]
    fn noise_gaussian_deterministic() {
        let mut buf1 = test_buf();
        let mut buf2 = test_buf();
        noise_gaussian(&mut buf1, 0.3, 123).unwrap();
        noise_gaussian(&mut buf2, 0.3, 123).unwrap();
        assert_eq!(buf1.data, buf2.data);
    }

    #[test]
    fn noise_salt_pepper_creates_extremes() {
        let mut buf = PixelBuffer::new(vec![128; 32 * 32 * 4], 32, 32, PixelFormat::Rgba8).unwrap();
        noise_salt_pepper(&mut buf, 0.5, 42).unwrap();
        let has_black = buf
            .data
            .chunks_exact(4)
            .any(|p| p[0] == 0 && p[1] == 0 && p[2] == 0);
        let has_white = buf
            .data
            .chunks_exact(4)
            .any(|p| p[0] == 255 && p[1] == 255 && p[2] == 255);
        assert!(has_black, "should have some black pixels");
        assert!(has_white, "should have some white pixels");
    }

    #[test]
    fn noise_salt_pepper_preserves_alpha() {
        let mut buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        noise_salt_pepper(&mut buf, 0.5, 99).unwrap();
        // All alpha values should still be 128
        for pixel in buf.data.chunks_exact(4) {
            assert_eq!(pixel[3], 128);
        }
    }
}
