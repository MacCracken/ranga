//! Pixel buffer type — unified image buffer with format awareness.

use serde::{Deserialize, Serialize};

use crate::RangaError;

/// Supported pixel formats.
///
/// Each variant describes the channel layout and byte size per pixel.
///
/// # Examples
///
/// ```
/// use ranga::pixel::PixelFormat;
///
/// let size = PixelFormat::Rgba8.buffer_size(1920, 1080);
/// assert_eq!(size, 1920 * 1080 * 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormat {
    /// 4 bytes per pixel: R, G, B, A
    Rgba8,
    /// 4 bytes per pixel: A, R, G, B (used by aethersafta)
    Argb8,
    /// 3 bytes per pixel: R, G, B
    Rgb8,
    /// Planar YUV 4:2:0 (Y plane + U plane + V plane)
    Yuv420p,
    /// Semi-planar YUV 4:2:0 (Y plane + interleaved UV plane)
    Nv12,
    /// 4 channels of f32 per pixel (linear color, HDR)
    RgbaF32,
}

impl PixelFormat {
    /// Compute expected buffer size in bytes for this format at the given dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::PixelFormat;
    ///
    /// assert_eq!(PixelFormat::Rgb8.buffer_size(10, 10), 300);
    /// assert_eq!(PixelFormat::Yuv420p.buffer_size(320, 240), 320 * 240 + 2 * 160 * 120);
    /// ```
    #[must_use]
    pub fn buffer_size(self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        match self {
            Self::Rgba8 | Self::Argb8 => w * h * 4,
            Self::Rgb8 => w * h * 3,
            Self::Yuv420p => w * h + 2 * (w / 2) * (h / 2),
            Self::Nv12 => w * h + (w / 2) * (h / 2) * 2,
            Self::RgbaF32 => w * h * 16,
        }
    }
}

/// A pixel buffer holding image data in a known format.
///
/// All ranga operations validate the buffer format before processing,
/// ensuring type-safe pixel access.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// // Create a zeroed 64x64 RGBA buffer
/// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
/// assert_eq!(buf.pixel_count(), 64 * 64);
/// assert_eq!(buf.data.len(), 64 * 64 * 4);
///
/// // Create from existing data
/// let buf = PixelBuffer::new(vec![255; 4], 1, 1, PixelFormat::Rgba8).unwrap();
/// assert_eq!(buf.data[0], 255);
/// ```
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

impl PixelBuffer {
    /// Create a new pixel buffer, validating data length.
    ///
    /// Returns an error if `data.len()` does not match the expected size
    /// for the given format and dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::new(vec![0; 400], 10, 10, PixelFormat::Rgba8).unwrap();
    /// assert_eq!(buf.width, 10);
    ///
    /// // Wrong size is rejected
    /// assert!(PixelBuffer::new(vec![0; 100], 10, 10, PixelFormat::Rgba8).is_err());
    /// ```
    pub fn new(
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Self, RangaError> {
        let expected = format.buffer_size(width, height);
        if data.len() != expected {
            return Err(RangaError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            format,
        })
    }

    /// Create a zero-filled buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);
    /// assert!(buf.data.iter().all(|&b| b == 0));
    /// ```
    #[must_use]
    pub fn zeroed(width: u32, height: u32, format: PixelFormat) -> Self {
        let size = format.buffer_size(width, height);
        Self {
            data: vec![0u8; size],
            width,
            height,
            format,
        }
    }

    /// Number of pixels.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
    /// assert_eq!(buf.pixel_count(), 1920 * 1080);
    /// ```
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_buffer_size() {
        assert_eq!(PixelFormat::Rgba8.buffer_size(1920, 1080), 1920 * 1080 * 4);
    }

    #[test]
    fn yuv420p_buffer_size() {
        assert_eq!(
            PixelFormat::Yuv420p.buffer_size(320, 240),
            320 * 240 + 2 * 160 * 120
        );
    }

    #[test]
    fn new_validates_length() {
        let result = PixelBuffer::new(vec![0; 100], 10, 10, PixelFormat::Rgba8);
        assert!(result.is_err());

        let result = PixelBuffer::new(vec![0; 400], 10, 10, PixelFormat::Rgba8);
        assert!(result.is_ok());
    }

    #[test]
    fn zeroed_creates_correct_size() {
        let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgb8);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }
}
