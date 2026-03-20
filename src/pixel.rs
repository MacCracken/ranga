//! Pixel buffer type — unified image buffer with format awareness.

use serde::{Deserialize, Serialize};

use crate::RangaError;

/// Supported pixel formats.
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
#[derive(Debug, Clone)]
pub struct PixelBuffer {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}

impl PixelBuffer {
    /// Create a new pixel buffer, validating data length.
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: PixelFormat) -> Result<Self, RangaError> {
        let expected = format.buffer_size(width, height);
        if data.len() != expected {
            return Err(RangaError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, width, height, format })
    }

    /// Create a zero-filled buffer.
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
