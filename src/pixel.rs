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
#[non_exhaustive]
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

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rgba8 => write!(f, "RGBA8"),
            Self::Argb8 => write!(f, "ARGB8"),
            Self::Rgb8 => write!(f, "RGB8"),
            Self::Yuv420p => write!(f, "YUV420p"),
            Self::Nv12 => write!(f, "NV12"),
            Self::RgbaF32 => write!(f, "RgbaF32"),
        }
    }
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

    /// Iterate over rows as byte slices.
    ///
    /// Each row is `width * bytes_per_pixel` bytes. Only valid for
    /// non-planar formats (Rgba8, Argb8, Rgb8, RgbaF32).
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::zeroed(4, 3, PixelFormat::Rgba8);
    /// assert_eq!(buf.rows().count(), 3);
    /// assert_eq!(buf.rows().next().unwrap().len(), 16); // 4 pixels * 4 bytes
    /// ```
    pub fn rows(&self) -> impl Iterator<Item = &[u8]> {
        let stride = match self.format {
            PixelFormat::Rgba8 | PixelFormat::Argb8 => self.width as usize * 4,
            PixelFormat::Rgb8 => self.width as usize * 3,
            PixelFormat::RgbaF32 => self.width as usize * 16,
            _ => self.width as usize, // planar: Y-plane row
        };
        self.data.chunks_exact(stride)
    }

    /// Iterate over rows as mutable byte slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let mut buf = PixelBuffer::zeroed(4, 2, PixelFormat::Rgba8);
    /// for row in buf.rows_mut() {
    ///     row[0] = 255; // set first byte of each row
    /// }
    /// assert_eq!(buf.data[0], 255);
    /// assert_eq!(buf.data[16], 255);
    /// ```
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [u8]> {
        let stride = match self.format {
            PixelFormat::Rgba8 | PixelFormat::Argb8 => self.width as usize * 4,
            PixelFormat::Rgb8 => self.width as usize * 3,
            PixelFormat::RgbaF32 => self.width as usize * 16,
            _ => self.width as usize,
        };
        self.data.chunks_exact_mut(stride)
    }

    /// Get the RGBA value of a pixel at (x, y). Requires RGBA8 format.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::new(vec![255, 128, 64, 200], 1, 1, PixelFormat::Rgba8).unwrap();
    /// assert_eq!(buf.get_rgba(0, 0), Some([255, 128, 64, 200]));
    /// assert_eq!(buf.get_rgba(99, 99), None); // out of bounds
    /// ```
    #[must_use]
    pub fn get_rgba(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if self.format != PixelFormat::Rgba8 || x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ])
    }

    /// Set the RGBA value of a pixel at (x, y). Requires RGBA8 format.
    ///
    /// Returns `false` if out of bounds or wrong format.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let mut buf = PixelBuffer::zeroed(2, 2, PixelFormat::Rgba8);
    /// assert!(buf.set_rgba(0, 0, [255, 0, 0, 255]));
    /// assert_eq!(buf.data[0], 255);
    /// assert!(!buf.set_rgba(99, 99, [0; 4])); // out of bounds
    /// ```
    pub fn set_rgba(&mut self, x: u32, y: u32, pixel: [u8; 4]) -> bool {
        if self.format != PixelFormat::Rgba8 || x >= self.width || y >= self.height {
            return false;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.data[i..i + 4].copy_from_slice(&pixel);
        true
    }

    /// Create an owned `PixelBuffer` from a `PixelView` (copies data).
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelView, PixelFormat};
    ///
    /// let data = vec![128u8; 4 * 4 * 4];
    /// let view = PixelView::new(&data, 4, 4, PixelFormat::Rgba8).unwrap();
    /// let buf = PixelBuffer::from_view(&view);
    /// assert_eq!(buf.data, data);
    /// ```
    #[must_use]
    pub fn from_view(view: &PixelView<'_>) -> Self {
        Self {
            data: view.data().to_vec(),
            width: view.width(),
            height: view.height(),
            format: view.format(),
        }
    }

    /// Borrow this buffer as a read-only [`PixelView`].
    ///
    /// Zero-copy — no allocation. Useful for passing existing buffer data
    /// to functions without cloning.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let buf = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);
    /// let view = buf.as_view();
    /// assert_eq!(view.width(), 8);
    /// ```
    #[must_use]
    pub fn as_view(&self) -> PixelView<'_> {
        PixelView {
            data: &self.data,
            width: self.width,
            height: self.height,
            format: self.format,
        }
    }

    /// Borrow this buffer as a mutable [`PixelViewMut`].
    ///
    /// Zero-copy — allows in-place modification through a borrowed view.
    ///
    /// # Examples
    ///
    /// ```
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let mut buf = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);
    /// let mut view = buf.as_view_mut();
    /// view.data_mut()[0] = 255;
    /// assert_eq!(buf.data[0], 255);
    /// ```
    pub fn as_view_mut(&mut self) -> PixelViewMut<'_> {
        PixelViewMut {
            data: &mut self.data,
            width: self.width,
            height: self.height,
            format: self.format,
        }
    }
}

/// A read-only borrowed view over pixel data — zero-copy.
///
/// Created from [`PixelBuffer::as_view`] or directly from a byte slice.
/// Allows rasa/tazama/aethersafta to pass their existing buffers to ranga
/// without copying.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelView, PixelFormat};
///
/// let data = vec![128u8; 4 * 4 * 4];
/// let view = PixelView::new(&data, 4, 4, PixelFormat::Rgba8).unwrap();
/// assert_eq!(view.pixel_count(), 16);
/// ```
#[derive(Debug)]
pub struct PixelView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    format: PixelFormat,
}

impl<'a> PixelView<'a> {
    /// Create a view from a byte slice, validating length.
    pub fn new(
        data: &'a [u8],
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

    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// A mutable borrowed view over pixel data — zero-copy.
///
/// Created from [`PixelBuffer::as_view_mut`] or directly from a mutable byte slice.
///
/// # Examples
///
/// ```
/// use ranga::pixel::{PixelViewMut, PixelFormat};
///
/// let mut data = vec![0u8; 4 * 4 * 4];
/// let mut view = PixelViewMut::new(&mut data, 4, 4, PixelFormat::Rgba8).unwrap();
/// view.data_mut()[0] = 255;
/// assert_eq!(data[0], 255);
/// ```
#[derive(Debug)]
pub struct PixelViewMut<'a> {
    data: &'a mut [u8],
    width: u32,
    height: u32,
    format: PixelFormat,
}

impl<'a> PixelViewMut<'a> {
    /// Create a mutable view from a byte slice, validating length.
    pub fn new(
        data: &'a mut [u8],
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

    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[must_use]
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// A reusable buffer pool for reducing allocation overhead in pipelines.
///
/// Useful for video editors (tazama) and compositors (aethersafta) that
/// process many frames with the same dimensions.
///
/// # Examples
///
/// ```
/// use ranga::pixel::BufferPool;
///
/// let mut pool = BufferPool::new(4);
/// let buf = pool.acquire(1920 * 1080 * 4);
/// assert_eq!(buf.len(), 1920 * 1080 * 4);
/// pool.release(buf); // returns to pool for reuse
/// let buf2 = pool.acquire(1920 * 1080 * 4); // reused, no allocation
/// ```
#[derive(Debug)]
pub struct BufferPool {
    pool: Vec<Vec<u8>>,
    max_buffers: usize,
}

impl BufferPool {
    /// Create a new buffer pool with the given maximum retained buffers.
    #[must_use]
    pub fn new(max_buffers: usize) -> Self {
        Self {
            pool: Vec::new(),
            max_buffers,
        }
    }

    /// Acquire a buffer of at least `size` bytes.
    ///
    /// Reuses a pooled buffer if one of sufficient size exists, otherwise
    /// allocates a new one. The returned buffer is zero-filled.
    pub fn acquire(&mut self, size: usize) -> Vec<u8> {
        // Find the smallest buffer that fits
        if let Some(pos) = self.pool.iter().position(|b| b.capacity() >= size) {
            let mut buf = self.pool.swap_remove(pos);
            buf.clear();
            buf.resize(size, 0);
            buf
        } else {
            vec![0u8; size]
        }
    }

    /// Return a buffer to the pool for future reuse.
    pub fn release(&mut self, buf: Vec<u8>) {
        if self.pool.len() < self.max_buffers {
            self.pool.push(buf);
        }
        // Otherwise drop it
    }

    /// Number of buffers currently in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Clear all pooled buffers, freeing memory.
    pub fn clear(&mut self) {
        self.pool.clear();
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

    #[test]
    fn pixel_view_from_buffer() {
        let buf = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);
        let view = buf.as_view();
        assert_eq!(view.width(), 8);
        assert_eq!(view.pixel_count(), 64);
    }

    #[test]
    fn pixel_view_from_slice() {
        let data = vec![0u8; 4 * 4 * 4];
        let view = PixelView::new(&data, 4, 4, PixelFormat::Rgba8).unwrap();
        assert_eq!(view.data().len(), 64);
    }

    #[test]
    fn pixel_view_mut_modifies_original() {
        let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        {
            let mut view = buf.as_view_mut();
            view.data_mut()[0] = 42;
        }
        assert_eq!(buf.data[0], 42);
    }

    #[test]
    fn buffer_pool_reuse() {
        let mut pool = BufferPool::new(4);
        let buf = pool.acquire(1024);
        assert_eq!(buf.len(), 1024);
        pool.release(buf);
        assert_eq!(pool.len(), 1);
        let buf2 = pool.acquire(512); // reuses the 1024-cap buffer
        assert_eq!(buf2.len(), 512);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn buffer_pool_max_limit() {
        let mut pool = BufferPool::new(2);
        pool.release(vec![0; 100]);
        pool.release(vec![0; 200]);
        pool.release(vec![0; 300]); // exceeds max, dropped
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn buffer_pool_zero_filled() {
        let mut pool = BufferPool::new(4);
        let mut buf = pool.acquire(16);
        buf.iter_mut().for_each(|b| *b = 0xFF);
        pool.release(buf);
        let buf2 = pool.acquire(16);
        assert!(
            buf2.iter().all(|&b| b == 0),
            "reused buffer should be zeroed"
        );
    }
}
