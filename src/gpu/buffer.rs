//! GPU buffer wrapper for pixel data upload/download.
//!
//! [`GpuBuffer`] manages a wgpu storage buffer that holds pixel data on the GPU.
//! It provides upload and download helpers that work directly with [`PixelBuffer`].

use super::context::{GpuContext, GpuError};
use crate::pixel::PixelBuffer;

/// A GPU-side pixel buffer for compute operations.
///
/// Wraps a [`wgpu::Buffer`] with upload/download helpers for [`PixelBuffer`].
/// The buffer is created with storage, copy-src, and copy-dst usage flags
/// so it can be used in compute shaders and read back to the CPU.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, GpuBuffer};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let buf = PixelBuffer::zeroed(256, 256, PixelFormat::Rgba8);
/// let gpu_buf = GpuBuffer::upload(&ctx, &buf);
/// let result = gpu_buf.download(&ctx).unwrap();
/// assert_eq!(result.width, 256);
/// ```
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    width: u32,
    height: u32,
}

impl GpuBuffer {
    /// Upload a [`PixelBuffer`] to GPU storage.
    ///
    /// Creates a new GPU buffer initialized with the pixel data. The buffer
    /// is usable as both a storage buffer (for compute shaders) and as a
    /// copy source/destination (for readback).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuBuffer};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let gpu_buf = GpuBuffer::upload(&ctx, &buf);
    /// assert_eq!(gpu_buf.width(), 64);
    /// ```
    #[must_use]
    pub fn upload(ctx: &GpuContext, buf: &PixelBuffer) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ranga_pixel_buf"),
                contents: &buf.data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            size: buf.data.len() as u64,
            width: buf.width,
            height: buf.height,
            buffer,
        }
    }

    /// Download GPU buffer back to a [`PixelBuffer`].
    ///
    /// Creates a staging buffer, copies the GPU data into it, maps it for
    /// reading, and constructs a new [`PixelBuffer`] with the result.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BufferOp`] if the buffer mapping or copy fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuBuffer};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(32, 32, PixelFormat::Rgba8);
    /// let gpu_buf = GpuBuffer::upload(&ctx, &buf);
    /// let result = gpu_buf.download(&ctx).unwrap();
    /// assert_eq!(result.data.len(), 32 * 32 * 4);
    /// ```
    pub fn download(&self, ctx: &GpuContext) -> Result<PixelBuffer, GpuError> {
        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: self.size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        ctx.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        ctx.device().poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| GpuError::BufferOp(e.to_string()))?
            .map_err(|e| GpuError::BufferOp(e.to_string()))?;

        let data = slice.get_mapped_range().to_vec();
        drop(staging);

        PixelBuffer::new(
            data,
            self.width,
            self.height,
            crate::pixel::PixelFormat::Rgba8,
        )
        .map_err(|e| GpuError::BufferOp(e.to_string()))
    }

    /// Access the underlying [`wgpu::Buffer`].
    #[must_use]
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Size of the buffer in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Width of the pixel buffer in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the pixel buffer in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}
