//! High-level GPU compute operations.
//!
//! Each function wraps the low-level wgpu pipeline setup, shader dispatch,
//! and buffer readback into a simple call that operates on [`PixelBuffer`]s.
//! All operations require RGBA8 format.

use super::buffer::GpuBuffer;
use super::context::GpuContext;
use super::shaders;
use crate::RangaError;
use crate::blend::BlendMode;
use crate::pixel::{PixelBuffer, PixelFormat};

/// GPU-accelerated blend of `src` over `dst` using the specified blend mode.
///
/// Both buffers must be RGBA8 with identical dimensions. The `dst` buffer is
/// modified in-place with the blended result.
///
/// # Errors
///
/// Returns an error if buffers are not RGBA8, dimensions do not match, or
/// a GPU operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_blend};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
/// use ranga::blend::BlendMode;
///
/// let ctx = GpuContext::new().unwrap();
/// let src = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
/// let mut dst = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
/// gpu_blend(&ctx, &src, &mut dst, BlendMode::Normal, 1.0).unwrap();
/// ```
pub fn gpu_blend(
    ctx: &GpuContext,
    src: &PixelBuffer,
    dst: &mut PixelBuffer,
    mode: BlendMode,
    opacity: f32,
) -> Result<(), RangaError> {
    if src.format != PixelFormat::Rgba8 || dst.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU blend requires RGBA8".into()));
    }
    if src.width != dst.width || src.height != dst.height {
        return Err(RangaError::DimensionMismatch {
            expected: src.pixel_count(),
            actual: dst.pixel_count(),
        });
    }

    let pixel_count = u32::try_from(src.pixel_count()).map_err(|_| {
        RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
    })?;
    let mode_id: u32 = match mode {
        BlendMode::Normal => 0,
        BlendMode::Multiply => 1,
        BlendMode::Screen => 2,
        BlendMode::Overlay => 3,
        BlendMode::Darken => 4,
        BlendMode::Lighten => 5,
        BlendMode::ColorDodge => 6,
        BlendMode::ColorBurn => 7,
        BlendMode::SoftLight => 8,
        BlendMode::HardLight => 9,
        BlendMode::Difference => 10,
        BlendMode::Exclusion => 11,
    };

    let src_gpu = GpuBuffer::upload(ctx, src);
    let dst_gpu = GpuBuffer::upload(ctx, dst);

    // Params: count, mode, opacity, padding — matches shader Params struct
    let params = [pixel_count, mode_id, opacity.to_bits(), 0u32];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };

    dispatch_3buf_shader(
        ctx,
        "blend_all",
        &shaders::build_shader(shaders::BLEND_ALL),
        src_gpu.wgpu_buffer(),
        dst_gpu.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );

    let result = dst_gpu.download(ctx)?;
    dst.data = result.data;
    Ok(())
}

/// GPU-accelerated color inversion.
///
/// Inverts R, G, B channels (`1.0 - value`) while preserving alpha.
/// The buffer is modified in-place.
///
/// # Errors
///
/// Returns an error if the buffer is not RGBA8 or a GPU operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_invert};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let mut buf = PixelBuffer::new(vec![100, 150, 200, 255], 1, 1, PixelFormat::Rgba8).unwrap();
/// gpu_invert(&ctx, &mut buf).unwrap();
/// ```
pub fn gpu_invert(ctx: &GpuContext, buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU requires RGBA8".into()));
    }
    let pixel_count = u32::try_from(buf.pixel_count()).map_err(|_| {
        RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
    })?;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
        "invert",
        &shaders::build_shader(shaders::INVERT),
        gpu_buf.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );
    let result = gpu_buf.download(ctx)?;
    buf.data = result.data;
    Ok(())
}

/// GPU-accelerated grayscale conversion (BT.709 luminance).
///
/// Sets R, G, B channels to the BT.709 luminance value while preserving alpha.
/// The buffer is modified in-place.
///
/// # Errors
///
/// Returns an error if the buffer is not RGBA8 or a GPU operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_grayscale};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let mut buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
/// gpu_grayscale(&ctx, &mut buf).unwrap();
/// ```
pub fn gpu_grayscale(ctx: &GpuContext, buf: &mut PixelBuffer) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU requires RGBA8".into()));
    }
    let pixel_count = u32::try_from(buf.pixel_count()).map_err(|_| {
        RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
    })?;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
        "grayscale",
        &shaders::build_shader(shaders::GRAYSCALE),
        gpu_buf.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );
    let result = gpu_buf.download(ctx)?;
    buf.data = result.data;
    Ok(())
}

/// GPU-accelerated brightness and contrast adjustment.
///
/// `brightness` is an offset in the -1.0 to 1.0 range. `contrast` is a
/// multiplier where 1.0 is unchanged. The buffer is modified in-place.
///
/// # Errors
///
/// Returns an error if the buffer is not RGBA8 or a GPU operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_brightness_contrast};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let mut buf = PixelBuffer::zeroed(128, 128, PixelFormat::Rgba8);
/// gpu_brightness_contrast(&ctx, &mut buf, 0.1, 1.5).unwrap();
/// ```
pub fn gpu_brightness_contrast(
    ctx: &GpuContext,
    buf: &mut PixelBuffer,
    brightness: f32,
    contrast: f32,
) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU requires RGBA8".into()));
    }
    let pixel_count = u32::try_from(buf.pixel_count()).map_err(|_| {
        RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
    })?;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count, brightness.to_bits(), contrast.to_bits(), 0u32];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
        "brightness_contrast",
        &shaders::build_shader(shaders::BRIGHTNESS_CONTRAST),
        gpu_buf.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );
    let result = gpu_buf.download(ctx)?;
    buf.data = result.data;
    Ok(())
}

/// GPU-accelerated saturation adjustment.
///
/// `factor` of 1.0 is unchanged; 0.0 is grayscale; >1.0 increases saturation.
/// Uses BT.601 luminance coefficients. The buffer is modified in-place.
///
/// # Errors
///
/// Returns an error if the buffer is not RGBA8 or a GPU operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_saturation};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let mut buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
/// gpu_saturation(&ctx, &mut buf, 0.5).unwrap();
/// ```
pub fn gpu_saturation(
    ctx: &GpuContext,
    buf: &mut PixelBuffer,
    factor: f32,
) -> Result<(), RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU requires RGBA8".into()));
    }
    let pixel_count = u32::try_from(buf.pixel_count()).map_err(|_| {
        RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
    })?;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count, factor.to_bits(), 0u32, 0u32];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
        "saturation",
        &shaders::build_shader(shaders::SATURATION),
        gpu_buf.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );
    let result = gpu_buf.download(ctx)?;
    buf.data = result.data;
    Ok(())
}

// ── GPU chain builder ──────────────────────────────────────────────────────

/// A chain of GPU operations executed without CPU readback between steps.
///
/// Upload once, apply multiple operations on the GPU, download once at the end.
/// This avoids the overhead of CPU-to-GPU transfers between chained operations.
///
/// Uses a ping-pong buffer pattern internally: in-place shaders (invert,
/// grayscale, brightness/contrast, saturation) operate on the current buffer
/// directly, while shaders that need separate input/output (blur) swap between
/// two GPU buffers.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, GpuChain};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let buf = PixelBuffer::zeroed(1920, 1080, PixelFormat::Rgba8);
/// let result = GpuChain::new(&ctx, &buf).unwrap()
///     .invert().unwrap()
///     .brightness_contrast(0.1, 1.2).unwrap()
///     .saturation(1.5).unwrap()
///     .finish().unwrap();
/// assert_eq!(result.width, 1920);
/// ```
pub struct GpuChain<'a> {
    ctx: &'a GpuContext,
    /// Current GPU buffer being operated on (ping).
    buf_a: GpuBuffer,
    /// Secondary buffer for operations that need a separate output (pong).
    buf_b: GpuBuffer,
    /// Which buffer is "current" (`true` = `buf_a`, `false` = `buf_b`).
    current_is_a: bool,
    width: u32,
    height: u32,
    pixel_count: u32,
}

impl<'a> GpuChain<'a> {
    /// Upload a [`PixelBuffer`] and begin a chain of GPU operations.
    ///
    /// Creates both ping and pong GPU buffers. The input data is uploaded to
    /// buffer A; buffer B is initialized to zeroes.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not RGBA8 or exceeds `u32` pixel count.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let chain = GpuChain::new(&ctx, &buf).unwrap();
    /// ```
    pub fn new(ctx: &'a GpuContext, buf: &PixelBuffer) -> Result<Self, RangaError> {
        if buf.format != PixelFormat::Rgba8 {
            return Err(RangaError::InvalidFormat("GpuChain requires RGBA8".into()));
        }
        let pixel_count = u32::try_from(buf.pixel_count()).map_err(|_| {
            RangaError::Other("image too large for GPU pipeline (exceeds u32 pixel count)".into())
        })?;

        let buf_a = GpuBuffer::upload(ctx, buf);
        let zeroed = PixelBuffer::zeroed(buf.width, buf.height, PixelFormat::Rgba8);
        let buf_b = GpuBuffer::upload(ctx, &zeroed);

        Ok(Self {
            ctx,
            buf_a,
            buf_b,
            current_is_a: true,
            width: buf.width,
            height: buf.height,
            pixel_count,
        })
    }

    /// Reference to whichever buffer is currently active.
    #[must_use]
    fn current_buf(&self) -> &GpuBuffer {
        if self.current_is_a {
            &self.buf_a
        } else {
            &self.buf_b
        }
    }

    /// Reference to whichever buffer is NOT currently active.
    #[must_use]
    fn other_buf(&self) -> &GpuBuffer {
        if self.current_is_a {
            &self.buf_b
        } else {
            &self.buf_a
        }
    }

    /// Apply color inversion (in-place on current buffer).
    ///
    /// Inverts R, G, B channels while preserving alpha.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .invert().unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn invert(self) -> Result<Self, RangaError> {
        let params = [self.pixel_count];
        let params_bytes = params_to_bytes(&params);
        dispatch_1buf_shader(
            self.ctx,
            "invert",
            &shaders::build_shader(shaders::INVERT),
            self.current_buf().wgpu_buffer(),
            params_bytes,
            self.pixel_count.div_ceil(256),
        );
        Ok(self)
    }

    /// Apply grayscale conversion (in-place on current buffer).
    ///
    /// Sets R, G, B to BT.709 luminance while preserving alpha.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .grayscale().unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn grayscale(self) -> Result<Self, RangaError> {
        let params = [self.pixel_count];
        let params_bytes = params_to_bytes(&params);
        dispatch_1buf_shader(
            self.ctx,
            "grayscale",
            &shaders::build_shader(shaders::GRAYSCALE),
            self.current_buf().wgpu_buffer(),
            params_bytes,
            self.pixel_count.div_ceil(256),
        );
        Ok(self)
    }

    /// Apply brightness and contrast adjustment (in-place on current buffer).
    ///
    /// `brightness` is an offset in the -1.0 to 1.0 range.
    /// `contrast` is a multiplier where 1.0 is unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .brightness_contrast(0.1, 1.2).unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn brightness_contrast(self, brightness: f32, contrast: f32) -> Result<Self, RangaError> {
        let params = [
            self.pixel_count,
            brightness.to_bits(),
            contrast.to_bits(),
            0u32,
        ];
        let params_bytes = params_to_bytes(&params);
        dispatch_1buf_shader(
            self.ctx,
            "brightness_contrast",
            &shaders::build_shader(shaders::BRIGHTNESS_CONTRAST),
            self.current_buf().wgpu_buffer(),
            params_bytes,
            self.pixel_count.div_ceil(256),
        );
        Ok(self)
    }

    /// Apply saturation adjustment (in-place on current buffer).
    ///
    /// `factor` of 1.0 is unchanged; 0.0 is grayscale; >1.0 increases saturation.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .saturation(1.5).unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn saturation(self, factor: f32) -> Result<Self, RangaError> {
        let params = [self.pixel_count, factor.to_bits(), 0u32, 0u32];
        let params_bytes = params_to_bytes(&params);
        dispatch_1buf_shader(
            self.ctx,
            "saturation",
            &shaders::build_shader(shaders::SATURATION),
            self.current_buf().wgpu_buffer(),
            params_bytes,
            self.pixel_count.div_ceil(256),
        );
        Ok(self)
    }

    /// Apply Gaussian blur using a separable two-pass approach.
    ///
    /// Uses ping-pong buffers: horizontal pass writes current -> other,
    /// then vertical pass writes other -> current. The active buffer
    /// remains the same after the operation.
    ///
    /// A `radius` of 0 is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU dispatch fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .gaussian_blur(3).unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn gaussian_blur(self, radius: u32) -> Result<Self, RangaError> {
        if radius == 0 {
            return Ok(self);
        }

        let kernel = build_gaussian_kernel(radius);

        // SAFETY: kernel is a contiguous Vec<f32>, reinterpreted as bytes for GPU upload.
        let kernel_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                kernel.as_ptr().cast::<u8>(),
                kernel.len() * std::mem::size_of::<f32>(),
            )
        };

        use wgpu::util::DeviceExt;
        let kernel_gpu = self
            .ctx
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blur_kernel"),
                contents: kernel_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params = [self.width, self.height, radius, 0u32];
        // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
        };

        let workgroups_x = self.width.div_ceil(16);
        let workgroups_y = self.height.div_ceil(16);

        // Pass 1: horizontal blur (current -> other)
        dispatch_blur_shader(
            self.ctx,
            "blur_horizontal",
            &shaders::build_shader(shaders::BLUR_HORIZONTAL),
            self.current_buf().wgpu_buffer(),
            self.other_buf().wgpu_buffer(),
            &kernel_gpu,
            params_bytes,
            workgroups_x,
            workgroups_y,
        );

        // Pass 2: vertical blur (other -> current)
        // After this, the result is back in the current buffer.
        dispatch_blur_shader(
            self.ctx,
            "blur_vertical",
            &shaders::build_shader(shaders::BLUR_VERTICAL),
            self.other_buf().wgpu_buffer(),
            self.current_buf().wgpu_buffer(),
            &kernel_gpu,
            params_bytes,
            workgroups_x,
            workgroups_y,
        );

        // current_is_a stays the same — result is back in the current buffer
        Ok(self)
    }

    /// Blend another image over the current buffer using the specified mode.
    ///
    /// The `other` buffer is uploaded to the GPU and blended onto the current
    /// buffer. Dimensions must match.
    ///
    /// # Errors
    ///
    /// Returns an error if `other` is not RGBA8, dimensions do not match,
    /// or a GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    /// use ranga::blend::BlendMode;
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let base = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let overlay = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &base).unwrap()
    ///     .blend(&overlay, BlendMode::Normal, 0.5).unwrap()
    ///     .finish().unwrap();
    /// ```
    pub fn blend(
        self,
        other: &PixelBuffer,
        mode: BlendMode,
        opacity: f32,
    ) -> Result<Self, RangaError> {
        if other.format != PixelFormat::Rgba8 {
            return Err(RangaError::InvalidFormat(
                "GpuChain blend requires RGBA8".into(),
            ));
        }
        if other.width != self.width || other.height != self.height {
            return Err(RangaError::DimensionMismatch {
                expected: (self.width as usize) * (self.height as usize),
                actual: other.pixel_count(),
            });
        }

        let mode_id: u32 = match mode {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            BlendMode::Overlay => 3,
            BlendMode::Darken => 4,
            BlendMode::Lighten => 5,
            BlendMode::ColorDodge => 6,
            BlendMode::ColorBurn => 7,
            BlendMode::SoftLight => 8,
            BlendMode::HardLight => 9,
            BlendMode::Difference => 10,
            BlendMode::Exclusion => 11,
        };

        let src_gpu = GpuBuffer::upload(self.ctx, other);

        let params = [self.pixel_count, mode_id, opacity.to_bits(), 0u32];
        let params_bytes = params_to_bytes(&params);

        // Blend shader: src (read-only) at binding 0, dst (read-write) at binding 1.
        // The current buffer is the destination.
        dispatch_3buf_shader(
            self.ctx,
            "blend_all",
            &shaders::build_shader(shaders::BLEND_ALL),
            src_gpu.wgpu_buffer(),
            self.current_buf().wgpu_buffer(),
            params_bytes,
            self.pixel_count.div_ceil(256),
        );

        Ok(self)
    }

    /// Download the current GPU buffer back to CPU and return the result.
    ///
    /// This is the terminal operation of the chain. After calling `finish`,
    /// the chain is consumed and all GPU resources are released.
    ///
    /// # Errors
    ///
    /// Returns an error if the GPU download fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::{GpuContext, GpuChain};
    /// use ranga::pixel::{PixelBuffer, PixelFormat};
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let buf = PixelBuffer::zeroed(64, 64, PixelFormat::Rgba8);
    /// let result = GpuChain::new(&ctx, &buf).unwrap()
    ///     .invert().unwrap()
    ///     .finish().unwrap();
    /// assert_eq!(result.width, 64);
    /// ```
    pub fn finish(self) -> Result<PixelBuffer, RangaError> {
        self.current_buf().download(self.ctx).map_err(Into::into)
    }
}

/// Convert a `[u32]` params array to a byte slice.
///
/// # Safety
///
/// This is safe because `u32` has no padding bytes and `u8` has alignment 1.
/// The returned slice borrows the input array and is valid for its lifetime.
#[inline]
fn params_to_bytes(params: &[u32]) -> &[u8] {
    // SAFETY: params is a contiguous slice of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(params))
    }
}

// ── Internal dispatch helpers ──────────────────────────────────────────────

/// Dispatch a compute shader with 1 storage buffer + 1 uniform buffer.
///
/// Uses the pipeline cache on [`GpuContext`] to avoid redundant compilation.
/// The uniform buffer is padded to 16-byte alignment as required by wgpu.
fn dispatch_1buf_shader(
    ctx: &GpuContext,
    name: &'static str,
    shader_src: &str,
    storage_buf: &wgpu::Buffer,
    params_data: &[u8],
    workgroups: u32,
) {
    // Get or create the cached pipeline (returns raw pointer to avoid holding borrow)
    let pipeline_ptr = ctx.get_or_create_pipeline_1buf(name, shader_src);

    // Pad uniform buffer to 16-byte alignment
    let aligned_size = params_data.len().div_ceil(16) * 16;
    let mut aligned_data = vec![0u8; aligned_size];
    aligned_data[..params_data.len()].copy_from_slice(params_data);

    let params_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: aligned_size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue().write_buffer(&params_buf, 0, &aligned_data);

    let bgl = ctx
        .bind_group_layout_1buf()
        .expect("layout created by get_or_create");
    let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    // Drop the Ref before submitting (not strictly required, but clean)
    drop(bgl);

    // SAFETY: The pipeline pointer remains valid because GpuContext owns the cache
    // and we hold a shared reference to GpuContext for the duration of this function.
    // SAFETY: pipeline_ptr is valid for the lifetime of ctx.cache (RefCell<PipelineCache>).
    // The borrow is dropped before this point, so no aliasing conflict.
    let pipeline = unsafe { &*pipeline_ptr };

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue().submit(Some(encoder.finish()));
}

/// Dispatch a compute shader with 2 storage buffers (src read-only, dst read-write) + 1 uniform.
///
/// Uses the pipeline cache on [`GpuContext`] to avoid redundant compilation.
/// The uniform buffer is padded to 16-byte alignment as required by wgpu.
fn dispatch_3buf_shader(
    ctx: &GpuContext,
    name: &'static str,
    shader_src: &str,
    src_buf: &wgpu::Buffer,
    dst_buf: &wgpu::Buffer,
    params_data: &[u8],
    workgroups: u32,
) {
    let pipeline_ptr = ctx.get_or_create_pipeline_3buf(name, shader_src);

    let aligned_size = params_data.len().div_ceil(16) * 16;
    let mut aligned_data = vec![0u8; aligned_size];
    aligned_data[..params_data.len()].copy_from_slice(params_data);

    let params_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: aligned_size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue().write_buffer(&params_buf, 0, &aligned_data);

    let bgl = ctx
        .bind_group_layout_3buf()
        .expect("layout created by get_or_create");
    let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    drop(bgl);

    // SAFETY: The pipeline pointer remains valid because GpuContext owns the cache
    // and we hold a shared reference to GpuContext for the duration of this function.
    // SAFETY: pipeline_ptr is valid for the lifetime of ctx.cache (RefCell<PipelineCache>).
    // The borrow is dropped before this point, so no aliasing conflict.
    let pipeline = unsafe { &*pipeline_ptr };

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue().submit(Some(encoder.finish()));
}

/// GPU-accelerated Gaussian blur (separable two-pass).
///
/// Applies a Gaussian blur with the given pixel `radius` using two GPU compute
/// passes — horizontal then vertical. The kernel sigma is `radius / 3` (clamped
/// to a minimum of 0.5), matching [`crate::filter::gaussian_blur`].
///
/// Returns a new blurred [`PixelBuffer`]. The input is not modified.
///
/// # Errors
///
/// Returns an error if the buffer is not RGBA8, radius is zero, or a GPU
/// operation fails.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::{GpuContext, gpu_gaussian_blur};
/// use ranga::pixel::{PixelBuffer, PixelFormat};
///
/// let ctx = GpuContext::new().unwrap();
/// let buf = PixelBuffer::new(vec![128; 64 * 64 * 4], 64, 64, PixelFormat::Rgba8).unwrap();
/// let blurred = gpu_gaussian_blur(&ctx, &buf, 3).unwrap();
/// assert_eq!(blurred.width, 64);
/// ```
pub fn gpu_gaussian_blur(
    ctx: &GpuContext,
    buf: &PixelBuffer,
    radius: u32,
) -> Result<PixelBuffer, RangaError> {
    if buf.format != PixelFormat::Rgba8 {
        return Err(RangaError::InvalidFormat("GPU blur requires RGBA8".into()));
    }
    if radius == 0 {
        return Ok(buf.clone());
    }

    let w = buf.width;
    let h = buf.height;

    // Build Gaussian kernel on CPU (same algorithm as filter.rs)
    let kernel = build_gaussian_kernel(radius);

    // Upload input and kernel
    let input_gpu = GpuBuffer::upload(ctx, buf);

    // Create output buffer (same size as input)
    let temp_buf = PixelBuffer::zeroed(w, h, PixelFormat::Rgba8);
    let temp_gpu = GpuBuffer::upload(ctx, &temp_buf);

    let output_buf = PixelBuffer::zeroed(w, h, PixelFormat::Rgba8);
    let output_gpu = GpuBuffer::upload(ctx, &output_buf);

    // SAFETY: kernel is a contiguous Vec<f32>, reinterpreted as bytes for GPU upload.
    let kernel_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            kernel.as_ptr().cast::<u8>(),
            kernel.len() * std::mem::size_of::<f32>(),
        )
    };

    use wgpu::util::DeviceExt;
    let kernel_gpu = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_kernel"),
            contents: kernel_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Params: width, height, radius, _pad
    let params = [w, h, radius, 0u32];
    // SAFETY: params is a contiguous array of u32, reinterpreted as bytes.
    // Alignment is satisfied (u8 has alignment 1) and the size is exact.
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };

    let workgroups_x = w.div_ceil(16);
    let workgroups_y = h.div_ceil(16);

    // Pass 1: horizontal blur (input -> temp)
    dispatch_blur_shader(
        ctx,
        "blur_horizontal",
        &shaders::build_shader(shaders::BLUR_HORIZONTAL),
        input_gpu.wgpu_buffer(),
        temp_gpu.wgpu_buffer(),
        &kernel_gpu,
        params_bytes,
        workgroups_x,
        workgroups_y,
    );

    // Pass 2: vertical blur (temp -> output)
    dispatch_blur_shader(
        ctx,
        "blur_vertical",
        &shaders::build_shader(shaders::BLUR_VERTICAL),
        temp_gpu.wgpu_buffer(),
        output_gpu.wgpu_buffer(),
        &kernel_gpu,
        params_bytes,
        workgroups_x,
        workgroups_y,
    );

    output_gpu.download(ctx).map_err(Into::into)
}

/// Build a 1D Gaussian kernel with the given radius.
///
/// Sigma is `radius / 3` clamped to a minimum of 0.5, matching the CPU
/// implementation in [`crate::filter`].
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

/// Dispatch a blur compute shader with 4 bindings (input, output, kernel, params).
///
/// Uses a dedicated bind group layout with read-only input, read-write output,
/// read-only kernel storage, and a uniform params buffer. The pipeline is cached
/// by name on the context.
#[allow(clippy::too_many_arguments)]
fn dispatch_blur_shader(
    ctx: &GpuContext,
    name: &'static str,
    shader_src: &str,
    input_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    kernel_buf: &wgpu::Buffer,
    params_data: &[u8],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    // Blur pipelines need a 4-binding layout — we create them inline and cache
    // via the pipeline name. Since the layout differs from 1buf/3buf, we build
    // the pipeline directly when not cached.
    let cache = ctx.cache.borrow();
    let cached = cache.pipelines.contains_key(name);
    drop(cache);

    if !cached {
        let bgl = ctx
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ranga_blur_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pl = ctx
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ranga_blur_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let shader = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut cache = ctx.cache.borrow_mut();
        cache.pipelines.insert(name, pipeline);
    }

    // Pad params to 16-byte alignment
    let aligned_size = params_data.len().div_ceil(16) * 16;
    let mut aligned_data = vec![0u8; aligned_size];
    aligned_data[..params_data.len()].copy_from_slice(params_data);

    let params_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_params"),
        size: aligned_size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue().write_buffer(&params_buf, 0, &aligned_data);

    // Get the pipeline's bind group layout from the pipeline itself
    let cache = ctx.cache.borrow();
    let pipeline = cache.pipelines.get(name).expect("just created");
    let bgl = pipeline.get_bind_group_layout(0);

    let bg = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: kernel_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
    drop(cache);
    ctx.queue().submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;

    /// Helper to get a GPU context, returning None if unavailable (e.g. headless CI).
    fn try_gpu() -> Option<GpuContext> {
        GpuContext::new().ok()
    }

    #[test]
    fn gpu_invert_matches_cpu() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return, // Skip in headless CI
        };

        let data: Vec<u8> = (0..64u8).flat_map(|i| [i * 4, i * 3, i * 2, 255]).collect();
        let mut gpu_buf = PixelBuffer::new(data.clone(), 8, 8, PixelFormat::Rgba8).unwrap();
        let mut cpu_buf = PixelBuffer::new(data, 8, 8, PixelFormat::Rgba8).unwrap();

        gpu_invert(&ctx, &mut gpu_buf).unwrap();
        crate::filter::invert(&mut cpu_buf).unwrap();

        // GPU uses f32 rounding, CPU uses integer subtraction — allow +/-1 tolerance
        for (g, c) in gpu_buf.data.iter().zip(cpu_buf.data.iter()) {
            assert!(
                (*g as i16 - *c as i16).unsigned_abs() <= 1,
                "GPU={g} CPU={c}"
            );
        }
    }

    #[test]
    fn gpu_grayscale_produces_uniform_channels() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..16u8)
            .flat_map(|i| {
                [
                    200u8.wrapping_add(i * 3),
                    100u8.wrapping_add(i * 7),
                    50u8.wrapping_add(i * 11),
                    255,
                ]
            })
            .collect();
        let mut buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();
        gpu_grayscale(&ctx, &mut buf).unwrap();

        // After grayscale, R == G == B for each pixel
        for pixel in buf.data.chunks_exact(4) {
            assert_eq!(pixel[0], pixel[1], "R != G: {} != {}", pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2], "G != B: {} != {}", pixel[1], pixel[2]);
            assert_eq!(pixel[3], 255, "alpha changed");
        }
    }

    #[test]
    fn gpu_brightness_contrast_identity() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..16u8)
            .flat_map(|i| [i * 16, i * 8, i * 4, 255])
            .collect();
        let original = data.clone();
        let mut buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();

        // brightness=0.0, contrast=1.0 should be identity
        gpu_brightness_contrast(&ctx, &mut buf, 0.0, 1.0).unwrap();

        for (g, o) in buf.data.iter().zip(original.iter()) {
            assert!(
                (*g as i16 - *o as i16).unsigned_abs() <= 1,
                "GPU={g} original={o}"
            );
        }
    }

    #[test]
    fn gpu_blend_normal_opaque() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        // Opaque red over opaque blue with Normal blend
        let src_data = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let dst_data = vec![0, 0, 255, 255, 0, 0, 255, 255];

        let src = PixelBuffer::new(src_data, 2, 1, PixelFormat::Rgba8).unwrap();
        let mut dst = PixelBuffer::new(dst_data, 2, 1, PixelFormat::Rgba8).unwrap();

        gpu_blend(&ctx, &src, &mut dst, BlendMode::Normal, 1.0).unwrap();

        // With opacity 1.0 and src alpha 255, result should be the source color
        assert!(dst.data[0] > 250, "red channel: {}", dst.data[0]);
        assert!(dst.data[1] < 5, "green channel: {}", dst.data[1]);
        assert!(dst.data[2] < 5, "blue channel: {}", dst.data[2]);
    }

    #[test]
    fn gpu_blend_dimension_mismatch() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let src = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        let mut dst = PixelBuffer::zeroed(8, 8, PixelFormat::Rgba8);

        let result = gpu_blend(&ctx, &src, &mut dst, BlendMode::Normal, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn gpu_invert_rejects_non_rgba8() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgb8);
        let result = gpu_invert(&ctx, &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn gpu_gaussian_blur_uniform_unchanged() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        // Uniform buffer should be (nearly) unchanged by blur
        let buf = PixelBuffer::new(vec![128; 8 * 8 * 4], 8, 8, PixelFormat::Rgba8).unwrap();
        let blurred = gpu_gaussian_blur(&ctx, &buf, 2).unwrap();
        for (i, &v) in blurred.data.iter().enumerate() {
            assert!(
                (v as i16 - 128).unsigned_abs() <= 1,
                "pixel byte {i}: expected ~128, got {v}"
            );
        }
    }

    #[test]
    fn gpu_gaussian_blur_radius_zero_is_identity() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..16u8)
            .flat_map(|i| [i * 16, i * 8, i * 4, 255])
            .collect();
        let buf = PixelBuffer::new(data.clone(), 4, 4, PixelFormat::Rgba8).unwrap();
        let blurred = gpu_gaussian_blur(&ctx, &buf, 0).unwrap();
        assert_eq!(blurred.data, data);
    }

    #[test]
    fn gpu_gaussian_blur_rejects_non_rgba8() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgb8);
        let result = gpu_gaussian_blur(&ctx, &buf, 2);
        assert!(result.is_err());
    }

    #[test]
    fn build_gaussian_kernel_sums_to_one() {
        let kernel = super::build_gaussian_kernel(5);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum={sum}");
        assert_eq!(kernel.len(), 11); // 2*5+1
    }

    #[test]
    fn pipeline_cache_reuses_pipelines() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        // Run invert twice — second call should reuse cached pipeline
        let mut buf1 = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        gpu_invert(&ctx, &mut buf1).unwrap();
        let mut buf2 = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
        gpu_invert(&ctx, &mut buf2).unwrap();

        // Verify cache has the entry
        let cache = ctx.cache.borrow();
        assert!(cache.pipelines.contains_key("invert"));
    }

    // ── GpuChain tests ────────────────────────────────────────────────────

    #[test]
    fn gpu_chain_invert_matches_single() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..64u8).flat_map(|i| [i * 4, i * 3, i * 2, 255]).collect();
        let buf = PixelBuffer::new(data.clone(), 8, 8, PixelFormat::Rgba8).unwrap();

        // Chain with just invert
        let chain_result = GpuChain::new(&ctx, &buf)
            .unwrap()
            .invert()
            .unwrap()
            .finish()
            .unwrap();

        // Single gpu_invert
        let mut single_buf = PixelBuffer::new(data, 8, 8, PixelFormat::Rgba8).unwrap();
        gpu_invert(&ctx, &mut single_buf).unwrap();

        for (c, s) in chain_result.data.iter().zip(single_buf.data.iter()) {
            assert!(
                (*c as i16 - *s as i16).unsigned_abs() <= 1,
                "chain={c} single={s}"
            );
        }
    }

    #[test]
    fn gpu_chain_multiple_ops() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..64u8).flat_map(|i| [i * 4, i * 3, i * 2, 255]).collect();
        let buf = PixelBuffer::new(data.clone(), 8, 8, PixelFormat::Rgba8).unwrap();

        let result = GpuChain::new(&ctx, &buf)
            .unwrap()
            .invert()
            .unwrap()
            .brightness_contrast(0.1, 1.2)
            .unwrap()
            .finish()
            .unwrap();

        // Result should differ from the original (non-trivial transformation)
        assert_ne!(result.data, data, "chain should produce a different result");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.data.len(), 8 * 8 * 4);
    }

    #[test]
    fn gpu_chain_with_blur() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let data: Vec<u8> = (0..64u8).flat_map(|i| [i * 4, i * 3, i * 2, 255]).collect();
        let buf = PixelBuffer::new(data, 8, 8, PixelFormat::Rgba8).unwrap();

        let result = GpuChain::new(&ctx, &buf)
            .unwrap()
            .invert()
            .unwrap()
            .gaussian_blur(2)
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        assert_eq!(result.data.len(), 8 * 8 * 4);
    }

    #[test]
    fn gpu_chain_rejects_non_rgba8() {
        let ctx = match try_gpu() {
            Some(ctx) => ctx,
            None => return,
        };

        let buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgb8);
        let result = GpuChain::new(&ctx, &buf);
        assert!(result.is_err());
    }
}
