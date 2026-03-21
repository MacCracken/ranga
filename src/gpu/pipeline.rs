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

    let pixel_count = src.pixel_count() as u32;
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
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };

    dispatch_3buf_shader(
        ctx,
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
    let pixel_count = buf.pixel_count() as u32;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
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
    let pixel_count = buf.pixel_count() as u32;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
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
    let pixel_count = buf.pixel_count() as u32;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count, brightness.to_bits(), contrast.to_bits(), 0u32];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
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
    let pixel_count = buf.pixel_count() as u32;
    let gpu_buf = GpuBuffer::upload(ctx, buf);
    let params = [pixel_count, factor.to_bits(), 0u32, 0u32];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
    };
    dispatch_1buf_shader(
        ctx,
        &shaders::build_shader(shaders::SATURATION),
        gpu_buf.wgpu_buffer(),
        params_bytes,
        pixel_count.div_ceil(256),
    );
    let result = gpu_buf.download(ctx)?;
    buf.data = result.data;
    Ok(())
}

// ── Internal dispatch helpers ──────────────────────────────────────────────

/// Dispatch a compute shader with 1 storage buffer + 1 uniform buffer.
///
/// The uniform buffer is padded to 16-byte alignment as required by wgpu.
fn dispatch_1buf_shader(
    ctx: &GpuContext,
    shader_src: &str,
    storage_buf: &wgpu::Buffer,
    params_data: &[u8],
    workgroups: u32,
) {
    let shader = ctx
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ranga_compute"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let bgl = ctx
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

    let pl = ctx
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ranga_1buf"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue().submit(Some(encoder.finish()));
}

/// Dispatch a compute shader with 2 storage buffers (src read-only, dst read-write) + 1 uniform.
///
/// The uniform buffer is padded to 16-byte alignment as required by wgpu.
fn dispatch_3buf_shader(
    ctx: &GpuContext,
    shader_src: &str,
    src_buf: &wgpu::Buffer,
    dst_buf: &wgpu::Buffer,
    params_data: &[u8],
    workgroups: u32,
) {
    let shader = ctx
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ranga_composite"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let bgl = ctx
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

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

    let pl = ctx
        .device()
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let pipeline = ctx
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ranga_3buf"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
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
}
