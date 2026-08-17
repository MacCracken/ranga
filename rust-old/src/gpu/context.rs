//! GPU device/queue management.
//!
//! [`GpuContext`] wraps a [`mabda::GpuContext`] to provide the foundation for
//! all GPU compute operations. Create one context and reuse it across multiple
//! operations to amortize initialization cost.

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::RangaError;

/// Errors specific to GPU operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GpuError {
    /// No suitable GPU adapter was found on the system.
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    /// Failed to request a GPU device from the adapter.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(String),
    /// A GPU buffer operation (upload, download, map) failed.
    #[error("GPU buffer operation failed: {0}")]
    BufferOp(String),
}

impl From<GpuError> for RangaError {
    fn from(e: GpuError) -> Self {
        RangaError::Other(e.to_string())
    }
}

impl From<mabda::GpuError> for GpuError {
    fn from(e: mabda::GpuError) -> Self {
        match e {
            mabda::GpuError::AdapterNotFound => GpuError::NoAdapter,
            mabda::GpuError::DeviceRequest(inner) => GpuError::DeviceRequest(inner.to_string()),
            other => GpuError::BufferOp(other.to_string()),
        }
    }
}

/// GPU compute context — wraps [`mabda::GpuContext`] with pipeline and shader caches.
///
/// Create once and reuse for multiple operations to amortize initialization cost.
/// Includes a pipeline cache that stores compiled compute pipelines so
/// repeated calls to the same shader avoid redundant compilation.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::GpuContext;
///
/// let ctx = GpuContext::new().expect("GPU required");
/// println!("Using GPU: {} ({})", ctx.adapter_name(), ctx.backend_name());
/// ```
pub struct GpuContext {
    pub(super) inner: mabda::GpuContext,
    pub(super) cache: mabda::PipelineCache,
    #[allow(dead_code)]
    pub(super) shader_cache: mabda::ShaderCache,
    adapter_name: String,
    backend: String,
}

impl GpuContext {
    /// Create a new GPU context, requesting a high-performance adapter.
    ///
    /// Returns [`GpuError::NoAdapter`] if no suitable GPU is found, or
    /// [`GpuError::DeviceRequest`] if the device cannot be created.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let ctx = GpuContext::new().expect("GPU required");
    /// ```
    pub fn new() -> Result<Self, GpuError> {
        let inner = pollster::block_on(mabda::GpuContext::new())?;

        let info = inner.adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = format!("{:?}", info.backend);

        Ok(Self {
            inner,
            cache: mabda::PipelineCache::new(),
            shader_cache: mabda::ShaderCache::new(),
            adapter_name,
            backend,
        })
    }

    /// Create a GPU context using ai-hwaccel to detect hardware first.
    ///
    /// Logs detected GPU information via tracing before falling back to
    /// standard wgpu initialization. Only available with the `hwaccel` feature.
    #[cfg(feature = "hwaccel")]
    pub fn new_with_hwaccel() -> Result<Self, GpuError> {
        let report = crate::hwaccel::probe();
        if !report.has_gpu {
            return Err(GpuError::NoAdapter);
        }
        tracing::info!(
            gpu = %report.gpu_name,
            memory_mb = report.gpu_memory_mb,
            "GPU detected via ai-hwaccel"
        );
        Self::new()
    }

    /// The name of the GPU adapter (e.g. "NVIDIA GeForce RTX 4090").
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The wgpu backend in use (e.g. "Vulkan", "Metal").
    #[must_use]
    pub fn backend_name(&self) -> &str {
        &self.backend
    }

    /// Access the underlying wgpu device.
    #[must_use]
    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Access the underlying wgpu queue.
    #[must_use]
    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// Return a cached 1-buffer compute pipeline, creating it on first use.
    ///
    /// The pipeline uses a bind group layout with one read-write storage buffer
    /// at `@binding(0)` and one uniform buffer at `@binding(1)`.
    ///
    /// `name` must be a unique `&'static str` key (typically the shader constant
    /// name). `shader_src` is the full WGSL source including pack helpers.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let mut ctx = GpuContext::new().unwrap();
    /// let pipeline = ctx.get_or_create_pipeline_1buf("invert", "/* wgsl */");
    /// ```
    pub fn get_or_create_pipeline_1buf(
        &mut self,
        name: &'static str,
        shader_src: &str,
    ) -> Result<&mabda::compute::ComputePipeline, RangaError> {
        let key = hash_name(name);
        let device = &self.inner.device;

        let pipeline = self.cache.get_or_insert_compute(key, || {
            let entries: &[wgpu::BindGroupLayoutEntry] = &[
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
            ];
            mabda::compute::ComputePipeline::with_layout(device, shader_src, "main", entries)
        });

        Ok(pipeline)
    }

    /// Return a cached 3-buffer compute pipeline, creating it on first use.
    ///
    /// The pipeline uses a bind group layout with a read-only storage buffer
    /// at `@binding(0)`, a read-write storage buffer at `@binding(1)`, and a
    /// uniform buffer at `@binding(2)`.
    ///
    /// `name` must be a unique `&'static str` key. `shader_src` is the full
    /// WGSL source including pack helpers.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let mut ctx = GpuContext::new().unwrap();
    /// let pipeline = ctx.get_or_create_pipeline_3buf("blend", "/* wgsl */");
    /// ```
    pub fn get_or_create_pipeline_3buf(
        &mut self,
        name: &'static str,
        shader_src: &str,
    ) -> Result<&mabda::compute::ComputePipeline, RangaError> {
        let key = hash_name(name);
        let device = &self.inner.device;

        let pipeline = self.cache.get_or_insert_compute(key, || {
            let entries: &[wgpu::BindGroupLayoutEntry] = &[
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
            ];
            mabda::compute::ComputePipeline::with_layout(device, shader_src, "main", entries)
        });

        Ok(pipeline)
    }

    /// Return a cached 4-binding compute pipeline (for blur shaders), creating it on first use.
    ///
    /// Layout: input (read-only storage), output (read-write storage), kernel (read-only storage),
    /// params (uniform).
    pub(super) fn get_or_create_pipeline_4buf(
        &mut self,
        name: &'static str,
        shader_src: &str,
    ) -> Result<&mabda::compute::ComputePipeline, RangaError> {
        let key = hash_name(name);
        let device = &self.inner.device;

        let pipeline = self.cache.get_or_insert_compute(key, || {
            let entries: &[wgpu::BindGroupLayoutEntry] = &[
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
            ];
            mabda::compute::ComputePipeline::with_layout(device, shader_src, "main", entries)
        });

        Ok(pipeline)
    }
}

/// Hash a pipeline name to a `u64` cache key.
#[inline]
fn hash_name(name: &str) -> u64 {
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_error_converts_to_ranga_error() {
        let gpu_err = GpuError::NoAdapter;
        let ranga_err: RangaError = gpu_err.into();
        assert!(ranga_err.to_string().contains("no suitable GPU"));
    }

    #[test]
    fn gpu_error_device_request_message() {
        let gpu_err = GpuError::DeviceRequest("limits exceeded".into());
        assert!(gpu_err.to_string().contains("limits exceeded"));
    }

    #[test]
    fn gpu_error_buffer_op_message() {
        let gpu_err = GpuError::BufferOp("map failed".into());
        assert!(gpu_err.to_string().contains("map failed"));
    }

    #[test]
    fn mabda_error_converts_to_gpu_error() {
        let mabda_err = mabda::GpuError::AdapterNotFound;
        let gpu_err: GpuError = mabda_err.into();
        assert!(matches!(gpu_err, GpuError::NoAdapter));
    }
}
