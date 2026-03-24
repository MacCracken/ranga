//! GPU device/queue management.
//!
//! [`GpuContext`] wraps a wgpu device and queue, providing the foundation for
//! all GPU compute operations. Create one context and reuse it across multiple
//! operations to amortize initialization cost.

use std::cell::RefCell;
use std::collections::HashMap;

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

/// Cached compute pipelines and their associated bind group / pipeline layouts.
///
/// Stores compiled pipelines keyed by shader name so they can be reused across
/// multiple dispatches, avoiding the expensive per-call pipeline compilation.
///
/// # Examples
///
/// ```no_run
/// use ranga::gpu::GpuContext;
///
/// let ctx = GpuContext::new().expect("GPU required");
/// // Pipelines are cached automatically when using gpu_* functions.
/// ```
pub struct PipelineCache {
    pub(super) pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    /// Cached bind group layout + pipeline layout for 1-buffer (storage + uniform) shaders.
    layouts_1buf: Option<(wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    /// Cached bind group layout + pipeline layout for 3-buffer (src + dst + uniform) shaders.
    layouts_3buf: Option<(wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            layouts_1buf: None,
            layouts_3buf: None,
        }
    }
}

/// GPU compute context — manages wgpu device and queue.
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend: String,
    pub(super) cache: RefCell<PipelineCache>,
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(GpuError::NoAdapter)?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = format!("{:?}", info.backend);

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ranga-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            cache: RefCell::new(PipelineCache::new()),
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
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the underlying wgpu queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Return a cached 1-buffer compute pipeline, creating it on first use.
    ///
    /// The pipeline uses a bind group layout with one read-write storage buffer
    /// at `@binding(0)` and one uniform buffer at `@binding(1)`.
    ///
    /// `name` must be a unique `&'static str` key (typically the shader constant
    /// name). `shader_src` is the full WGSL source including pack helpers.
    ///
    /// Uses interior mutability ([`RefCell`]) so the context can be shared
    /// with `&self`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// let pipeline = ctx.get_or_create_pipeline_1buf("invert", "/* wgsl */");
    /// // `pipeline` is a raw pointer valid for the lifetime of `ctx`.
    /// ```
    pub fn get_or_create_pipeline_1buf(
        &self,
        name: &'static str,
        shader_src: &str,
    ) -> *const wgpu::ComputePipeline {
        let mut cache = self.cache.borrow_mut();

        if !cache.pipelines.contains_key(name) {
            let (_bgl, pl) = cache.layouts_1buf.get_or_insert_with(|| {
                let bgl = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("ranga_1buf_bgl"),
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
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ranga_1buf_pl"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });
                (bgl, pl)
            });

            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(pl),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            cache.pipelines.insert(name, pipeline);
        }

        let pipeline = cache.pipelines.get(name).expect("just inserted");
        pipeline as *const wgpu::ComputePipeline
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
    /// let ctx = GpuContext::new().unwrap();
    /// let pipeline = ctx.get_or_create_pipeline_3buf("blend", "/* wgsl */");
    /// ```
    pub fn get_or_create_pipeline_3buf(
        &self,
        name: &'static str,
        shader_src: &str,
    ) -> *const wgpu::ComputePipeline {
        let mut cache = self.cache.borrow_mut();

        if !cache.pipelines.contains_key(name) {
            let (_bgl, pl) = cache.layouts_3buf.get_or_insert_with(|| {
                let bgl = self
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("ranga_3buf_bgl"),
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
                let pl = self
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("ranga_3buf_pl"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });
                (bgl, pl)
            });

            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(pl),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            cache.pipelines.insert(name, pipeline);
        }

        let pipeline = cache.pipelines.get(name).expect("just inserted");
        pipeline as *const wgpu::ComputePipeline
    }

    /// Access the bind group layout for 1-buffer pipelines.
    ///
    /// Returns `None` if no 1-buffer pipeline has been created yet.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// // After creating a 1-buffer pipeline, the layout is available:
    /// // let bgl = ctx.bind_group_layout_1buf().unwrap();
    /// ```
    pub fn bind_group_layout_1buf(&self) -> Option<std::cell::Ref<'_, wgpu::BindGroupLayout>> {
        let cache = self.cache.borrow();
        if cache.layouts_1buf.is_some() {
            Some(std::cell::Ref::map(cache, |c| {
                &c.layouts_1buf.as_ref().unwrap().0
            }))
        } else {
            None
        }
    }

    /// Access the bind group layout for 3-buffer pipelines.
    ///
    /// Returns `None` if no 3-buffer pipeline has been created yet.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ranga::gpu::GpuContext;
    ///
    /// let ctx = GpuContext::new().unwrap();
    /// // After creating a 3-buffer pipeline, the layout is available:
    /// // let bgl = ctx.bind_group_layout_3buf().unwrap();
    /// ```
    pub fn bind_group_layout_3buf(&self) -> Option<std::cell::Ref<'_, wgpu::BindGroupLayout>> {
        let cache = self.cache.borrow();
        if cache.layouts_3buf.is_some() {
            Some(std::cell::Ref::map(cache, |c| {
                &c.layouts_3buf.as_ref().unwrap().0
            }))
        } else {
            None
        }
    }
}

/// Simple block_on implementation that avoids an async runtime dependency.
///
/// Uses a spin-yield loop with a no-op waker. Suitable for the short-lived
/// futures returned by wgpu (adapter request, device request).
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut future = pin!(f);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::yield_now(),
        }
    }
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
}
