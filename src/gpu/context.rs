//! GPU device/queue management.
//!
//! [`GpuContext`] wraps a wgpu device and queue, providing the foundation for
//! all GPU compute operations. Create one context and reuse it across multiple
//! operations to amortize initialization cost.

use crate::RangaError;

/// Errors specific to GPU operations.
#[derive(Debug, thiserror::Error)]
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

/// GPU compute context — manages wgpu device and queue.
///
/// Create once and reuse for multiple operations to amortize initialization cost.
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
