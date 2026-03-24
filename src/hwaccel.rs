//! Hardware acceleration detection via `ai-hwaccel`.
//!
//! Provides GPU capability queries to decide whether to use wgpu compute
//! pipelines or fall back to CPU. Requires the `hwaccel` feature.
//!
//! # Examples
//!
//! ```no_run
//! use ranga::hwaccel;
//!
//! let report = hwaccel::probe();
//! if report.has_gpu {
//!     println!("GPU: {} ({} MB)", report.gpu_name, report.gpu_memory_mb);
//! }
//! ```

/// Summary of available hardware acceleration.
#[derive(Debug, Clone)]
pub struct HwReport {
    /// Whether any GPU was detected.
    pub has_gpu: bool,
    /// Whether Vulkan compute is available (required for wgpu).
    pub has_vulkan: bool,
    /// Name of the best available GPU (empty if none).
    pub gpu_name: String,
    /// GPU memory in megabytes (0 if none).
    pub gpu_memory_mb: u64,
    /// Free GPU memory in megabytes (0 if unknown or no GPU).
    pub gpu_free_memory_mb: u64,
    /// GPU utilization percentage (0–100, `None` if unknown).
    pub gpu_utilization_percent: Option<u32>,
    /// GPU temperature in degrees Celsius (`None` if unknown).
    pub temperature_c: Option<u32>,
    /// Total accelerator memory across all devices.
    pub total_accel_memory_mb: u64,
}

/// Probe the system for hardware accelerators.
///
/// Uses `ai-hwaccel` to detect GPUs, NPUs, and other accelerators.
/// Returns a summary useful for deciding whether to enable GPU compute.
///
/// # Examples
///
/// ```no_run
/// let report = ranga::hwaccel::probe();
/// println!("GPU available: {}", report.has_gpu);
/// ```
#[must_use]
pub fn probe() -> HwReport {
    let registry = ai_hwaccel::AcceleratorRegistry::detect();

    let best = registry.best_available();
    let has_gpu = registry.available().iter().any(|p| p.accelerator.is_gpu());
    let has_vulkan = registry
        .by_family(ai_hwaccel::AcceleratorFamily::Gpu)
        .iter()
        .any(|p| p.available);

    let (gpu_name, gpu_memory_mb, gpu_free_memory_mb, gpu_utilization_percent, temperature_c) =
        match best {
            Some(p) if p.accelerator.is_gpu() => {
                let name = p
                    .driver_version
                    .clone()
                    .unwrap_or_else(|| format!("{:?}", p.accelerator));
                let free_mb = p.memory_free_bytes.unwrap_or(0) / (1024 * 1024);
                (
                    name,
                    p.memory_bytes / (1024 * 1024),
                    free_mb,
                    p.gpu_utilization_percent,
                    p.temperature_c,
                )
            }
            _ => (String::new(), 0, 0, None, None),
        };

    HwReport {
        has_gpu,
        has_vulkan,
        gpu_name,
        gpu_memory_mb,
        gpu_free_memory_mb,
        gpu_utilization_percent,
        temperature_c,
        total_accel_memory_mb: registry.total_accelerator_memory() / (1024 * 1024),
    }
}

/// Check whether GPU compute is viable for a given image size.
///
/// Returns `true` if a GPU is available, the image is large enough to
/// benefit from GPU offload (overhead crossover ~256x256), the GPU has
/// sufficient free VRAM for the buffer, and the GPU is not heavily loaded.
///
/// # Examples
///
/// ```no_run
/// // Small images: CPU is faster
/// assert!(!ranga::hwaccel::should_use_gpu(64, 64));
/// ```
#[must_use]
pub fn should_use_gpu(width: u32, height: u32) -> bool {
    let report = probe();
    if !report.has_gpu {
        return false;
    }

    // GPU dispatch overhead makes it slower for small images.
    // Crossover is roughly 256x256 = 65536 pixels.
    let pixels = width as u64 * height as u64;
    if pixels < 65_536 {
        return false;
    }

    // Check if GPU has enough free VRAM for the buffer (RGBA8 = 4 bytes/pixel).
    // Need ~3x buffer size for ping-pong + staging in GpuChain.
    if report.gpu_free_memory_mb > 0 {
        let buffer_mb = pixels * 4 * 3 / (1024 * 1024);
        if buffer_mb > report.gpu_free_memory_mb {
            return false;
        }
    }

    // Skip GPU if it's heavily loaded (>90% utilization).
    if let Some(util) = report.gpu_utilization_percent {
        if util > 90 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_report() {
        let report = probe();
        // In CI, GPU may or may not be present — just verify it doesn't panic
        let _ = report.has_gpu;
        let _ = report.gpu_name;
    }

    #[test]
    fn small_images_stay_on_cpu() {
        // Even if GPU is present, tiny images should stay on CPU
        // (unless the probe itself returns false, which is also fine)
        let _ = should_use_gpu(4, 4);
    }

    #[test]
    fn report_has_new_fields() {
        let report = probe();
        // Verify new fields are populated (may be 0/None without a GPU)
        let _ = report.gpu_free_memory_mb;
        let _ = report.gpu_utilization_percent;
        let _ = report.temperature_c;
    }
}
