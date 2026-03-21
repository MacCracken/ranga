//! GPU compute pipeline via wgpu.
//!
//! Provides GPU-accelerated versions of blend, color conversion, and filter
//! operations. Requires the `gpu` feature flag.
//!
//! # Architecture
//!
//! - [`GpuContext`] manages the wgpu device/queue with optional ai-hwaccel detection
//! - [`GpuBuffer`] wraps GPU storage buffers for pixel data upload/download
//! - Shader modules provide WGSL compute shaders for each operation
//! - Automatic CPU fallback when GPU is unavailable

mod buffer;
mod context;
mod pipeline;
mod shaders;

pub use buffer::GpuBuffer;
pub use context::{GpuContext, GpuError};
pub use pipeline::{
    gpu_blend, gpu_brightness_contrast, gpu_gaussian_blur, gpu_grayscale, gpu_invert,
    gpu_saturation,
};
