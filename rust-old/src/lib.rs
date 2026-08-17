//! # Ranga — Core Image Processing Library
//!
//! Ranga (रंग, Sanskrit: color/hue) provides shared image processing primitives
//! for the AGNOS creative suite. It eliminates duplicate implementations across
//! rasa (image editor), tazama (video editor), and aethersafta (compositor).
//!
//! ## Modules
//!
//! - [`color`] — Color spaces (sRGB, linear, HSL, CIE XYZ/Lab, Oklab/Oklch, CMYK, P3), Delta-E, color temperature
//! - [`pixel`] — Pixel buffer with 6 formats, zero-copy views, buffer pool
//! - [`blend`] — 12 Porter-Duff blend modes with SSE2/AVX2/NEON SIMD
//! - [`convert`] — Pixel format conversion (BT.601/709/2020, ARGB↔NV12, RGB8↔RGBA8, RgbaF32)
//! - [`filter`] — 24+ CPU filters (blur, sharpen, hue shift, 3D LUT, noise, median, bilateral)
//! - [`composite`] — Layer compositing, masks, transitions (dissolve/fade/wipe), gradients
//! - [`histogram`] — Luminance/RGB histograms, equalization, auto-levels
//! - [`transform`] — Crop, resize (nearest/bilinear/bicubic), affine, perspective, flip
//! - [`icc`] — ICC v2/v4 profile parsing, tone curves, embedded sRGB profile
//! - `gpu` — GPU compute: blend, filters, noise, transitions, crop/resize/flip, batched dispatch (`GpuChain`)
//! - `hwaccel` — GPU detection with VRAM/utilization-aware offload decisions
//! - `spectral` — Physically-based color science via prakash (SPD, CIE CMFs, illuminants, CRI)
//!
//! ## Feature Flags
//!
//! - `simd` (default) — SSE2/AVX2/NEON SIMD acceleration for blend and convert
//! - `gpu` — wgpu compute pipelines with `GpuChain` batched dispatch
//! - `hwaccel` — hardware accelerator detection via ai-hwaccel 0.23.3
//! - `parallel` — rayon row-parallel blur
//! - `spectral` — physically-based color science via prakash
//! - `full` — all features
//!
//! ## Example
//!
//! ```
//! use ranga::pixel::{PixelBuffer, PixelFormat};
//! use ranga::filter;
//!
//! let mut buf = PixelBuffer::zeroed(4, 4, PixelFormat::Rgba8);
//! filter::brightness(&mut buf, 0.5).unwrap();
//! ```

pub mod blend;
pub mod color;
pub mod composite;
pub mod convert;
pub mod filter;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod histogram;
#[cfg(feature = "hwaccel")]
pub mod hwaccel;
pub mod icc;
pub mod pixel;
#[cfg(feature = "spectral")]
pub mod spectral;
pub mod transform;

mod error;
pub use error::RangaError;

/// Result type alias for ranga operations.
pub type Result<T> = std::result::Result<T, RangaError>;
