//! # Ranga — Core Image Processing Library
//!
//! Ranga (रंग, Sanskrit: color/hue) provides shared image processing primitives
//! for the AGNOS creative suite. It eliminates duplicate implementations across
//! rasa (image editor), tazama (video editor), and aethersafta (compositor).
//!
//! ## Modules
//!
//! - [`color`] — Color spaces, conversions, ICC profiles
//! - [`pixel`] — Pixel buffer type with format-aware operations
//! - [`blend`] — 12 Porter-Duff blend modes with SIMD acceleration
//! - [`convert`] — Pixel format conversion (RGB↔YUV, ARGB↔NV12, etc.)
//! - [`filter`] — CPU image filters (brightness, contrast, saturation, levels, curves)
//! - [`histogram`] — Luminance and color histogram computation
//!
//! ## Feature Flags
//!
//! - `simd` (default) — SSE2/AVX2/NEON SIMD acceleration for blend and convert
//! - `gpu` — wgpu-based GPU compute pipelines for filters and compositing
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
pub mod convert;
pub mod filter;
pub mod histogram;
pub mod pixel;

mod error;
pub use error::RangaError;

/// Result type alias for ranga operations.
pub type Result<T> = std::result::Result<T, RangaError>;
