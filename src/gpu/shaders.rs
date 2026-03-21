//! WGSL compute shader sources.
//!
//! All shaders operate on packed RGBA8 pixels stored as `array<u32>`.
//! Each `u32` contains `[R, G, B, A]` packed as bytes (R in the lowest byte).
//! This avoids expensive CPU-side f32 conversion before upload — unpacking
//! happens entirely on the GPU.

/// Unpack/pack helpers shared by all shaders (prepended to each shader).
///
/// - `unpack_rgba` extracts a `vec4<f32>` (0.0–1.0) from a packed `u32`.
/// - `pack_rgba` converts a `vec4<f32>` back to a packed `u32` with rounding.
pub const PACK_HELPERS: &str = r#"
fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(c.r * 255.0 + 0.5, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0 + 0.5, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0 + 0.5, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0 + 0.5, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}
"#;

/// All 12 blend modes in one shader, selected by `params.mode`.
///
/// Mode values: 0=Normal, 1=Multiply, 2=Screen, 3=Overlay, 4=Darken, 5=Lighten,
/// 6=ColorDodge, 7=ColorBurn, 8=SoftLight, 9=HardLight, 10=Difference, 11=Exclusion.
///
/// Bindings:
/// - `@binding(0)`: source pixels (`array<u32>`, read-only)
/// - `@binding(1)`: destination pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params (`count: u32`, `mode: u32`, `opacity: f32`, `_pad: u32`)
pub const BLEND_ALL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    count: u32,
    mode: u32,
    opacity: f32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn blend_channel(s: f32, d: f32, mode: u32) -> f32 {
    switch mode {
        case 0u: { return s; }
        case 1u: { return s * d; }
        case 2u: { return 1.0 - (1.0 - s) * (1.0 - d); }
        case 3u: {
            if d < 0.5 { return 2.0 * s * d; }
            else { return 1.0 - 2.0 * (1.0 - s) * (1.0 - d); }
        }
        case 4u: { return min(s, d); }
        case 5u: { return max(s, d); }
        case 6u: {
            if s >= 1.0 { return 1.0; }
            else { return min(d / (1.0 - s), 1.0); }
        }
        case 7u: {
            if s <= 0.0 { return 0.0; }
            else { return 1.0 - min((1.0 - d) / s, 1.0); }
        }
        case 8u: {
            if s <= 0.5 { return d - (1.0 - 2.0 * s) * d * (1.0 - d); }
            else {
                var g: f32;
                if d <= 0.25 { g = ((16.0 * d - 12.0) * d + 4.0) * d; }
                else { g = sqrt(d); }
                return d + (2.0 * s - 1.0) * (g - d);
            }
        }
        case 9u: {
            if s < 0.5 { return 2.0 * s * d; }
            else { return 1.0 - 2.0 * (1.0 - s) * (1.0 - d); }
        }
        case 10u: { return abs(s - d); }
        case 11u: { return s + d - 2.0 * s * d; }
        default: { return s; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }

    let s = unpack_rgba(src[idx]);
    let d = unpack_rgba(dst[idx]);

    let sa = s.a * params.opacity;
    if sa <= 0.0 { return; }

    let br = blend_channel(s.r, d.r, params.mode);
    let bg = blend_channel(s.g, d.g, params.mode);
    let bb = blend_channel(s.b, d.b, params.mode);

    let inv_sa = 1.0 - sa;
    let out_r = clamp(br * sa + d.r * inv_sa, 0.0, 1.0);
    let out_g = clamp(bg * sa + d.g * inv_sa, 0.0, 1.0);
    let out_b = clamp(bb * sa + d.b * inv_sa, 0.0, 1.0);
    let out_a = clamp(sa + d.a * inv_sa, 0.0, 1.0);

    dst[idx] = pack_rgba(vec4<f32>(out_r, out_g, out_b, out_a));
}
"#;

/// Invert colors shader.
///
/// Inverts R, G, B channels while preserving alpha.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`)
pub const INVERT: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params { count: u32, }
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    pixels[idx] = pack_rgba(vec4<f32>(1.0 - px.r, 1.0 - px.g, 1.0 - px.b, px.a));
}
"#;

/// Grayscale conversion shader (BT.709 luminance).
///
/// Sets R, G, B to the BT.709 luminance value while preserving alpha.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`)
pub const GRAYSCALE: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params { count: u32, }
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    let lum = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;
    pixels[idx] = pack_rgba(vec4<f32>(lum, lum, lum, px.a));
}
"#;

/// Brightness and contrast adjustment shader.
///
/// Applies brightness offset and contrast scaling per channel.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`, `brightness: f32`, `contrast: f32`, `_pad: u32`)
pub const BRIGHTNESS_CONTRAST: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params {
    count: u32,
    brightness: f32,
    contrast: f32,
    _pad: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    let b = params.brightness;
    let c = params.contrast;
    let r = clamp((px.r + b - 0.5) * c + 0.5, 0.0, 1.0);
    let g = clamp((px.g + b - 0.5) * c + 0.5, 0.0, 1.0);
    let blue = clamp((px.b + b - 0.5) * c + 0.5, 0.0, 1.0);
    pixels[idx] = pack_rgba(vec4<f32>(r, g, blue, px.a));
}
"#;

/// Saturation adjustment shader.
///
/// Adjusts saturation using BT.601 luminance coefficients.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`, `factor: f32`, `_pad1: u32`, `_pad2: u32`)
pub const SATURATION: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params {
    count: u32,
    factor: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    let lum = 0.299 * px.r + 0.587 * px.g + 0.114 * px.b;
    let f = params.factor;
    let r = clamp(lum + f * (px.r - lum), 0.0, 1.0);
    let g = clamp(lum + f * (px.g - lum), 0.0, 1.0);
    let b = clamp(lum + f * (px.b - lum), 0.0, 1.0);
    pixels[idx] = pack_rgba(vec4<f32>(r, g, b, px.a));
}
"#;

/// Horizontal Gaussian blur pass.
///
/// Performs a 1D horizontal convolution using a pre-computed kernel. Each thread
/// processes one pixel, summing weighted samples across its row.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: kernel weights (`array<f32>`, read-only storage)
/// - `@binding(3)`: uniform params (`width: u32`, `height: u32`, `radius: u32`, `_pad: u32`)
pub const BLUR_HORIZONTAL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;

struct Params {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let r = i32(params.radius);
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = -r; i <= r; i = i + 1) {
        let sx = clamp(i32(x) + i, 0, i32(params.width) - 1);
        let idx = u32(sx) + y * params.width;
        let w = kernel[u32(i + r)];
        acc = acc + unpack_rgba(input[idx]) * w;
    }
    output[x + y * params.width] = pack_rgba(acc);
}
"#;

/// Vertical Gaussian blur pass.
///
/// Performs a 1D vertical convolution using a pre-computed kernel. Each thread
/// processes one pixel, summing weighted samples down its column.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: kernel weights (`array<f32>`, read-only storage)
/// - `@binding(3)`: uniform params (`width: u32`, `height: u32`, `radius: u32`, `_pad: u32`)
pub const BLUR_VERTICAL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;

struct Params {
    width: u32,
    height: u32,
    radius: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let r = i32(params.radius);
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = -r; i <= r; i = i + 1) {
        let sy = clamp(i32(y) + i, 0, i32(params.height) - 1);
        let idx = x + u32(sy) * params.width;
        let w = kernel[u32(i + r)];
        acc = acc + unpack_rgba(input[idx]) * w;
    }
    output[x + y * params.width] = pack_rgba(acc);
}
"#;

/// RGBA8 to YUV420p Y-plane extraction (BT.601).
///
/// Extracts the Y (luma) plane from packed RGBA8 pixels. Processes 4 Y values
/// per thread, packing them into a single `u32`.
///
/// Bindings:
/// - `@binding(0)`: source pixels (`array<u32>`, read-only)
/// - `@binding(1)`: Y plane output (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params (`pixel_count: u32`)
#[allow(dead_code)]
pub const RGBA_TO_Y_BT601: &str = r#"
@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> y_plane: array<u32>;

struct Params {
    pixel_count: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    if base >= params.pixel_count { return; }
    var packed_y: u32 = 0u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let idx = base + i;
        if idx >= params.pixel_count { break; }
        let px = unpack_rgba(pixels[idx]);
        let y = u32(clamp(px.r * 0.299 + px.g * 0.587 + px.b * 0.114, 0.0, 1.0) * 255.0 + 0.5);
        packed_y = packed_y | (y << (i * 8u));
    }
    y_plane[gid.x] = packed_y;
}
"#;

/// Build a complete shader by prepending the pack/unpack helpers.
///
/// All shaders require the `unpack_rgba` and `pack_rgba` functions. This
/// helper concatenates them with the shader body.
///
/// # Examples
///
/// ```ignore
/// let full = build_shader("// my shader body");
/// assert!(full.contains("unpack_rgba"));
/// assert!(full.contains("// my shader body"));
/// ```
#[must_use]
pub fn build_shader(body: &str) -> String {
    format!("{}\n{}", PACK_HELPERS, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_shader_prepends_helpers() {
        let shader = build_shader("fn main() {}");
        assert!(shader.contains("unpack_rgba"));
        assert!(shader.contains("pack_rgba"));
        assert!(shader.contains("fn main() {}"));
    }

    #[test]
    fn pack_helpers_contain_both_functions() {
        assert!(PACK_HELPERS.contains("fn unpack_rgba"));
        assert!(PACK_HELPERS.contains("fn pack_rgba"));
    }

    #[test]
    fn blend_shader_has_all_modes() {
        assert!(BLEND_ALL.contains("case 0u"));
        assert!(BLEND_ALL.contains("case 11u"));
        assert!(BLEND_ALL.contains("blend_channel"));
    }

    #[test]
    fn invert_shader_preserves_alpha() {
        assert!(INVERT.contains("px.a"));
        assert!(INVERT.contains("1.0 - px.r"));
    }

    #[test]
    fn blur_horizontal_shader_has_kernel_binding() {
        assert!(BLUR_HORIZONTAL.contains("var<storage, read> kernel"));
        assert!(BLUR_HORIZONTAL.contains("@workgroup_size(16, 16)"));
    }

    #[test]
    fn blur_vertical_shader_has_kernel_binding() {
        assert!(BLUR_VERTICAL.contains("var<storage, read> kernel"));
        assert!(BLUR_VERTICAL.contains("@workgroup_size(16, 16)"));
    }
}
