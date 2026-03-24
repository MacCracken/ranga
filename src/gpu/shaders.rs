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

/// 3D LUT application shader with trilinear interpolation.
///
/// Applies a 3D color lookup table to each pixel. The LUT is stored as a
/// flattened array of f32 RGBA values (`size^3 * 4` entries). R/G/B map to
/// the LUT axes.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: LUT data (`array<f32>`, read-only storage)
/// - `@binding(2)`: uniform params (`count: u32`, `lut_size: u32`, `_pad1: u32`, `_pad2: u32`)
#[allow(dead_code)]
pub const LUT3D: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;

struct Params {
    count: u32,
    lut_size: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn lut_sample(r: u32, g: u32, b: u32) -> vec4<f32> {
    let s = params.lut_size;
    let idx = (r + g * s + b * s * s) * 4u;
    return vec4<f32>(lut[idx], lut[idx + 1u], lut[idx + 2u], lut[idx + 3u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    let s = f32(params.lut_size - 1u);

    let rf = px.r * s;
    let gf = px.g * s;
    let bf = px.b * s;

    let r0 = u32(floor(rf));
    let g0 = u32(floor(gf));
    let b0 = u32(floor(bf));
    let r1 = min(r0 + 1u, params.lut_size - 1u);
    let g1 = min(g0 + 1u, params.lut_size - 1u);
    let b1 = min(b0 + 1u, params.lut_size - 1u);

    let fr = rf - floor(rf);
    let fg = gf - floor(gf);
    let fb = bf - floor(bf);

    // Trilinear interpolation.
    let c000 = lut_sample(r0, g0, b0);
    let c100 = lut_sample(r1, g0, b0);
    let c010 = lut_sample(r0, g1, b0);
    let c110 = lut_sample(r1, g1, b0);
    let c001 = lut_sample(r0, g0, b1);
    let c101 = lut_sample(r1, g0, b1);
    let c011 = lut_sample(r0, g1, b1);
    let c111 = lut_sample(r1, g1, b1);

    let c00 = mix(c000, c100, fr);
    let c10 = mix(c010, c110, fr);
    let c01 = mix(c001, c101, fr);
    let c11 = mix(c011, c111, fr);
    let c0 = mix(c00, c10, fg);
    let c1 = mix(c01, c11, fg);
    let result = mix(c0, c1, fb);

    pixels[idx] = pack_rgba(vec4<f32>(
        clamp(result.r, 0.0, 1.0),
        clamp(result.g, 0.0, 1.0),
        clamp(result.b, 0.0, 1.0),
        px.a
    ));
}
"#;

/// Hue shift shader.
///
/// Converts RGB → HSL, adds the shift to hue, converts back to RGB.
/// Preserves alpha.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`, `shift: f32`, `_pad1: u32`, `_pad2: u32`)
#[allow(dead_code)]
pub const HUE_SHIFT: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params {
    count: u32,
    shift: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(max(c.r, c.g), c.b);
    let mn = min(min(c.r, c.g), c.b);
    let d = mx - mn;
    let l = (mx + mn) * 0.5;
    if d < 0.00001 { return vec3<f32>(0.0, 0.0, l); }
    var s: f32;
    if l > 0.5 { s = d / (2.0 - mx - mn); }
    else { s = d / (mx + mn); }
    var h: f32;
    if abs(mx - c.r) < 0.00001 {
        h = (c.g - c.b) / d;
        if c.g < c.b { h = h + 6.0; }
    } else if abs(mx - c.g) < 0.00001 {
        h = (c.b - c.r) / d + 2.0;
    } else {
        h = (c.r - c.g) / d + 4.0;
    }
    return vec3<f32>(h / 6.0, s, l);
}

fn hue2rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if t < 0.0 { t = t + 1.0; }
    if t > 1.0 { t = t - 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    if hsl.y < 0.00001 { return vec3<f32>(hsl.z, hsl.z, hsl.z); }
    var q: f32;
    if hsl.z < 0.5 { q = hsl.z * (1.0 + hsl.y); }
    else { q = hsl.z + hsl.y - hsl.z * hsl.y; }
    let p = 2.0 * hsl.z - q;
    return vec3<f32>(
        hue2rgb(p, q, hsl.x + 1.0 / 3.0),
        hue2rgb(p, q, hsl.x),
        hue2rgb(p, q, hsl.x - 1.0 / 3.0)
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    var hsl = rgb_to_hsl(vec3<f32>(px.r, px.g, px.b));
    hsl.x = fract(hsl.x + params.shift / 360.0);
    let rgb = hsl_to_rgb(hsl);
    pixels[idx] = pack_rgba(vec4<f32>(
        clamp(rgb.r, 0.0, 1.0),
        clamp(rgb.g, 0.0, 1.0),
        clamp(rgb.b, 0.0, 1.0),
        px.a
    ));
}
"#;

/// Color balance shader (shadows/midtones/highlights adjustment).
///
/// Uses luminance-based weighting: shadows affect dark pixels, midtones
/// affect mid-range, and highlights affect bright pixels.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params
#[allow(dead_code)]
pub const COLOR_BALANCE: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params {
    count: u32,
    shadow_r: f32,
    shadow_g: f32,
    shadow_b: f32,
    mid_r: f32,
    mid_g: f32,
    mid_b: f32,
    high_r: f32,
    high_g: f32,
    high_b: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);
    let lum = 0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b;

    // Weight functions.
    let shadow_w = clamp(1.0 - lum * 4.0, 0.0, 1.0);
    let high_w = clamp(lum * 4.0 - 3.0, 0.0, 1.0);
    let mid_w = 1.0 - shadow_w - high_w;

    let adj_r = shadow_w * params.shadow_r + mid_w * params.mid_r + high_w * params.high_r;
    let adj_g = shadow_w * params.shadow_g + mid_w * params.mid_g + high_w * params.high_g;
    let adj_b = shadow_w * params.shadow_b + mid_w * params.mid_b + high_w * params.high_b;

    pixels[idx] = pack_rgba(vec4<f32>(
        clamp(px.r + adj_r, 0.0, 1.0),
        clamp(px.g + adj_g, 0.0, 1.0),
        clamp(px.b + adj_b, 0.0, 1.0),
        px.a
    ));
}
"#;

/// Gaussian noise generation shader using PCG hash.
///
/// Applies Gaussian-distributed noise to R, G, B channels using a PCG hash
/// for deterministic pseudo-random generation and Box-Muller transform for
/// Gaussian distribution. Preserves alpha.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`, `seed: u32`, `strength: f32`, `_pad: u32`)
pub const NOISE_GAUSSIAN: &str = r#"
@group(0) @binding(0) var<storage, read_write> pixels: array<u32>;

struct Params {
    count: u32,
    seed: u32,
    strength: f32,
    _pad: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_uniform(seed_val: u32) -> f32 {
    return f32(pcg(seed_val)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }
    let px = unpack_rgba(pixels[idx]);

    // Generate two uniform random values for Box-Muller
    let u1 = max(rand_uniform(idx * 2u + params.seed), 0.0001);
    let u2 = rand_uniform(idx * 2u + params.seed + 1u);

    // Box-Muller transform for Gaussian distribution
    let mag = sqrt(-2.0 * log(u1)) * params.strength;
    let angle = 6.283185307 * u2;
    let g1 = mag * cos(angle);
    let g2 = mag * sin(angle);

    // Apply noise using g1 for R/B and g2 for G
    let r = clamp(px.r + g1, 0.0, 1.0);
    let g = clamp(px.g + g2, 0.0, 1.0);
    let b = clamp(px.b + g1 * 0.7 + g2 * 0.3, 0.0, 1.0);

    pixels[idx] = pack_rgba(vec4<f32>(r, g, b, px.a));
}
"#;

/// Cross-dissolve transition between two buffers.
///
/// Linearly interpolates between source and destination pixels.
/// `factor` 0.0 = all source, 1.0 = all destination (original).
///
/// Bindings:
/// - `@binding(0)`: source pixels (`array<u32>`, read-only)
/// - `@binding(1)`: destination pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params (`count: u32`, `factor_bits: u32`, `_pad1: u32`, `_pad2: u32`)
pub const DISSOLVE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    count: u32,
    factor: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }

    let s = unpack_rgba(src[idx]);
    let d = unpack_rgba(dst[idx]);
    let f = params.factor;

    dst[idx] = pack_rgba(mix(s, d, vec4<f32>(f, f, f, f)));
}
"#;

/// Fade toward black shader.
///
/// Multiplies R, G, B channels by the fade factor while preserving alpha.
/// `factor` 0.0 = black, 1.0 = identity.
///
/// Bindings:
/// - `@binding(0)`: pixels (`array<u32>`, read-write)
/// - `@binding(1)`: uniform params (`count: u32`, `factor: f32`, `_pad1: u32`, `_pad2: u32`)
pub const FADE: &str = r#"
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
    let f = params.factor;
    pixels[idx] = pack_rgba(vec4<f32>(px.r * f, px.g * f, px.b * f, px.a));
}
"#;

/// Horizontal wipe transition between two buffers.
///
/// Pixels left of the wipe line come from destination, pixels right come from
/// source. `progress` 0.0 = all source, 1.0 = all destination.
///
/// Bindings:
/// - `@binding(0)`: source pixels (`array<u32>`, read-only)
/// - `@binding(1)`: destination pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params (`count: u32`, `width: u32`, `height: u32`, `progress: f32`)
pub const WIPE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    count: u32,
    width: u32,
    height: u32,
    progress: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count { return; }

    let x = idx % params.width;
    let wipe_x = u32(f32(params.width) * params.progress);

    if x < wipe_x {
        // Keep dst as-is (already in dst buffer)
    } else {
        dst[idx] = src[idx];
    }
}
"#;

/// Crop shader — copies a rectangular region from input to output.
///
/// Each thread copies one output pixel, reading from the corresponding
/// position in the source (offset by `src_x`, `src_y`).
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params
pub const CROP: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    src_width: u32,
    dst_width: u32,
    dst_height: u32,
    src_x: u32,
    src_y: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= params.dst_width || gy >= params.dst_height { return; }

    let src_idx = (params.src_y + gy) * params.src_width + (params.src_x + gx);
    let dst_idx = gy * params.dst_width + gx;
    output[dst_idx] = input[src_idx];
}
"#;

/// Bilinear interpolation resize shader.
///
/// Each thread computes one output pixel using bilinear sampling from input.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params
pub const RESIZE_BILINEAR: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= params.dst_width || gy >= params.dst_height { return; }

    let sw = f32(params.src_width);
    let sh = f32(params.src_height);
    let dw = f32(params.dst_width);
    let dh = f32(params.dst_height);

    var sx: f32;
    var sy: f32;
    if params.dst_width > 1u {
        sx = f32(gx) * (sw - 1.0) / (dw - 1.0);
    } else {
        sx = 0.0;
    }
    if params.dst_height > 1u {
        sy = f32(gy) * (sh - 1.0) / (dh - 1.0);
    } else {
        sy = 0.0;
    }

    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, params.src_width - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);

    let c00 = unpack_rgba(input[y0 * params.src_width + x0]);
    let c10 = unpack_rgba(input[y0 * params.src_width + x1]);
    let c01 = unpack_rgba(input[y1 * params.src_width + x0]);
    let c11 = unpack_rgba(input[y1 * params.src_width + x1]);

    let top = mix(c00, c10, fx);
    let bot = mix(c01, c11, fx);
    let result = mix(top, bot, fy);

    output[gy * params.dst_width + gx] = pack_rgba(result);
}
"#;

/// Nearest-neighbor resize shader.
///
/// Each thread copies one output pixel from the nearest source pixel.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params
pub const RESIZE_NEAREST: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = gid.x;
    let gy = gid.y;
    if gx >= params.dst_width || gy >= params.dst_height { return; }

    let sx = min(gx * params.src_width / params.dst_width, params.src_width - 1u);
    let sy = min(gy * params.src_height / params.dst_height, params.src_height - 1u);

    output[gy * params.dst_width + gx] = input[sy * params.src_width + sx];
}
"#;

/// Horizontal flip shader — mirrors pixels left-to-right.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params
pub const FLIP_HORIZONTAL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let src_idx = y * params.width + x;
    let dst_idx = y * params.width + (params.width - 1u - x);
    output[dst_idx] = input[src_idx];
}
"#;

/// Vertical flip shader — mirrors pixels top-to-bottom.
///
/// Bindings:
/// - `@binding(0)`: input pixels (`array<u32>`, read-only)
/// - `@binding(1)`: output pixels (`array<u32>`, read-write)
/// - `@binding(2)`: uniform params
pub const FLIP_VERTICAL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    width: u32,
    height: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let src_idx = y * params.width + x;
    let dst_idx = (params.height - 1u - y) * params.width + x;
    output[dst_idx] = input[src_idx];
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

    #[test]
    fn lut3d_shader_has_trilinear() {
        assert!(LUT3D.contains("lut_sample"));
        assert!(LUT3D.contains("mix"));
        assert!(LUT3D.contains("lut_size"));
    }

    #[test]
    fn hue_shift_shader_has_hsl() {
        assert!(HUE_SHIFT.contains("rgb_to_hsl"));
        assert!(HUE_SHIFT.contains("hsl_to_rgb"));
        assert!(HUE_SHIFT.contains("params.shift"));
    }

    #[test]
    fn color_balance_shader_has_weights() {
        assert!(COLOR_BALANCE.contains("shadow_w"));
        assert!(COLOR_BALANCE.contains("mid_w"));
        assert!(COLOR_BALANCE.contains("high_w"));
        assert!(COLOR_BALANCE.contains("shadow_r"));
    }

    #[test]
    fn noise_shader_has_pcg() {
        assert!(NOISE_GAUSSIAN.contains("pcg"));
        assert!(NOISE_GAUSSIAN.contains("Box-Muller"));
        assert!(NOISE_GAUSSIAN.contains("params.strength"));
    }

    #[test]
    fn dissolve_shader_has_mix() {
        assert!(DISSOLVE.contains("mix"));
        assert!(DISSOLVE.contains("params.factor"));
    }

    #[test]
    fn fade_shader_preserves_alpha() {
        assert!(FADE.contains("px.a"));
        assert!(FADE.contains("params.factor"));
    }

    #[test]
    fn wipe_shader_has_width() {
        assert!(WIPE.contains("width"));
        assert!(WIPE.contains("params.progress"));
    }

    #[test]
    fn crop_shader_has_src_offset() {
        assert!(CROP.contains("src_x"));
        assert!(CROP.contains("src_y"));
        assert!(CROP.contains("src_width"));
        assert!(CROP.contains("@workgroup_size(16, 16)"));
    }

    #[test]
    fn resize_bilinear_shader_has_mix() {
        assert!(RESIZE_BILINEAR.contains("mix"));
        assert!(RESIZE_BILINEAR.contains("unpack_rgba"));
        assert!(RESIZE_BILINEAR.contains("src_width"));
        assert!(RESIZE_BILINEAR.contains("dst_width"));
    }

    #[test]
    fn resize_nearest_shader_has_sampling() {
        assert!(RESIZE_NEAREST.contains("src_width"));
        assert!(RESIZE_NEAREST.contains("dst_width"));
        assert!(RESIZE_NEAREST.contains("@workgroup_size(16, 16)"));
    }

    #[test]
    fn flip_horizontal_shader_mirrors() {
        assert!(FLIP_HORIZONTAL.contains("width - 1u - x"));
        assert!(FLIP_HORIZONTAL.contains("@workgroup_size(16, 16)"));
    }

    #[test]
    fn flip_vertical_shader_mirrors() {
        assert!(FLIP_VERTICAL.contains("height - 1u - y"));
        assert!(FLIP_VERTICAL.contains("@workgroup_size(16, 16)"));
    }
}
