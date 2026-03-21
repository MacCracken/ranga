# Color Science Guide

## Color Types

| Type | Space | Components | Range |
|------|-------|-----------|-------|
| `Srgba` | sRGB | R, G, B, A | 0–255 (u8) |
| `LinRgba` | Linear RGB | R, G, B, A | 0.0–1.0 (f32) |
| `Hsl` | HSL | H, S, L | H: 0–360, S/L: 0–1 |
| `CieXyz` | CIE XYZ (D65) | X, Y, Z | f64 |
| `CieLab` | CIE L*a*b* (D65) | L, a, b | L: 0–100, a/b: unbounded |
| `Cmyk` | CMYK | C, M, Y, K | 0.0–1.0 (f32) |

## Conversion Chains

```
Srgba ↔ LinRgba ↔ CieXyz ↔ CieLab
Srgba ↔ Hsl
Srgba ↔ Cmyk (naive)
LinRgba ↔ Display P3 (via matrix)
```

All conversions use `From` trait impls for ergonomic chaining:

```rust
let lab: CieLab = Srgba { r: 128, g: 64, b: 200, a: 255 }.into();
let hsl: Hsl = Srgba { r: 255, g: 0, b: 0, a: 255 }.into();
let back: Srgba = hsl.into(); // bidirectional
```

## Display P3

Convert between sRGB and Display P3 in linear space:

```rust
let (r, g, b) = p3_to_linear_srgb(0.8, 0.3, 0.5);
let (pr, pg, pb) = linear_srgb_to_p3(0.5, 0.3, 0.8);
```

P3 red (1,0,0) maps to sRGB values > 1.0 — it exceeds the sRGB gamut.

## Color Distance (Delta-E)

Three metrics for comparing colors in perceptual space:

| Metric | Speed | Accuracy | Use case |
|--------|-------|----------|----------|
| `delta_e_cie76` | ~2 ns | Low | Quick screening |
| `delta_e_cie94` | ~10 ns | Medium | Graphic arts QA |
| `delta_e_ciede2000` | ~109 ns | High | Precision color matching |

A Delta-E of 1.0 is approximately the smallest difference a human can perceive.

## Color Temperature

Map a Kelvin value to RGB multipliers for white balance:

```rust
let [r, g, b] = color_temperature(3200.0); // warm tungsten
let [r, g, b] = color_temperature(6500.0); // daylight (neutral)
let [r, g, b] = color_temperature(10000.0); // cool overcast
```

Multiply these with pixel values to shift white balance.

## CMYK

Naive conversion (no ICC profile):

```rust
let cmyk = srgb_to_cmyk(&srgba);
let back = cmyk_to_srgb(&cmyk);
```

For print-accurate CMYK, use the `icc` module with an ICC profile.

## ICC Profiles

Parse matrix-based ICC v2/v4 profiles:

```rust
let profile = IccProfile::from_bytes(&profile_data)?;
let (x, y, z) = profile.apply(0.5, 0.3, 0.8); // RGB → XYZ via profile matrix + TRC
```

Supports gamma and table-based tone response curves.
