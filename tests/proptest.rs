use proptest::prelude::*;
use ranga::blend::{BlendMode, blend_pixel};
use ranga::color::*;
use ranga::convert;
use ranga::pixel::{PixelBuffer, PixelFormat};

// ---------------------------------------------------------------------------
// Color conversion roundtrips
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn srgb_linear_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let orig = Srgba { r, g, b, a: 255 };
        let lin: LinRgba = orig.into();
        let back: Srgba = lin.into();
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1);
    }

    #[test]
    fn hsl_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let orig = Srgba { r, g, b, a: 255 };
        let hsl: Hsl = orig.into();
        let back: Srgba = hsl.into();
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1,
            "r: {} vs {}", orig.r, back.r);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1,
            "g: {} vs {}", orig.g, back.g);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1,
            "b: {} vs {}", orig.b, back.b);
    }

    #[test]
    fn lab_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let orig = Srgba { r, g, b, a: 255 };
        let lab: CieLab = orig.into();
        let xyz: CieXyz = lab.into();
        let lin: LinRgba = xyz.into();
        let back: Srgba = lin.into();
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1,
            "r: {} vs {}", orig.r, back.r);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1,
            "g: {} vs {}", orig.g, back.g);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1,
            "b: {} vs {}", orig.b, back.b);
    }

    #[test]
    fn cmyk_roundtrip(r in 1u8..=255, g in 1u8..=255, b in 1u8..=255) {
        // Skip pure black (CMYK roundtrip is exact only for non-black)
        let orig = Srgba { r, g, b, a: 255 };
        let cmyk = srgb_to_cmyk(&orig);
        let back = cmyk_to_srgb(&cmyk);
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1,
            "r: {} vs {}", orig.r, back.r);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1,
            "g: {} vs {}", orig.g, back.g);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1,
            "b: {} vs {}", orig.b, back.b);
    }

    #[test]
    fn p3_srgb_roundtrip(
        r in 0.0f64..=1.0,
        g in 0.0f64..=1.0,
        b in 0.0f64..=1.0
    ) {
        let (pr, pg, pb) = linear_srgb_to_p3(r, g, b);
        let (rr, gg, bb) = p3_to_linear_srgb(pr, pg, pb);
        prop_assert!((r - rr).abs() < 1e-4, "r: {} vs {}", r, rr);
        prop_assert!((g - gg).abs() < 1e-4, "g: {} vs {}", g, gg);
        prop_assert!((b - bb).abs() < 1e-4, "b: {} vs {}", b, bb);
    }
}

// ---------------------------------------------------------------------------
// Delta-E properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn delta_e_76_non_negative(
        l1 in 0.0f64..=100.0, a1 in -128.0f64..=128.0, b1 in -128.0f64..=128.0,
        l2 in 0.0f64..=100.0, a2 in -128.0f64..=128.0, b2 in -128.0f64..=128.0,
    ) {
        let c1 = CieLab { l: l1, a: a1, b: b1 };
        let c2 = CieLab { l: l2, a: a2, b: b2 };
        prop_assert!(delta_e_cie76(&c1, &c2) >= 0.0);
    }

    #[test]
    fn delta_e_76_identity(l in 0.0f64..=100.0, a in -128.0f64..=128.0, b_val in -128.0f64..=128.0) {
        let c = CieLab { l, a, b: b_val };
        prop_assert!(delta_e_cie76(&c, &c) < 1e-10);
    }

    #[test]
    fn delta_e_76_symmetric(
        l1 in 0.0f64..=100.0, a1 in -128.0f64..=128.0, b1 in -128.0f64..=128.0,
        l2 in 0.0f64..=100.0, a2 in -128.0f64..=128.0, b2 in -128.0f64..=128.0,
    ) {
        let c1 = CieLab { l: l1, a: a1, b: b1 };
        let c2 = CieLab { l: l2, a: a2, b: b2 };
        let d1 = delta_e_cie76(&c1, &c2);
        let d2 = delta_e_cie76(&c2, &c1);
        prop_assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn delta_e_2000_non_negative(
        l1 in 0.0f64..=100.0, a1 in -128.0f64..=128.0, b1 in -128.0f64..=128.0,
        l2 in 0.0f64..=100.0, a2 in -128.0f64..=128.0, b2 in -128.0f64..=128.0,
    ) {
        let c1 = CieLab { l: l1, a: a1, b: b1 };
        let c2 = CieLab { l: l2, a: a2, b: b2 };
        let de = delta_e_ciede2000(&c1, &c2);
        prop_assert!(de >= 0.0, "delta_e_ciede2000 returned {}", de);
    }
}

// ---------------------------------------------------------------------------
// Blend mode properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn blend_transparent_src_is_noop(
        dr in 0u8..=255, dg in 0u8..=255, db in 0u8..=255, da in 0u8..=255,
        sr in 0u8..=255, sg in 0u8..=255, sb in 0u8..=255,
        opacity in 0u8..=255,
    ) {
        // Source with alpha=0 should leave dest unchanged regardless of mode
        let dst = [dr, dg, db, da];
        let src = [sr, sg, sb, 0];
        let result = blend_pixel(src, dst, BlendMode::Normal, opacity);
        prop_assert_eq!(result, dst);
    }

    #[test]
    fn blend_all_modes_no_panic(
        sr in 0u8..=255, sg in 0u8..=255, sb in 0u8..=255, sa in 0u8..=255,
        dr in 0u8..=255, dg in 0u8..=255, db in 0u8..=255, da in 0u8..=255,
        opacity in 0u8..=255,
    ) {
        let src = [sr, sg, sb, sa];
        let dst = [dr, dg, db, da];
        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen,
            BlendMode::Overlay, BlendMode::Darken, BlendMode::Lighten,
            BlendMode::ColorDodge, BlendMode::ColorBurn, BlendMode::SoftLight,
            BlendMode::HardLight, BlendMode::Difference, BlendMode::Exclusion,
        ];
        for mode in modes {
            let _ = blend_pixel(src, dst, mode, opacity);
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel format conversion roundtrips
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn rgb8_rgba8_roundtrip(pixels in proptest::collection::vec(0u8..=255, 12..=300)) {
        // Trim to multiple of 3 for valid RGB8 data
        let len = pixels.len() / 3 * 3;
        if len < 3 { return Ok(()); }
        let data = pixels[..len].to_vec();
        let w = (len / 3) as u32;
        let buf = PixelBuffer::new(data.clone(), w, 1, PixelFormat::Rgb8).unwrap();
        let rgba = convert::rgb8_to_rgba8(&buf).unwrap();
        let back = convert::rgba8_to_rgb8(&rgba).unwrap();
        prop_assert_eq!(&back.data, &data);
    }

    #[test]
    fn argb8_rgba8_roundtrip(pixels in proptest::collection::vec(0u8..=255, 4..=400)) {
        let len = pixels.len() / 4 * 4;
        if len < 4 { return Ok(()); }
        let data = pixels[..len].to_vec();
        let w = (len / 4) as u32;
        let buf = PixelBuffer::new(data.clone(), w, 1, PixelFormat::Argb8).unwrap();
        let rgba = convert::argb8_to_rgba8(&buf).unwrap();
        let back = convert::rgba8_to_argb8(&rgba).unwrap();
        prop_assert_eq!(&back.data, &data);
    }

    #[test]
    fn yuv420p_bt601_roundtrip_gray(v in 16u8..=235) {
        // Uniform gray should roundtrip well since chroma is neutral
        let data: Vec<u8> = [v, v, v, 255].repeat(16);
        let buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = convert::rgba_to_yuv420p(&buf).unwrap();
        let back = convert::yuv420p_to_rgba(&yuv).unwrap();
        prop_assert!((back.data[0] as i16 - v as i16).unsigned_abs() < 3,
            "in={} out={}", v, back.data[0]);
    }
}

// ---------------------------------------------------------------------------
// Color temperature properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn color_temperature_in_range(kelvin in 1000.0f32..=40000.0) {
        let [r, g, b] = color_temperature(kelvin);
        prop_assert!((0.0..=1.0).contains(&r), "r={r}");
        prop_assert!((0.0..=1.0).contains(&g), "g={g}");
        prop_assert!((0.0..=1.0).contains(&b), "b={b}");
    }
}
