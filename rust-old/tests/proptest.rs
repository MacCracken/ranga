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
        prop_assert_eq!(back.data(), &data[..]);
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
        prop_assert_eq!(back.data(), &data[..]);
    }

    #[test]
    fn yuv420p_bt601_roundtrip_gray(v in 16u8..=235) {
        // Uniform gray should roundtrip well since chroma is neutral
        let data: Vec<u8> = [v, v, v, 255].repeat(16);
        let buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = convert::rgba_to_yuv420p(&buf).unwrap();
        let back = convert::yuv420p_to_rgba(&yuv).unwrap();
        prop_assert!((back.data()[0] as i16 - v as i16).unsigned_abs() < 3,
            "in={} out={}", v, back.data()[0]);
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

// ---------------------------------------------------------------------------
// Transform properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn crop_never_exceeds_source(
        w in 1u32..=100, h in 1u32..=100,
        l in 0u32..=100, t in 0u32..=100,
        r in 0u32..=200, b in 0u32..=200,
    ) {
        let buf = PixelBuffer::zeroed(w, h, PixelFormat::Rgba8);
        let cropped = ranga::transform::crop(&buf, l, t, r, b).unwrap();
        prop_assert!(cropped.width() <= w);
        prop_assert!(cropped.height() <= h);
    }

    #[test]
    fn resize_produces_correct_dimensions(
        w in 1u32..=50, h in 1u32..=50,
        nw in 1u32..=100, nh in 1u32..=100,
    ) {
        let buf = PixelBuffer::zeroed(w, h, PixelFormat::Rgba8);
        let resized = ranga::transform::resize(&buf, nw, nh, ranga::transform::ScaleFilter::Nearest).unwrap();
        prop_assert_eq!(resized.width(), nw);
        prop_assert_eq!(resized.height(), nh);
    }

    #[test]
    fn flip_h_is_involution(
        w in 1u32..=20, h in 1u32..=20,
    ) {
        let data: Vec<u8> = (0..(w * h * 4) as u8).collect();
        if let Ok(buf) = PixelBuffer::new(data.clone(), w, h, PixelFormat::Rgba8) {
            let f1 = ranga::transform::flip_horizontal(&buf).unwrap();
            let f2 = ranga::transform::flip_horizontal(&f1).unwrap();
            prop_assert_eq!(f2.data(), &data[..]);
        }
    }

    #[test]
    fn affine_identity_preserves(x in -100.0f64..=100.0, y in -100.0f64..=100.0) {
        let (ox, oy) = ranga::transform::Affine::IDENTITY.apply(x, y);
        prop_assert!((ox - x).abs() < 1e-10);
        prop_assert!((oy - y).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Composite properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn dissolve_at_zero_is_a(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255,
    ) {
        let a = PixelBuffer::new(vec![r, g, b, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let b_buf = PixelBuffer::new(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let result = ranga::composite::dissolve(&a, &b_buf, 0.0).unwrap();
        prop_assert_eq!(result.data()[0], r);
        prop_assert_eq!(result.data()[1], g);
        prop_assert_eq!(result.data()[2], b);
    }

    #[test]
    fn dissolve_at_one_is_b(
        rv in 0u8..=255, gv in 0u8..=255, bv in 0u8..=255,
    ) {
        let a_buf = PixelBuffer::new(vec![0, 0, 0, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let b_buf = PixelBuffer::new(vec![rv, gv, bv, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        let result = ranga::composite::dissolve(&a_buf, &b_buf, 1.0).unwrap();
        prop_assert_eq!(result.data()[0], rv);
        prop_assert_eq!(result.data()[1], gv);
        prop_assert_eq!(result.data()[2], bv);
    }

    #[test]
    fn premultiply_opaque_is_identity(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255,
    ) {
        let mut buf = PixelBuffer::new(vec![r, g, b, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        ranga::composite::premultiply_alpha(&mut buf).unwrap();
        prop_assert_eq!(buf.data()[0], r);
        prop_assert_eq!(buf.data()[1], g);
        prop_assert_eq!(buf.data()[2], b);
    }

    #[test]
    fn threshold_is_binary(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255, t in 0u8..=255,
    ) {
        let mut buf = PixelBuffer::new(vec![r, g, b, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        ranga::filter::threshold(&mut buf, t).unwrap();
        // Result must be either 0 or 255
        prop_assert!(buf.data()[0] == 0 || buf.data()[0] == 255);
        // R == G == B (grayscale output)
        prop_assert_eq!(buf.data()[0], buf.data()[1]);
        prop_assert_eq!(buf.data()[1], buf.data()[2]);
    }

    #[test]
    fn channel_mixer_identity_preserves(
        r in 0u8..=255, g in 0u8..=255, b in 0u8..=255,
    ) {
        let mut buf = PixelBuffer::new(vec![r, g, b, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        ranga::filter::channel_mixer(
            &mut buf,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ).unwrap();
        prop_assert_eq!(buf.data()[0], r);
        prop_assert_eq!(buf.data()[1], g);
        prop_assert_eq!(buf.data()[2], b);
    }
}

// ---------------------------------------------------------------------------
// Oklab/Oklch roundtrips
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn oklab_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let orig = Srgba { r, g, b, a: 255 };
        let lin: LinRgba = orig.into();
        let oklab: Oklab = lin.into();
        let back_lin: LinRgba = oklab.into();
        let back: Srgba = back_lin.into();
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 1,
            "r: {} vs {}", orig.r, back.r);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 1);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 1);
    }

    #[test]
    fn oklch_roundtrip(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let orig = Srgba { r, g, b, a: 255 };
        let lin: LinRgba = orig.into();
        let oklab: Oklab = lin.into();
        let oklch: Oklch = oklab.into();
        let back_oklab: Oklab = oklch.into();
        let back_lin: LinRgba = back_oklab.into();
        let back: Srgba = back_lin.into();
        prop_assert!((orig.r as i16 - back.r as i16).unsigned_abs() <= 2,
            "r: {} vs {}", orig.r, back.r);
        prop_assert!((orig.g as i16 - back.g as i16).unsigned_abs() <= 2);
        prop_assert!((orig.b as i16 - back.b as i16).unsigned_abs() <= 2);
    }
}

// ---------------------------------------------------------------------------
// Delta-E CIE94 properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn delta_e_94_non_negative(
        l1 in 0.0f64..=100.0, a1 in -128.0f64..=128.0, b1 in -128.0f64..=128.0,
        l2 in 0.0f64..=100.0, a2 in -128.0f64..=128.0, b2 in -128.0f64..=128.0,
    ) {
        let c1 = CieLab { l: l1, a: a1, b: b1 };
        let c2 = CieLab { l: l2, a: a2, b: b2 };
        prop_assert!(delta_e_cie94(&c1, &c2) >= 0.0);
    }

    #[test]
    fn delta_e_94_identity(l in 0.0f64..=100.0, a in -128.0f64..=128.0, b_val in -128.0f64..=128.0) {
        let c = CieLab { l, a, b: b_val };
        prop_assert!(delta_e_cie94(&c, &c) < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// BT.2020 YUV roundtrip
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn yuv420p_bt2020_roundtrip_gray(v in 16u8..=235) {
        let data: Vec<u8> = [v, v, v, 255].repeat(16);
        let buf = PixelBuffer::new(data, 4, 4, PixelFormat::Rgba8).unwrap();
        let yuv = convert::rgba_to_yuv420p_bt2020(&buf).unwrap();
        let back = convert::yuv420p_to_rgba_bt2020(&yuv).unwrap();
        prop_assert!((back.data()[0] as i16 - v as i16).unsigned_abs() < 3,
            "in={} out={}", v, back.data()[0]);
    }
}

// ---------------------------------------------------------------------------
// Additional composite properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn fade_one_is_identity(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let mut buf = PixelBuffer::new(vec![r, g, b, 255], 1, 1, PixelFormat::Rgba8).unwrap();
        ranga::composite::fade(&mut buf, 1.0).unwrap();
        prop_assert_eq!(buf.data()[0], r);
        prop_assert_eq!(buf.data()[1], g);
        prop_assert_eq!(buf.data()[2], b);
    }

    #[test]
    fn wipe_at_zero_is_all_a(r in 0u8..=255, g in 0u8..=255, b in 0u8..=255) {
        let a = PixelBuffer::new([r, g, b, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        let b_buf = PixelBuffer::new([0, 0, 0, 255].repeat(4), 2, 2, PixelFormat::Rgba8).unwrap();
        let result = ranga::composite::wipe(&a, &b_buf, 0.0).unwrap();
        prop_assert_eq!(result.data()[0], r);
    }
}
