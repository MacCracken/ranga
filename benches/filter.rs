use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn make_buf() -> PixelBuffer {
    PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap()
}

fn make_varied_buf() -> PixelBuffer {
    let data: Vec<u8> = (0..1920 * 1080 * 4).map(|i| (i % 256) as u8).collect();
    PixelBuffer::new(data, 1920, 1080, PixelFormat::Rgba8).unwrap()
}

fn bench_brightness(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("brightness_1080p", |b| {
        b.iter(|| filter::brightness(black_box(&mut buf), 0.1).unwrap())
    });
}

fn bench_contrast(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("contrast_1080p", |b| {
        b.iter(|| filter::contrast(black_box(&mut buf), 1.2).unwrap())
    });
}

fn bench_saturation(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("saturation_1080p", |b| {
        b.iter(|| filter::saturation(black_box(&mut buf), 1.5).unwrap())
    });
}

fn bench_grayscale(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("grayscale_1080p", |b| {
        b.iter(|| filter::grayscale(black_box(&mut buf)).unwrap())
    });
}

fn bench_invert(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("invert_1080p", |b| {
        b.iter(|| filter::invert(black_box(&mut buf)).unwrap())
    });
}

fn bench_levels(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("levels_1080p", |b| {
        b.iter(|| filter::levels(black_box(&mut buf), 0.1, 0.9, 1.2).unwrap())
    });
}

fn bench_curves(c: &mut Criterion) {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = (255.0 * ((i as f32 / 255.0).powf(0.8))) as u8;
    }
    let mut buf = make_buf();
    c.bench_function("curves_1080p", |b| {
        b.iter(|| filter::curves(black_box(&mut buf), &lut).unwrap())
    });
}

fn bench_gaussian_blur(c: &mut Criterion) {
    let buf = make_varied_buf();
    c.bench_function("gaussian_blur_r3_1080p", |b| {
        b.iter(|| filter::gaussian_blur(black_box(&buf), 3).unwrap())
    });
}

fn bench_box_blur(c: &mut Criterion) {
    let buf = make_varied_buf();
    c.bench_function("box_blur_r3_1080p", |b| {
        b.iter(|| filter::box_blur(black_box(&buf), 3).unwrap())
    });
}

fn bench_unsharp_mask(c: &mut Criterion) {
    let buf = make_varied_buf();
    c.bench_function("unsharp_mask_r2_1080p", |b| {
        b.iter(|| filter::unsharp_mask(black_box(&buf), 2, 1.0).unwrap())
    });
}

fn bench_hue_shift(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("hue_shift_1080p", |b| {
        b.iter(|| filter::hue_shift(black_box(&mut buf), 30.0).unwrap())
    });
}

fn bench_color_balance(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("color_balance_1080p", |b| {
        b.iter(|| {
            filter::color_balance(
                black_box(&mut buf),
                [0.0, 0.0, 0.05],
                [0.05, 0.0, -0.05],
                [0.0, 0.0, 0.0],
            )
            .unwrap()
        })
    });
}

fn bench_vignette(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("vignette_1080p", |b| {
        b.iter(|| filter::vignette(black_box(&mut buf), 0.5).unwrap())
    });
}

fn bench_lut3d(c: &mut Criterion) {
    // Build a 17x17x17 identity LUT (standard size)
    let size = 17;
    let mut cube = format!("LUT_3D_SIZE {size}\n");
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let rf = r as f32 / (size - 1) as f32;
                let gf = g as f32 / (size - 1) as f32;
                let bf = b as f32 / (size - 1) as f32;
                cube.push_str(&format!("{rf} {gf} {bf}\n"));
            }
        }
    }
    let lut = filter::Lut3d::from_cube(&cube).unwrap();
    let mut buf = make_varied_buf();
    c.bench_function("lut3d_17cube_1080p", |b| {
        b.iter(|| filter::apply_lut3d(black_box(&mut buf), &lut).unwrap())
    });
}

fn bench_noise_gaussian(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("noise_gaussian_1080p", |b| {
        b.iter(|| filter::noise_gaussian(black_box(&mut buf), 0.1, 42).unwrap())
    });
}

fn bench_noise_salt_pepper(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("noise_salt_pepper_1080p", |b| {
        b.iter(|| filter::noise_salt_pepper(black_box(&mut buf), 0.05, 42).unwrap())
    });
}

fn bench_median(c: &mut Criterion) {
    // Smaller size for median — it's O(n * radius^2) per pixel
    let buf = PixelBuffer::new(vec![128; 512 * 512 * 4], 512, 512, PixelFormat::Rgba8).unwrap();
    c.bench_function("median_r1_512x512", |b| {
        b.iter(|| filter::median(black_box(&buf), 1).unwrap())
    });
}

fn bench_bilateral(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 256 * 256 * 4], 256, 256, PixelFormat::Rgba8).unwrap();
    c.bench_function("bilateral_r2_256x256", |b| {
        b.iter(|| filter::bilateral(black_box(&buf), 2, 10.0, 30.0).unwrap())
    });
}

fn bench_vibrance(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("vibrance_1080p", |b| {
        b.iter(|| filter::vibrance(black_box(&mut buf), 0.5).unwrap())
    });
}

fn bench_channel_mixer(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("channel_mixer_1080p", |b| {
        b.iter(|| {
            filter::channel_mixer(
                black_box(&mut buf),
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            )
            .unwrap()
        })
    });
}

fn bench_threshold(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("threshold_1080p", |b| {
        b.iter(|| filter::threshold(black_box(&mut buf), 128).unwrap())
    });
}

fn bench_flood_fill(c: &mut Criterion) {
    // Uniform buffer — flood fill covers entire image
    let mut buf = make_buf();
    c.bench_function("flood_fill_1080p_uniform", |b| {
        b.iter(|| {
            // Reset to uniform before each fill
            buf.data.fill(128);
            filter::flood_fill(black_box(&mut buf), 0, 0, [255, 0, 0, 255], 10).unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_brightness,
    bench_contrast,
    bench_saturation,
    bench_grayscale,
    bench_invert,
    bench_levels,
    bench_curves,
    bench_gaussian_blur,
    bench_box_blur,
    bench_unsharp_mask,
    bench_hue_shift,
    bench_color_balance,
    bench_vignette,
    bench_lut3d,
    bench_noise_gaussian,
    bench_noise_salt_pepper,
    bench_median,
    bench_bilateral,
    bench_vibrance,
    bench_channel_mixer,
    bench_threshold,
    bench_flood_fill,
);
criterion_main!(benches);
