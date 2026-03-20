use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::convert;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn bench_rgba_to_yuv_bt601(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    c.bench_function("rgba_to_yuv420p_bt601_1080p", |b| {
        b.iter(|| convert::rgba_to_yuv420p(black_box(&buf)).unwrap())
    });
}

fn bench_yuv_to_rgba_bt601(c: &mut Criterion) {
    let rgba =
        PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();
    c.bench_function("yuv420p_to_rgba_bt601_1080p", |b| {
        b.iter(|| convert::yuv420p_to_rgba(black_box(&yuv)).unwrap())
    });
}

fn bench_rgba_to_yuv_bt709(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    c.bench_function("rgba_to_yuv420p_bt709_1080p", |b| {
        b.iter(|| convert::rgba_to_yuv420p_bt709(black_box(&buf)).unwrap())
    });
}

fn bench_yuv_to_rgba_bt709(c: &mut Criterion) {
    let rgba =
        PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    let yuv = convert::rgba_to_yuv420p_bt709(&rgba).unwrap();
    c.bench_function("yuv420p_to_rgba_bt709_1080p", |b| {
        b.iter(|| convert::yuv420p_to_rgba_bt709(black_box(&yuv)).unwrap())
    });
}

fn bench_nv12_to_rgba(c: &mut Criterion) {
    let argb_data: Vec<u8> = [255u8, 128, 128, 128].repeat(1920 * 1080);
    let argb = PixelBuffer::new(argb_data, 1920, 1080, PixelFormat::Argb8).unwrap();
    let nv12 = convert::argb_to_nv12(&argb).unwrap();
    c.bench_function("nv12_to_rgba_1080p", |b| {
        b.iter(|| convert::nv12_to_rgba(black_box(&nv12)).unwrap())
    });
}

fn bench_rgb8_to_rgba8(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 1920 * 1080 * 3], 1920, 1080, PixelFormat::Rgb8).unwrap();
    c.bench_function("rgb8_to_rgba8_1080p", |b| {
        b.iter(|| convert::rgb8_to_rgba8(black_box(&buf)).unwrap())
    });
}

fn bench_argb8_to_rgba8(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Argb8).unwrap();
    c.bench_function("argb8_to_rgba8_1080p", |b| {
        b.iter(|| convert::argb8_to_rgba8(black_box(&buf)).unwrap())
    });
}

fn bench_rgbaf32_to_rgba8(c: &mut Criterion) {
    // Build a valid RgbaF32 buffer
    let rgba =
        PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    let f32buf = convert::rgba8_to_rgbaf32(&rgba).unwrap();
    c.bench_function("rgbaf32_to_rgba8_1080p", |b| {
        b.iter(|| convert::rgbaf32_to_rgba8(black_box(&f32buf)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_rgba_to_yuv_bt601,
    bench_yuv_to_rgba_bt601,
    bench_rgba_to_yuv_bt709,
    bench_yuv_to_rgba_bt709,
    bench_nv12_to_rgba,
    bench_rgb8_to_rgba8,
    bench_argb8_to_rgba8,
    bench_rgbaf32_to_rgba8,
);
criterion_main!(benches);
