use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::filter;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn make_buf() -> PixelBuffer {
    PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap()
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

criterion_group!(
    benches,
    bench_brightness,
    bench_contrast,
    bench_saturation,
    bench_grayscale,
    bench_invert,
    bench_levels,
    bench_curves,
);
criterion_main!(benches);
