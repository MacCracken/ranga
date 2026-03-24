use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::histogram;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn make_varied_buf() -> PixelBuffer {
    let data: Vec<u8> = (0..1920 * 1080 * 4).map(|i| (i % 256) as u8).collect();
    PixelBuffer::new(data, 1920, 1080, PixelFormat::Rgba8).unwrap()
}

fn bench_luminance_histogram(c: &mut Criterion) {
    let buf = make_varied_buf();
    c.bench_function("luminance_histogram_1080p", |b| {
        b.iter(|| histogram::luminance_histogram(black_box(&buf), 256).unwrap())
    });
}

fn bench_rgb_histograms(c: &mut Criterion) {
    let buf = make_varied_buf();
    c.bench_function("rgb_histograms_1080p", |b| {
        b.iter(|| histogram::rgb_histograms(black_box(&buf)).unwrap())
    });
}

fn bench_equalize(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("equalize_1080p", |b| {
        b.iter(|| histogram::equalize(black_box(&mut buf)).unwrap())
    });
}

fn bench_auto_levels(c: &mut Criterion) {
    let mut buf = make_varied_buf();
    c.bench_function("auto_levels_1080p", |b| {
        b.iter(|| histogram::auto_levels(black_box(&mut buf)).unwrap())
    });
}

fn bench_chi_squared(c: &mut Criterion) {
    let a: Vec<f64> = (0..256).map(|i| (i as f64 / 255.0).powi(2)).collect();
    let b: Vec<f64> = (0..256).map(|i| (1.0 - i as f64 / 255.0).powi(2)).collect();
    c.bench_function("chi_squared_256bins", |bench| {
        bench.iter(|| histogram::chi_squared(black_box(&a), black_box(&b)))
    });
}

criterion_group!(
    benches,
    bench_luminance_histogram,
    bench_rgb_histograms,
    bench_equalize,
    bench_auto_levels,
    bench_chi_squared,
);
criterion_main!(benches);
