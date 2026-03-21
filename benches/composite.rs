use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::composite;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn make_buf() -> PixelBuffer {
    PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap()
}

fn bench_premultiply(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("premultiply_alpha_1080p", |b| {
        b.iter(|| composite::premultiply_alpha(black_box(&mut buf)).unwrap())
    });
}

fn bench_composite_at(c: &mut Criterion) {
    let mut dst = make_buf();
    let src = PixelBuffer::new(vec![200; 640 * 480 * 4], 640, 480, PixelFormat::Rgba8).unwrap();
    c.bench_function("composite_at_640x480_on_1080p", |b| {
        b.iter(|| composite::composite_at(black_box(&src), &mut dst, 100, 100, 0.8).unwrap())
    });
}

fn bench_dissolve(c: &mut Criterion) {
    let a = make_buf();
    let b = PixelBuffer::new(vec![64; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    c.bench_function("dissolve_1080p", |b_iter| {
        b_iter.iter(|| composite::dissolve(black_box(&a), black_box(&b), 0.5).unwrap())
    });
}

fn bench_fade(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("fade_1080p", |b| {
        b.iter(|| composite::fade(black_box(&mut buf), 0.5).unwrap())
    });
}

fn bench_wipe(c: &mut Criterion) {
    let a = make_buf();
    let b = PixelBuffer::new(vec![64; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    c.bench_function("wipe_1080p", |b_iter| {
        b_iter.iter(|| composite::wipe(black_box(&a), black_box(&b), 0.5).unwrap())
    });
}

fn bench_gradient(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("gradient_linear_1080p", |b| {
        b.iter(|| {
            composite::gradient_linear(black_box(&mut buf), [255, 0, 0, 255], [0, 0, 255, 255])
                .unwrap()
        })
    });
}

fn bench_checkerboard(c: &mut Criterion) {
    let mut buf = make_buf();
    c.bench_function("checkerboard_1080p", |b| {
        b.iter(|| {
            composite::fill_checkerboard(
                black_box(&mut buf),
                16,
                [200, 200, 200, 255],
                [255, 255, 255, 255],
            )
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_premultiply,
    bench_composite_at,
    bench_dissolve,
    bench_fade,
    bench_wipe,
    bench_gradient,
    bench_checkerboard,
);
criterion_main!(benches);
