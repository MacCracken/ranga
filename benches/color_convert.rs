use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::convert;
use ranga::pixel::{PixelBuffer, PixelFormat};

fn bench_rgba_to_yuv(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();

    c.bench_function("rgba_to_yuv420p_1080p", |b| {
        b.iter(|| convert::rgba_to_yuv420p(black_box(&buf)).unwrap())
    });
}

fn bench_yuv_to_rgba(c: &mut Criterion) {
    let rgba =
        PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    let yuv = convert::rgba_to_yuv420p(&rgba).unwrap();

    c.bench_function("yuv420p_to_rgba_1080p", |b| {
        b.iter(|| convert::yuv420p_to_rgba(black_box(&yuv)).unwrap())
    });
}

criterion_group!(benches, bench_rgba_to_yuv, bench_yuv_to_rgba);
criterion_main!(benches);
