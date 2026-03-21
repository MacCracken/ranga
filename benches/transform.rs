use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::{self, Affine, ScaleFilter};

fn make_buf() -> PixelBuffer {
    PixelBuffer::new(vec![128; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap()
}

fn bench_crop(c: &mut Criterion) {
    let buf = make_buf();
    c.bench_function("crop_1080p_to_720p", |b| {
        b.iter(|| transform::crop(black_box(&buf), 240, 180, 1520, 900).unwrap())
    });
}

fn bench_resize_nearest(c: &mut Criterion) {
    let buf = make_buf();
    c.bench_function("resize_nearest_1080p_to_720p", |b| {
        b.iter(|| transform::resize(black_box(&buf), 1280, 720, ScaleFilter::Nearest).unwrap())
    });
}

fn bench_resize_bilinear(c: &mut Criterion) {
    let buf = make_buf();
    c.bench_function("resize_bilinear_1080p_to_720p", |b| {
        b.iter(|| transform::resize(black_box(&buf), 1280, 720, ScaleFilter::Bilinear).unwrap())
    });
}

fn bench_flip_horizontal(c: &mut Criterion) {
    let buf = make_buf();
    c.bench_function("flip_horizontal_1080p", |b| {
        b.iter(|| transform::flip_horizontal(black_box(&buf)).unwrap())
    });
}

fn bench_flip_vertical(c: &mut Criterion) {
    let buf = make_buf();
    c.bench_function("flip_vertical_1080p", |b| {
        b.iter(|| transform::flip_vertical(black_box(&buf)).unwrap())
    });
}

fn bench_affine_rotate(c: &mut Criterion) {
    let buf = PixelBuffer::new(vec![128; 512 * 512 * 4], 512, 512, PixelFormat::Rgba8).unwrap();
    let rot = Affine::rotate(0.1);
    c.bench_function("affine_rotate_512x512", |b| {
        b.iter(|| {
            transform::affine_transform(black_box(&buf), &rot, 512, 512, ScaleFilter::Bilinear)
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_crop,
    bench_resize_nearest,
    bench_resize_bilinear,
    bench_flip_horizontal,
    bench_flip_vertical,
    bench_affine_rotate,
);
criterion_main!(benches);
