use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::blend::BlendMode;
use ranga::gpu::{self, GpuContext};
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::{blend, filter};

fn get_ctx() -> Option<GpuContext> {
    GpuContext::new().ok()
}

fn bench_gpu_vs_cpu_blend(c: &mut Criterion) {
    let ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => {
            eprintln!("No GPU available — skipping GPU benchmarks");
            return;
        }
    };

    let src =
        PixelBuffer::new(vec![128u8; 1920 * 1080 * 4], 1920, 1080, PixelFormat::Rgba8).unwrap();
    let dst_data = vec![64u8; 1920 * 1080 * 4];

    c.bench_function("gpu_blend_normal_1080p", |b| {
        let mut dst = PixelBuffer::new(dst_data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_blend(&ctx, black_box(&src), &mut dst, BlendMode::Normal, 0.8).unwrap())
    });

    c.bench_function("cpu_blend_normal_1080p", |b| {
        let src_data = vec![128u8; 1920 * 1080 * 4];
        let mut dst = dst_data.clone();
        b.iter(|| blend::blend_row_normal(black_box(&src_data), &mut dst, 204))
    });
}

fn bench_gpu_vs_cpu_invert(c: &mut Criterion) {
    let ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_invert_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_invert(&ctx, black_box(&mut buf)).unwrap())
    });

    c.bench_function("cpu_invert_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::invert(black_box(&mut buf)).unwrap())
    });
}

fn bench_gpu_vs_cpu_grayscale(c: &mut Criterion) {
    let ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_grayscale_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_grayscale(&ctx, black_box(&mut buf)).unwrap())
    });

    c.bench_function("cpu_grayscale_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::grayscale(black_box(&mut buf)).unwrap())
    });
}

fn bench_gpu_vs_cpu_brightness(c: &mut Criterion) {
    let ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_brightness_contrast_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_brightness_contrast(&ctx, black_box(&mut buf), 0.1, 1.2).unwrap())
    });

    c.bench_function("cpu_brightness_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::brightness(black_box(&mut buf), 0.1).unwrap())
    });
}

criterion_group!(
    benches,
    bench_gpu_vs_cpu_blend,
    bench_gpu_vs_cpu_invert,
    bench_gpu_vs_cpu_grayscale,
    bench_gpu_vs_cpu_brightness,
);
criterion_main!(benches);
