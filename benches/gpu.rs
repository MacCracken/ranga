use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::blend::BlendMode;
use ranga::gpu::{self, GpuContext};
use ranga::pixel::{PixelBuffer, PixelFormat};
use ranga::transform::ScaleFilter;
use ranga::{blend, composite, filter, transform};

fn get_ctx() -> Option<GpuContext> {
    GpuContext::new().ok()
}

fn bench_gpu_vs_cpu_blend(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
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
        b.iter(|| {
            gpu::gpu_blend(&mut ctx, black_box(&src), &mut dst, BlendMode::Normal, 0.8).unwrap()
        })
    });

    c.bench_function("cpu_blend_normal_1080p", |b| {
        let src_data = vec![128u8; 1920 * 1080 * 4];
        let mut dst = dst_data.clone();
        b.iter(|| blend::blend_row_normal(black_box(&src_data), &mut dst, 204))
    });
}

fn bench_gpu_vs_cpu_invert(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_invert_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_invert(&mut ctx, black_box(&mut buf)).unwrap())
    });

    c.bench_function("cpu_invert_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::invert(black_box(&mut buf)).unwrap())
    });
}

fn bench_gpu_vs_cpu_grayscale(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_grayscale_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_grayscale(&mut ctx, black_box(&mut buf)).unwrap())
    });

    c.bench_function("cpu_grayscale_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::grayscale(black_box(&mut buf)).unwrap())
    });
}

fn bench_gpu_vs_cpu_brightness(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_brightness_contrast_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_brightness_contrast(&mut ctx, black_box(&mut buf), 0.1, 1.2).unwrap())
    });

    c.bench_function("cpu_brightness_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::brightness(black_box(&mut buf), 0.1).unwrap())
    });
}

fn bench_gpu_vs_cpu_saturation(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_saturation_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_saturation(&mut ctx, black_box(&mut buf), 1.5).unwrap())
    });

    c.bench_function("cpu_saturation_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::saturation(black_box(&mut buf), 1.5).unwrap())
    });
}

fn bench_gpu_vs_cpu_blur(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data: Vec<u8> = (0..1920 * 1080 * 4).map(|i| (i % 256) as u8).collect();

    c.bench_function("gpu_gaussian_blur_r3_1080p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_gaussian_blur(&mut ctx, black_box(&buf), 3).unwrap())
    });

    c.bench_function("cpu_gaussian_blur_r3_1080p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| ranga::filter::gaussian_blur(black_box(&buf), 3).unwrap())
    });
}

fn bench_gpu_chain_vs_sequential(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_chain_invert_brightness_saturation_1080p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| {
            gpu::GpuChain::new(&mut ctx, black_box(&buf))
                .unwrap()
                .invert()
                .unwrap()
                .brightness_contrast(0.1, 1.2)
                .unwrap()
                .saturation(1.5)
                .unwrap()
                .finish()
                .unwrap()
        })
    });

    c.bench_function("gpu_sequential_invert_brightness_saturation_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| {
            gpu::gpu_invert(&mut ctx, black_box(&mut buf)).unwrap();
            gpu::gpu_brightness_contrast(&mut ctx, black_box(&mut buf), 0.1, 1.2).unwrap();
            gpu::gpu_saturation(&mut ctx, black_box(&mut buf), 1.5).unwrap();
        })
    });
}

fn bench_gpu_vs_cpu_noise(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_noise_gaussian_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_noise_gaussian(&mut ctx, black_box(&mut buf), 0.1, 42).unwrap())
    });

    c.bench_function("cpu_noise_gaussian_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| filter::noise_gaussian(black_box(&mut buf), 0.1, 42).unwrap())
    });
}

fn bench_gpu_vs_cpu_dissolve(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let src_data = vec![128u8; 1920 * 1080 * 4];
    let dst_data = vec![64u8; 1920 * 1080 * 4];

    c.bench_function("gpu_dissolve_1080p", |b| {
        let src = PixelBuffer::new(src_data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        let mut dst = PixelBuffer::new(dst_data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_dissolve(&mut ctx, black_box(&src), &mut dst, 0.5).unwrap())
    });

    c.bench_function("cpu_dissolve_1080p", |b| {
        let src = PixelBuffer::new(src_data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        let dst = PixelBuffer::new(dst_data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| composite::dissolve(black_box(&src), black_box(&dst), 0.5).unwrap())
    });
}

fn bench_gpu_vs_cpu_fade(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_fade_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_fade(&mut ctx, black_box(&mut buf), 0.5).unwrap())
    });

    c.bench_function("cpu_fade_1080p", |b| {
        let mut buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| composite::fade(black_box(&mut buf), 0.5).unwrap())
    });
}

fn bench_gpu_vs_cpu_crop(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_crop_1080p_to_720p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_crop(&mut ctx, black_box(&buf), 240, 180, 1520, 900).unwrap())
    });

    c.bench_function("cpu_crop_1080p_to_720p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| transform::crop(black_box(&buf), 240, 180, 1520, 900).unwrap())
    });
}

fn bench_gpu_vs_cpu_resize(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_resize_bilinear_1080p_to_720p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| {
            gpu::gpu_resize(&mut ctx, black_box(&buf), 1280, 720, ScaleFilter::Bilinear).unwrap()
        })
    });

    c.bench_function("cpu_resize_bilinear_1080p_to_720p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| transform::resize(black_box(&buf), 1280, 720, ScaleFilter::Bilinear).unwrap())
    });
}

fn bench_gpu_vs_cpu_flip(c: &mut Criterion) {
    let mut ctx = match get_ctx() {
        Some(ctx) => ctx,
        None => return,
    };

    let data = vec![128u8; 1920 * 1080 * 4];

    c.bench_function("gpu_flip_horizontal_1080p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| gpu::gpu_flip_horizontal(&mut ctx, black_box(&buf)).unwrap())
    });

    c.bench_function("cpu_flip_horizontal_1080p", |b| {
        let buf = PixelBuffer::new(data.clone(), 1920, 1080, PixelFormat::Rgba8).unwrap();
        b.iter(|| transform::flip_horizontal(black_box(&buf)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_gpu_vs_cpu_blend,
    bench_gpu_vs_cpu_invert,
    bench_gpu_vs_cpu_grayscale,
    bench_gpu_vs_cpu_brightness,
    bench_gpu_vs_cpu_saturation,
    bench_gpu_vs_cpu_blur,
    bench_gpu_chain_vs_sequential,
    bench_gpu_vs_cpu_noise,
    bench_gpu_vs_cpu_dissolve,
    bench_gpu_vs_cpu_fade,
    bench_gpu_vs_cpu_crop,
    bench_gpu_vs_cpu_resize,
    bench_gpu_vs_cpu_flip,
);
criterion_main!(benches);
