use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::blend::{
    BlendMode, blend_pixel, blend_pixel_argb, blend_row_normal, blend_row_normal_argb,
};

fn bench_blend_pixel(c: &mut Criterion) {
    c.bench_function("blend_pixel_normal", |b| {
        b.iter(|| {
            blend_pixel(
                black_box([200, 100, 50, 200]),
                black_box([50, 100, 200, 255]),
                BlendMode::Normal,
                200,
            )
        })
    });
}

fn bench_blend_row(c: &mut Criterion) {
    let src = vec![128u8; 1920 * 4];
    let mut dst = vec![64u8; 1920 * 4];
    c.bench_function("blend_row_1920px", |b| {
        b.iter(|| blend_row_normal(black_box(&src), black_box(&mut dst), 200))
    });
}

fn bench_blend_row_1080p(c: &mut Criterion) {
    let src = vec![128u8; 1920 * 4];
    let mut dst = vec![64u8; 1920 * 4];
    c.bench_function("blend_row_1080p_width", |b| {
        b.iter(|| blend_row_normal(black_box(&src), black_box(&mut dst), 200))
    });
}

fn bench_blend_pixel_all_modes(c: &mut Criterion) {
    let modes = [
        ("Normal", BlendMode::Normal),
        ("Multiply", BlendMode::Multiply),
        ("Screen", BlendMode::Screen),
        ("Overlay", BlendMode::Overlay),
        ("Darken", BlendMode::Darken),
        ("Lighten", BlendMode::Lighten),
        ("ColorDodge", BlendMode::ColorDodge),
        ("ColorBurn", BlendMode::ColorBurn),
        ("SoftLight", BlendMode::SoftLight),
        ("HardLight", BlendMode::HardLight),
        ("Difference", BlendMode::Difference),
        ("Exclusion", BlendMode::Exclusion),
    ];
    let mut group = c.benchmark_group("blend_pixel_modes");
    for (name, mode) in &modes {
        group.bench_function(*name, |b| {
            b.iter(|| {
                blend_pixel(
                    black_box([200, 100, 50, 200]),
                    black_box([50, 100, 200, 255]),
                    *mode,
                    200,
                )
            })
        });
    }
    group.finish();
}

fn bench_blend_pixel_argb(c: &mut Criterion) {
    c.bench_function("blend_pixel_argb_normal", |b| {
        b.iter(|| {
            blend_pixel_argb(
                black_box([200, 100, 50, 200]),
                black_box([255, 50, 100, 200]),
                BlendMode::Normal,
                200,
            )
        })
    });
}

fn bench_blend_row_argb(c: &mut Criterion) {
    let src = vec![128u8; 1920 * 4];
    let mut dst = vec![64u8; 1920 * 4];
    c.bench_function("blend_row_argb_1920px", |b| {
        b.iter(|| blend_row_normal_argb(black_box(&src), black_box(&mut dst), 200))
    });
}

criterion_group!(
    benches,
    bench_blend_pixel,
    bench_blend_row,
    bench_blend_row_1080p,
    bench_blend_pixel_all_modes,
    bench_blend_pixel_argb,
    bench_blend_row_argb,
);
criterion_main!(benches);
