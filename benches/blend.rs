use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::blend::{blend_pixel, blend_row_normal, BlendMode};

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

criterion_group!(benches, bench_blend_pixel, bench_blend_row);
criterion_main!(benches);
