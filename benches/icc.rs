use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::icc::{IccProfile, ToneCurve, srgb_v2_profile};

fn bench_srgb_v2_generation(c: &mut Criterion) {
    c.bench_function("srgb_v2_profile_generation", |b| b.iter(srgb_v2_profile));
}

fn bench_icc_parse(c: &mut Criterion) {
    let bytes = srgb_v2_profile();
    c.bench_function("icc_from_bytes_srgb_v2", |b| {
        b.iter(|| IccProfile::from_bytes(black_box(&bytes)).unwrap())
    });
}

fn bench_icc_apply(c: &mut Criterion) {
    let bytes = srgb_v2_profile();
    let profile = IccProfile::from_bytes(&bytes).unwrap();
    c.bench_function("icc_apply_mid_gray", |b| {
        b.iter(|| profile.apply(black_box(0.5), black_box(0.5), black_box(0.5)))
    });
    c.bench_function("icc_apply_white", |b| {
        b.iter(|| profile.apply(black_box(1.0), black_box(1.0), black_box(1.0)))
    });
    c.bench_function("icc_apply_black", |b| {
        b.iter(|| profile.apply(black_box(0.0), black_box(0.0), black_box(0.0)))
    });
    c.bench_function("icc_apply_saturated_red", |b| {
        b.iter(|| profile.apply(black_box(1.0), black_box(0.0), black_box(0.0)))
    });
}

fn bench_tone_curve_gamma(c: &mut Criterion) {
    let tc = ToneCurve::Gamma(2.2);
    c.bench_function("tone_curve_gamma_2_2", |b| {
        b.iter(|| tc.apply(black_box(0.5)))
    });
}

fn bench_tone_curve_table(c: &mut Criterion) {
    let table: Vec<f64> = (0..256).map(|i| (i as f64 / 255.0).powf(2.2)).collect();
    let tc = ToneCurve::Table(table);
    c.bench_function("tone_curve_table_256", |b| {
        b.iter(|| tc.apply(black_box(0.5)))
    });
}

criterion_group!(
    benches,
    bench_srgb_v2_generation,
    bench_icc_parse,
    bench_icc_apply,
    bench_tone_curve_gamma,
    bench_tone_curve_table,
);
criterion_main!(benches);
