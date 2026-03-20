use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::color::*;

fn bench_srgb_to_lab(c: &mut Criterion) {
    let color = Srgba {
        r: 128,
        g: 64,
        b: 200,
        a: 255,
    };
    c.bench_function("srgb_to_lab", |b| {
        b.iter(|| -> CieLab { black_box(color).into() })
    });
}

fn bench_delta_e_cie76(c: &mut Criterion) {
    let a = CieLab {
        l: 50.0,
        a: 25.0,
        b: -10.0,
    };
    let b_lab = CieLab {
        l: 55.0,
        a: 20.0,
        b: -15.0,
    };
    c.bench_function("delta_e_cie76", |b| {
        b.iter(|| delta_e_cie76(black_box(&a), black_box(&b_lab)))
    });
}

fn bench_delta_e_ciede2000(c: &mut Criterion) {
    let a = CieLab {
        l: 50.0,
        a: 2.6772,
        b: -79.7751,
    };
    let b_lab = CieLab {
        l: 50.0,
        a: 0.0,
        b: -82.7485,
    };
    c.bench_function("delta_e_ciede2000", |b| {
        b.iter(|| delta_e_ciede2000(black_box(&a), black_box(&b_lab)))
    });
}

fn bench_p3_to_srgb(c: &mut Criterion) {
    c.bench_function("p3_to_linear_srgb", |b| {
        b.iter(|| p3_to_linear_srgb(black_box(0.8), black_box(0.3), black_box(0.5)))
    });
}

fn bench_cmyk_roundtrip(c: &mut Criterion) {
    let color = Srgba {
        r: 200,
        g: 100,
        b: 50,
        a: 255,
    };
    c.bench_function("cmyk_roundtrip", |b| {
        b.iter(|| {
            let cmyk = srgb_to_cmyk(black_box(&color));
            cmyk_to_srgb(&cmyk)
        })
    });
}

fn bench_color_temperature(c: &mut Criterion) {
    c.bench_function("color_temperature_6600K", |b| {
        b.iter(|| color_temperature(black_box(6600.0)))
    });
}

fn bench_hsl_roundtrip(c: &mut Criterion) {
    let color = Srgba {
        r: 128,
        g: 64,
        b: 200,
        a: 255,
    };
    c.bench_function("hsl_roundtrip", |b| {
        b.iter(|| {
            let hsl: Hsl = black_box(color).into();
            let _back: Srgba = hsl.into();
        })
    });
}

criterion_group!(
    benches,
    bench_srgb_to_lab,
    bench_delta_e_cie76,
    bench_delta_e_ciede2000,
    bench_p3_to_srgb,
    bench_cmyk_roundtrip,
    bench_color_temperature,
    bench_hsl_roundtrip,
);
criterion_main!(benches);
