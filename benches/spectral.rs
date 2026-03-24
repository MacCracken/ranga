use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ranga::color::CieXyz;
use ranga::spectral::*;

fn bench_spd_to_xyz(c: &mut Criterion) {
    let d65_spd = illuminant_d65();
    c.bench_function("spd_to_xyz", |b| b.iter(|| spd_to_xyz(black_box(&d65_spd))));
}

fn bench_xyz_to_cct(c: &mut Criterion) {
    let d65 = d65_white();
    c.bench_function("xyz_to_cct", |b| b.iter(|| xyz_to_cct(black_box(&d65))));
}

fn bench_wavelength_to_xyz(c: &mut Criterion) {
    c.bench_function("wavelength_to_xyz", |b| {
        b.iter(|| wavelength_to_xyz(black_box(555.0)))
    });
}

fn bench_blackbody_spd(c: &mut Criterion) {
    c.bench_function("blackbody_spd", |b| {
        b.iter(|| blackbody_spd(black_box(5500.0)))
    });
}

fn bench_color_temperature_to_rgb(c: &mut Criterion) {
    c.bench_function("color_temperature_to_rgb", |b| {
        b.iter(|| color_temperature_to_rgb(black_box(6500.0)))
    });
}

fn bench_cie_cmf_at(c: &mut Criterion) {
    c.bench_function("cie_cmf_at", |b| b.iter(|| cie_cmf_at(black_box(555.0))));
}

fn bench_cri(c: &mut Criterion) {
    let d65_spd = illuminant_d65();
    c.bench_function("color_rendering_index", |b| {
        b.iter(|| color_rendering_index(black_box(&d65_spd)))
    });
}

fn bench_xyz_conversion_roundtrip(c: &mut Criterion) {
    let xyz = CieXyz {
        x: 0.95047,
        y: 1.0,
        z: 1.08883,
    };
    c.bench_function("xyz_conversion_roundtrip", |b| {
        b.iter(|| {
            let p: PrakashXyz = black_box(xyz).into();
            let _back: CieXyz = p.into();
        })
    });
}

criterion_group!(
    benches,
    bench_spd_to_xyz,
    bench_xyz_to_cct,
    bench_wavelength_to_xyz,
    bench_blackbody_spd,
    bench_color_temperature_to_rgb,
    bench_cie_cmf_at,
    bench_cri,
    bench_xyz_conversion_roundtrip,
);
criterion_main!(benches);
