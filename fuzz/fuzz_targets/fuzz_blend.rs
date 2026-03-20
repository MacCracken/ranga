#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::blend::{blend_pixel, blend_row_normal, BlendMode};

fuzz_target!(|data: &[u8]| {
    // Need at least 9 bytes: 4 src + 4 dst + 1 opacity
    if data.len() < 9 {
        return;
    }

    let src = [data[0], data[1], data[2], data[3]];
    let dst = [data[4], data[5], data[6], data[7]];
    let opacity = data[8];

    let modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
    ];

    for mode in modes {
        let _ = blend_pixel(src, dst, mode, opacity);
    }

    // Try blend_row_normal if we have enough data for at least one pixel pair
    if data.len() >= 17 {
        // Use a length that is a multiple of 4
        let usable = ((data.len() - 9) / 8) * 4;
        if usable >= 4 {
            let src_row = &data[9..9 + usable];
            let mut dst_row = data[9 + usable..9 + usable * 2].to_vec();
            if dst_row.len() == src_row.len() {
                blend_row_normal(src_row, &mut dst_row, opacity);
            }
        }
    }
});
