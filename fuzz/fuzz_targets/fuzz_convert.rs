#![no_main]

use libfuzzer_sys::fuzz_target;
use ranga::convert::{argb_to_nv12, rgba_to_yuv420p, yuv420p_to_rgba};
use ranga::pixel::{PixelBuffer, PixelFormat};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first two bytes for dimensions (even, non-zero, capped to keep buffers small)
    let w = ((data[0] as u32 & 0x3E).max(2)) & !1; // even, 2..62
    let h = ((data[1] as u32 & 0x3E).max(2)) & !1; // even, 2..62
    let remaining = &data[2..];

    // Try RGBA8 -> YUV420p -> RGBA8
    let rgba_size = (w * h * 4) as usize;
    if remaining.len() >= rgba_size {
        let rgba_data = remaining[..rgba_size].to_vec();
        if let Ok(buf) = PixelBuffer::new(rgba_data, w, h, PixelFormat::Rgba8)
            && let Ok(yuv) = rgba_to_yuv420p(&buf)
        {
            let _ = yuv420p_to_rgba(&yuv);
        }
    }

    // Try constructing a YUV420p buffer directly
    let yuv_size = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    if remaining.len() >= yuv_size {
        let yuv_data = remaining[..yuv_size].to_vec();
        if let Ok(buf) = PixelBuffer::new(yuv_data, w, h, PixelFormat::Yuv420p) {
            let _ = yuv420p_to_rgba(&buf);
        }
    }

    // Try ARGB8 -> NV12
    let argb_size = (w * h * 4) as usize;
    if remaining.len() >= argb_size {
        let argb_data = remaining[..argb_size].to_vec();
        if let Ok(buf) = PixelBuffer::new(argb_data, w, h, PixelFormat::Argb8) {
            let _ = argb_to_nv12(&buf);
        }
    }
});
