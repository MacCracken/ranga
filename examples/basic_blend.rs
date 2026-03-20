use ranga::blend::{BlendMode, blend_pixel};
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;

    // Create a solid red buffer (RGBA8)
    let red_data: Vec<u8> = (0..pixel_count).flat_map(|_| [255, 0, 0, 255]).collect();
    let red_buf = PixelBuffer::new(red_data, w, h, PixelFormat::Rgba8).unwrap();

    // Create a solid blue buffer (RGBA8)
    let blue_data: Vec<u8> = (0..pixel_count).flat_map(|_| [0, 0, 255, 255]).collect();
    let blue_buf = PixelBuffer::new(blue_data, w, h, PixelFormat::Rgba8).unwrap();

    // Blend red over blue, pixel by pixel, using Normal mode at 75% opacity
    let opacity: u8 = 192; // ~75%
    let mut result_data = Vec::with_capacity(pixel_count * 4);

    for i in 0..pixel_count {
        let si = i * 4;
        let src = [
            red_buf.data[si],
            red_buf.data[si + 1],
            red_buf.data[si + 2],
            red_buf.data[si + 3],
        ];
        let dst = [
            blue_buf.data[si],
            blue_buf.data[si + 1],
            blue_buf.data[si + 2],
            blue_buf.data[si + 3],
        ];
        let blended = blend_pixel(src, dst, BlendMode::Normal, opacity);
        result_data.extend_from_slice(&blended);
    }

    // Print the first few blended pixels
    println!("Blended red over blue at ~75% opacity (Normal mode):");
    for (i, pixel) in result_data.chunks_exact(4).enumerate().take(4) {
        println!(
            "  pixel[{i}]: R={} G={} B={} A={}",
            pixel[0], pixel[1], pixel[2], pixel[3]
        );
    }
}
