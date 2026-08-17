//! GPU compute example — demonstrates context creation, pipeline caching, and operations.
//!
//! Run with: `cargo run --example gpu_compute --features gpu`

use ranga::gpu::{GpuChain, GpuContext, gpu_brightness_contrast, gpu_invert};
use ranga::pixel::{PixelBuffer, PixelFormat};

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            eprintln!("GPU example failed: {e}");
            eprintln!("(This is expected if no GPU adapter is available.)");
            std::process::exit(1);
        }
    }
}

fn run() -> ranga::Result<()> {
    // ── Create GPU context ───────────────────────────────────────────────
    // GpuContext::new() requests a high-performance adapter. It can fail if
    // no suitable GPU is found, so we propagate the error.
    let mut ctx = GpuContext::new().map_err(ranga::RangaError::from)?;
    println!("GPU adapter: {}", ctx.adapter_name());
    println!("Backend:     {}", ctx.backend_name());

    // ── Create a 4x4 buffer with a gradient ─────────────────────────────
    let w = 4u32;
    let h = 4u32;
    let pixel_count = (w * h) as usize;
    let mut data = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let t = i as f32 / (pixel_count - 1) as f32;
        let v = (t * 255.0) as u8;
        data.extend_from_slice(&[v, v / 2, 255 - v, 255]);
    }
    let mut buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

    println!("\nOriginal (first 4 pixels):");
    print_pixels(&buf, 4);

    // ── GPU invert ──────────────────────────────────────────────────────
    // Inverts R, G, B channels (1.0 - value) while preserving alpha.
    gpu_invert(&mut ctx, &mut buf)?;
    println!("\nAfter gpu_invert:");
    print_pixels(&buf, 4);

    // ── GPU brightness/contrast ─────────────────────────────────────────
    // Brightness offset +0.1, contrast multiplier 1.3x.
    gpu_brightness_contrast(&mut ctx, &mut buf, 0.1, 1.3)?;
    println!("\nAfter gpu_brightness_contrast(+0.1, 1.3x):");
    print_pixels(&buf, 4);

    // ── GpuChain: batched operations ────────────────────────────────────
    // GpuChain uploads once, runs multiple operations on-GPU, then downloads
    // the result. This avoids repeated CPU<->GPU transfers.
    let chain_data: Vec<u8> = (0..pixel_count)
        .flat_map(|i| {
            let t = i as f32 / (pixel_count - 1) as f32;
            let v = (t * 255.0) as u8;
            [v, v / 2, 255 - v, 255]
        })
        .collect();
    let chain_input = PixelBuffer::new(chain_data, w, h, PixelFormat::Rgba8).unwrap();

    println!("\nGpuChain input (first 4 pixels):");
    print_pixels(&chain_input, 4);

    let chain = GpuChain::new(&mut ctx, &chain_input)?;
    let result = chain
        .invert()? // invert colors on GPU
        .grayscale()? // convert to grayscale on GPU
        .finish()?; // download result back to CPU

    println!("\nGpuChain result (invert -> grayscale):");
    print_pixels(&result, 4);

    // Verify grayscale: R == G == B for every pixel
    let all_gray = result
        .data()
        .chunks_exact(4)
        .all(|p| p[0] == p[1] && p[1] == p[2]);
    println!("All pixels are grayscale: {all_gray}");

    Ok(())
}

/// Print the first N pixels of a buffer.
fn print_pixels(buf: &PixelBuffer, n: usize) {
    for (i, pixel) in buf.data().chunks_exact(4).enumerate().take(n) {
        println!(
            "  pixel[{i}]: R={:>3} G={:>3} B={:>3} A={:>3}",
            pixel[0], pixel[1], pixel[2], pixel[3]
        );
    }
}
