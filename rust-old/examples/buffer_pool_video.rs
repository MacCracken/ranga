//! Buffer pool — simulating a video frame processing pipeline with buffer reuse.

use ranga::filter;
use ranga::pixel::{BufferPool, PixelBuffer, PixelFormat};

fn main() {
    let w = 64u32;
    let h = 64u32;
    let frame_bytes = PixelFormat::Rgba8.buffer_size(w, h);
    let total_frames = 10u32;

    let mut pool = BufferPool::new(4);
    println!("Buffer pool created (max 4 buffers)");
    println!("Frame size: {w}x{h} RGBA8 = {frame_bytes} bytes");
    println!("Processing {total_frames} frames...\n");

    for frame in 0..total_frames {
        // Acquire a buffer from the pool.
        let raw = pool.acquire(frame_bytes);
        let reused = raw.capacity() >= frame_bytes && frame > 0;

        // Build a PixelBuffer from the pooled allocation.
        // Fill with "frame data": use the frame number as a base pixel value.
        let base = ((frame as f32 / total_frames as f32) * 200.0 + 28.0) as u8;
        let data: Vec<u8> = (0..(w * h)).flat_map(|_| [base, base, base, 255]).collect();
        let mut buf = PixelBuffer::new(data, w, h, PixelFormat::Rgba8).unwrap();

        // Apply a brightness filter (simulating per-frame processing).
        let offset = (frame as f32 - 5.0) * 0.05; // -0.25 to +0.20
        filter::brightness(&mut buf, offset).unwrap();

        // Sample pixel to show the effect.
        let sample = buf.data()[0];

        println!(
            "  frame {frame:>2}: base={base:>3} brightness={offset:>+.2} -> pixel[0].r={sample:>3}  pool.len()={} {}",
            pool.len(),
            if reused { "(reused)" } else { "(new)" },
        );

        // Release the underlying data back to the pool.
        pool.release(buf.into_data());
    }

    println!("\nFinal pool.len() = {} (bounded by max=4)", pool.len());
    assert!(pool.len() <= 4, "pool should never exceed max_buffers");

    // Demonstrate that acquire reuses allocated capacity.
    let buf = pool.acquire(frame_bytes);
    println!(
        "Re-acquired buffer: len={} capacity={} (capacity >= frame size: {})",
        buf.len(),
        buf.capacity(),
        buf.capacity() >= frame_bytes,
    );
}
