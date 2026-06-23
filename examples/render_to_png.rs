//! Minimal library usage: decode an audio file and write its spectrogram PNG.
//!
//! Run with:  cargo run --example render_to_png -- test/stereo.wav out.png
//!
//! This uses only the core (image-output) API, so it builds without the GUI:
//!   cargo run --no-default-features --example render_to_png -- in.wav out.png

use std::path::PathBuf;

use rustspeck::Config;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or("usage: render_to_png <input-audio> [out.png]")?;
    let output = args.next().unwrap_or_else(|| "spectrogram.png".to_string());

    // Config::default() mirrors the CLI defaults; adjust fields before passing
    // it in (it is finalized internally). For example, to set a title:
    let mut cfg = Config::default();
    cfg.title = Some(PathBuf::from(&input).display().to_string());

    // RgbaImage path — flexible if you want to manipulate pixels:
    let img = rustspeck::render_file(&input, &cfg)?;
    img.save(&output).map_err(|e| format!("failed to write {output}: {e}"))?;
    eprintln!("wrote {output} ({}x{})", img.width(), img.height());

    // PNG-bytes path (identical bytes to the CLI) is also available:
    //   let bytes = rustspeck::render_file_png(&input, &cfg)?;

    Ok(())
}
