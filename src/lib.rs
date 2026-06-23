//! RustSpeck — an audio spectrogram renderer and viewer, usable as a library.
//! Output is compatible with SoX's `spectrogram` effect, whose DSP it grew from.
//!
//! The pipeline is: decode an audio file ([`audio`]), compute its spectrogram
//! ([`spectrogram::process`]), then render it to a PNG or an RGBA image
//! ([`render`]). For the common "file in, image out" case use the convenience
//! functions on this module:
//!
//! ```no_run
//! let cfg = rustspeck::Config::default();
//! let img = rustspeck::render_file("song.flac", &cfg)?;   // image::RgbaImage
//! img.save("out.png").unwrap();
//!
//! let png_bytes = rustspeck::render_file_png("song.flac", &cfg)?; // Vec<u8>
//! # Ok::<(), String>(())
//! ```
//!
//! `Config::default()` mirrors the CLI defaults; tweak its fields before passing
//! it in (don't call [`Config::finalize`] yourself — these helpers finalize a
//! clone internally). For finer control — streaming, custom output layout, or
//! reusing a decoded buffer — drop down to [`audio::read`], [`spectrogram`] and
//! [`render`] directly.
//!
//! The interactive GUI viewer ([`gui`]) lives behind the default-on `gui`
//! feature, which pulls in gpui and friends. Library consumers that only want
//! image output should disable default features:
//!
//! ```toml
//! rustspeck = { git = "...", default-features = false }
//! ```

pub mod audio;
pub mod render;
pub mod spectrogram;
pub mod window;

mod fft;
mod tables;
mod timeparse;

#[cfg(feature = "gui")]
pub mod gui;

use std::path::Path;

pub use image::RgbaImage;
pub use spectrogram::{Config, MAX_X_SIZE, MAX_Y_SIZE, Spectrogram, StreamProcessor, process};
pub use window::WindowType;

/// Decode `path`, compute its spectrogram with `cfg`, and return the rendered
/// image as an [`image::RgbaImage`] (the same pixels as the PNG output).
///
/// `cfg` is taken as the un-finalized configuration you'd build from CLI-style
/// options; it is cloned and [`Config::finalize`]d internally, so pass a fresh
/// `Config` (e.g. `Config::default()` with fields adjusted), not one you've
/// already finalized.
pub fn render_file<P: AsRef<Path>>(path: P, cfg: &Config) -> Result<RgbaImage, String> {
    render::render_rgba(&compute(path, cfg)?)
}

/// Like [`render_file`] but returns encoded PNG bytes, byte-for-byte identical
/// to what the CLI writes for the same options.
pub fn render_file_png<P: AsRef<Path>>(path: P, cfg: &Config) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    render::render_png(&compute(path, cfg)?, &mut buf)?;
    Ok(buf)
}

/// Shared decode + process step for the convenience renderers.
fn compute<P: AsRef<Path>>(path: P, cfg: &Config) -> Result<Spectrogram, String> {
    let cfg = cfg.clone().finalize()?;
    let audio = audio::read(path.as_ref())?;
    if audio.frames() == 0 {
        return Err("input file contains no audio samples".into());
    }
    process(cfg, audio.rate, audio.channels, &audio.samples)
}
