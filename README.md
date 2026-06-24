# RustSpeck

An audio spectrogram renderer and viewer. It produces PNG/RGBA spectrograms
(output compatible with SoX's `spectrogram` effect, whose DSP it grew from),
with a friendly CLI and an optional [gpui](https://www.gpui.rs/)-based GUI
viewer. Audio decoding is handled by [Symphonia](https://github.com/pdeljanov/Symphonia),
so WAV, FLAC, MP3, OGG/Vorbis, AAC, ALAC, AIFF and more all work.

It can be used three ways: as a **GUI viewer**, as a **CLI** that writes PNGs,
and as a **library**.

## Build

```sh
cargo build --release
# binary at target/release/rustspeck
```

### FFmpeg fallback (optional, runtime)

Decoding is handled by [Symphonia](https://github.com/pdeljanov/Symphonia)
(pure Rust). For the occasional container/codec Symphonia doesn't support (e.g.
Opus, WavPack), RustSpeck automatically falls back to the system **`ffmpeg`**
binary if it's found on `PATH` — no build flag and no FFmpeg libraries required,
just the regular `ffmpeg`/`ffprobe` executables installed (e.g. `winget install
Gyan.FFmpeg`, `brew install ffmpeg`, `apt install ffmpeg`). If `ffmpeg` isn't
installed, an unsupported file simply reports the Symphonia error with a hint.

## GUI viewer (default)

Open a file in the interactive viewer — just pass an input (or launch with no
arguments and drag a file in):

```sh
rustspeck song.flac
```

With a track loaded, these keys re-render it live:

| Key | Action |
|-----|--------|
| `f` / `Shift+F` | step the FFT size up / down (powers of two) |
| `w` | cycle the window function |
| `c` | cycle the colour palette |

## CLI (render a PNG)

Pass `--output`/`-o` to render a PNG instead of opening the viewer (`-o -`
writes PNG to stdout):

```sh
rustspeck song.flac -o spectrogram.png
```

Common options (run `rustspeck --help` for the full list):

| Flag | Meaning |
|------|---------|
| `-x, --width <PX>` | X-axis width in pixels |
| `-X, --pixels-per-sec <N>` | horizontal time resolution |
| `-d, --duration <TIME>` | fit this much audio to the X-axis (e.g. `1:30`) |
| `-S, --start <POS>` | start at this input position |
| `-y, --height <PX>` | Y-axis (frequency) height per channel |
| `-F, --fft-size <N>` | FFT window size in points (finer freq vs. time) |
| `-z, --db-range <DB>` | dynamic range in dB |
| `-Z, --db-max <DBFS>` | level mapped to the brightest colour |
| `-w, --window <NAME>` | window function (hann, hamming, kaiser, …) |
| `--palette <NAME>` | colour gradient: sox (default), viridis, magma, inferno, plasma, grayscale, green, amber |
| `-m, --monochrome` / `-l, --light-background` | appearance |
| `-t, --title <TEXT>` | title drawn above the image |

```sh
# 10 s starting at 0:30, taller FFT, custom title:
rustspeck track.mp3 -o out.png -S 0:30 -d 10 -F 4096 -t "Track"
```

## Library

Add it as a dependency. If you only want image output, disable the default
`gui` feature so the GUI stack (gpui and friends) isn't pulled in:

```toml
[dependencies]
rustspeck = { git = "https://github.com/rokkhonorg/rustspeck", default-features = false }
```

`Config::default()` mirrors the CLI defaults; adjust its fields, then call one
of the convenience renderers (they finalize the config internally):

```rust
use rustspeck::Config;

let mut cfg = Config::default();
cfg.title = Some("My Track".into());

// Decoded image you can resize / composite / save:
let img = rustspeck::render_file("song.flac", &cfg)?;   // image::RgbaImage
img.save("out.png")?;

// Or encoded PNG bytes, identical to the CLI's output:
let png: Vec<u8> = rustspeck::render_file_png("song.flac", &cfg)?;
# Ok::<(), String>(())
```

For finer control (streaming, custom layout, reusing a decoded buffer) drop
down to [`rustspeck::audio`], [`rustspeck::spectrogram`] and
[`rustspeck::render`]. See `examples/render_to_png.rs` for a runnable example.

## License

LGPL-2.1-or-later. RustSpeck is a derivative of SoX's spectrogram effect
(`spectrogram.c`, © 2008-2009 robs@users.sourceforge.net), so it carries SoX's
license. See [LICENSE](LICENSE).

Because LGPL is weak copyleft, applications that merely depend on the
`rustspeck` crate (via Cargo) can use their own license — the LGPL terms cover
RustSpeck itself, not works that link to it.
