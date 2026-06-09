//! rustspek — a Rust port of SoX's `spectrogram` effect.
//!
//! Produces a PNG spectrogram bit-compatible with
//! `sox <in> -n spectrogram [options]`, with a friendlier CLI. Input decoding is
//! handled by Symphonia, so any format it supports works (WAV, FLAC, MP3, …).

mod audio;
mod fft;
mod render;
mod spectrogram;
mod tables;
mod timeparse;
mod window;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use spectrogram::{Config, MAX_X_SIZE, MAX_Y_SIZE};
use window::WindowType;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum WindowArg {
    Hann,
    Hamming,
    Bartlett,
    Rectangular,
    Kaiser,
    Dolph,
}

impl From<WindowArg> for WindowType {
    fn from(w: WindowArg) -> Self {
        match w {
            WindowArg::Hann => WindowType::Hann,
            WindowArg::Hamming => WindowType::Hamming,
            WindowArg::Bartlett => WindowType::Bartlett,
            WindowArg::Rectangular => WindowType::Rectangular,
            WindowArg::Kaiser => WindowType::Kaiser,
            WindowArg::Dolph => WindowType::Dolph,
        }
    }
}

/// Generate a spectrogram PNG from an audio file (a port of SoX `spectrogram`).
///
/// Decodes any format Symphonia supports (WAV, FLAC, MP3, OGG/Vorbis, AAC,
/// ALAC, AIFF, CAF, …). At most two of --width, --pixels-per-sec and --duration
/// may be combined, and only one of --height / --total-height.
#[derive(Parser, Debug)]
#[command(name = "rustspek", version, about, long_about = None)]
struct Args {
    /// Input audio file (WAV, FLAC, MP3, OGG, AAC, ALAC, …)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output PNG file ("-" for stdout)
    #[arg(short = 'o', long, value_name = "FILE", default_value = "spectrogram.png")]
    output: String,

    // --- Horizontal (time) axis ---
    /// X-axis size in pixels (100-200000); default derived, else 800
    #[arg(short = 'x', long, value_name = "PX",
          value_parser = clap::value_parser!(i32).range(100..=MAX_X_SIZE as i64))]
    width: Option<i32>,

    /// X-axis pixels per second (1-5000); default derived, else 100
    #[arg(short = 'X', long = "pixels-per-sec", value_name = "N")]
    pixels_per_sec: Option<f64>,

    /// Audio duration to fit to the X-axis, e.g. 1:00, 48, 0.5
    #[arg(short = 'd', long, value_name = "TIME")]
    duration: Option<String>,

    /// Start the spectrogram at this input position, e.g. 0:30, 10s
    #[arg(short = 'S', long, value_name = "POS")]
    start: Option<String>,

    // --- Vertical (frequency) axis ---
    /// Y-axis size per channel in pixels (64+); fast when 1 + 2^n
    #[arg(short = 'y', long, value_name = "PX",
          value_parser = clap::value_parser!(i32).range(64..=MAX_Y_SIZE as i64))]
    height: Option<i32>,

    /// Total Y height across all channels (130+); default 550
    #[arg(short = 'Y', long = "total-height", value_name = "PX",
          value_parser = clap::value_parser!(i32).range(130..=MAX_Y_SIZE as i64))]
    total_height: Option<i32>,

    // --- Levels (Z-axis) ---
    /// Z-axis dynamic range in dB (20-180); default 120
    #[arg(short = 'z', long = "db-range", value_name = "DB",
          value_parser = clap::value_parser!(i32).range(20..=180))]
    db_range: Option<i32>,

    /// Z-axis maximum in dBFS (-100..100); default 0
    #[arg(short = 'Z', long = "db-max", value_name = "DBFS", allow_hyphen_values = true,
          value_parser = clap::value_parser!(i32).range(-100..=100))]
    db_max: Option<i32>,

    /// Set Z-axis maximum to the brightest pixel
    #[arg(short = 'n', long)]
    normalize: bool,

    /// Z-axis colour quantisation / number of colours (0-249); default 249
    #[arg(short = 'q', long = "colors", visible_alias = "quantisation", value_name = "N",
          value_parser = clap::value_parser!(i32).range(0..=249))]
    colors: Option<i32>,

    // --- Analysis window ---
    /// Window function
    #[arg(short = 'w', long, value_enum, ignore_case = true,
          default_value_t = WindowArg::Hann, value_name = "NAME")]
    window: WindowArg,

    /// Window adjustment parameter (-10..10); Kaiser/Dolph only
    #[arg(short = 'W', long = "window-adjust", value_name = "N", allow_hyphen_values = true)]
    window_adjust: Option<f64>,

    /// Slacken the overlap of windows
    #[arg(short = 's', long = "slack-overlap")]
    slack_overlap: bool,

    // --- Appearance ---
    /// Light background
    #[arg(short = 'l', long = "light-background")]
    light_background: bool,

    /// Monochrome
    #[arg(short = 'm', long)]
    monochrome: bool,

    /// High colour mode
    #[arg(long = "high-color", visible_alias = "high-colour")]
    high_colour: bool,

    /// Permute colours (1-6); default 1
    #[arg(short = 'p', long = "permute", value_name = "N",
          value_parser = clap::value_parser!(i32).range(1..=6))]
    permute: Option<i32>,

    /// Use the alternative (inferior) fixed colour set (compatibility only)
    #[arg(short = 'A', long = "alt-palette")]
    alt_palette: bool,

    /// Suppress axis lines (keep labels)
    #[arg(short = 'a', long = "no-axis-lines")]
    no_axis_lines: bool,

    /// Raw spectrogram: no axes, labels or legend
    #[arg(short = 'r', long)]
    raw: bool,

    /// Stop output as soon as the X-axis is full (truncate)
    #[arg(short = 'T', long)]
    truncate: bool,

    /// Title text drawn above the image
    #[arg(short = 't', long, value_name = "TEXT")]
    title: Option<String>,

    /// Footer comment text; default "Created by SoX"
    #[arg(short = 'c', long, value_name = "TEXT")]
    comment: Option<String>,
}

fn build_config(args: &Args) -> Result<Config, String> {
    let mut cfg = Config::default();

    cfg.out_name = args.output.clone();
    if let Some(v) = args.width {
        cfg.x_size0 = v;
    }
    if let Some(v) = args.pixels_per_sec {
        if !(1.0..=5000.0).contains(&v) {
            return Err("pixels-per-sec (-X) must be between 1 and 5000".into());
        }
        cfg.pixels_per_sec = v;
    }
    cfg.duration_str = args.duration.clone();
    cfg.start_time_str = args.start.clone();
    if let Some(v) = args.height {
        cfg.y_size = v;
    }
    if let Some(v) = args.total_height {
        cfg.big_y_size = v;
    }
    if let Some(v) = args.db_range {
        cfg.db_range = v;
    }
    if let Some(v) = args.db_max {
        cfg.gain = v;
    }
    cfg.normalize = args.normalize;
    if let Some(v) = args.colors {
        cfg.spectrum_points = v;
    }
    cfg.win_type = args.window.into();
    if let Some(v) = args.window_adjust {
        if !(-10.0..=10.0).contains(&v) {
            return Err("window-adjust (-W) must be between -10 and 10".into());
        }
        cfg.window_adjust = v;
    }
    cfg.slack_overlap = args.slack_overlap;
    cfg.light_background = args.light_background;
    cfg.monochrome = args.monochrome;
    cfg.high_colour = args.high_colour;
    if let Some(v) = args.permute {
        cfg.perm = v;
    }
    cfg.alt_palette = args.alt_palette;
    cfg.no_axes = args.no_axis_lines;
    cfg.raw = args.raw;
    cfg.truncate = args.truncate;
    cfg.title = args.title.clone();
    if let Some(c) = &args.comment {
        cfg.comment = c.clone();
    }

    cfg.finalize()
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    let cfg = build_config(&args)?;
    let out_name = cfg.out_name.clone();

    let audio = audio::read(&args.input).map_err(|e| format!("{}: {e}", args.input.display()))?;
    if audio.frames() == 0 {
        return Err("input file contains no audio samples".into());
    }

    let spec = spectrogram::process(cfg, audio.rate, audio.channels, &audio.samples)?;

    if out_name == "-" {
        let stdout = io::stdout();
        let w = BufWriter::new(stdout.lock());
        render::render_png(&spec, w)?;
    } else {
        let file =
            File::create(&out_name).map_err(|e| format!("failed to create {out_name}: {e}"))?;
        let w = BufWriter::new(file);
        render::render_png(&spec, w)?;
        eprintln!(
            "wrote {out_name} ({} channel{})",
            spec.chans,
            if spec.chans == 1 { "" } else { "s" }
        );
    }

    let _ = io::stdout().flush();
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rustspek: error: {e}");
            ExitCode::FAILURE
        }
    }
}
