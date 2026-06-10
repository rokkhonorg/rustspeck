//! Core spectrogram DSP — a faithful port of the signal-processing half of
//! SoX's `spectrogram.c` (`start`, `make_window`, `flow`, `drain`, `do_column`).
//! Rendering lives in `render.rs`.

use rayon::prelude::*;

use crate::fft::Dft;
use crate::timeparse;
use crate::window::{self, WindowType};

pub const MAX_X_SIZE: i32 = 200_000;
pub const MAX_Y_SIZE: i32 = 200_000;
const ALT_PALETTE_LEN: i32 = 168; // array_length(alt_palette)/3

/// Spectrogram options, already normalised the way `getopts` leaves them
/// (gain negated, perm decremented, spectrum_points += 2, alt clamp applied).
#[derive(Clone, Debug)]
pub struct Config {
    pub pixels_per_sec: f64, // -X (0 = unset)
    pub window_adjust: f64,  // -W
    pub x_size0: i32,        // -x (0 = unset)
    pub y_size: i32,         // -y (0 = unset)
    pub big_y_size: i32,     // -Y (0 = unset)
    pub db_range: i32,       // -z
    pub gain: i32,           // -Z, already negated
    pub spectrum_points: i32, // -q, already +2
    pub perm: i32,           // -p, already decremented (0..5)
    pub monochrome: bool,
    pub light_background: bool,
    pub high_colour: bool,
    pub slack_overlap: bool,
    pub no_axes: bool,
    pub normalize: bool,
    pub raw: bool,
    pub alt_palette: bool,
    pub truncate: bool,
    pub win_type: WindowType,
    pub title: Option<String>,
    pub comment: String,
    pub duration_str: Option<String>,
    pub start_time_str: Option<String>,
    pub out_name: String,
}

impl Default for Config {
    fn default() -> Self {
        // Mirrors getopts defaults *before* the normalisation tail.
        Config {
            pixels_per_sec: 0.0,
            window_adjust: 0.0,
            x_size0: 0,
            y_size: 0,
            big_y_size: 0,
            db_range: 120,
            gain: 0,
            spectrum_points: 249,
            perm: 1,
            monochrome: false,
            light_background: false,
            high_colour: false,
            slack_overlap: false,
            no_axes: false,
            normalize: false,
            raw: false,
            alt_palette: false,
            truncate: false,
            win_type: WindowType::Hann,
            title: None,
            comment: "Created by RustSpeck".to_string(),
            duration_str: None,
            start_time_str: None,
            out_name: "spectrogram.png".to_string(),
        }
    }
}

impl Config {
    /// Apply the normalisation tail of `getopts` and validate cross-option
    /// constraints. Call once after collecting raw CLI values.
    pub fn finalize(mut self) -> Result<Config, String> {
        if (self.x_size0 != 0) as i32
            + (self.pixels_per_sec != 0.0) as i32
            + (self.duration_str.is_some()) as i32
            > 2
        {
            return Err("only two of width (-x), pixels/sec (-X), duration (-d) may be given".into());
        }
        if self.y_size != 0 && self.big_y_size != 0 {
            return Err("only one of height (-y), total-height (-Y) may be given".into());
        }
        self.gain = -self.gain;
        self.perm -= 1;
        self.spectrum_points += 2;
        if self.alt_palette {
            self.spectrum_points = self.spectrum_points.min(ALT_PALETTE_LEN);
        }
        Ok(self)
    }
}

/// Geometry + transform setup, shared by all channels (the `start` function).
struct Geometry {
    dft_size: i32,
    step_size: i32,
    block_steps: i32,
    block_norm: f64,
    rows: i32,
    x_size: i32,
    skip: u64,
}

/// The finished spectrogram: per-channel dB data plus everything the renderer
/// needs.
pub struct Spectrogram {
    pub cfg: Config,
    pub chans: usize,
    pub rows: i32,
    pub cols: i32,
    pub step_size: i32,
    pub block_steps: i32,
    pub rate: f64,
    /// Per-channel dBfs, each of length rows*cols, column-major as in SoX:
    /// `dbfs[ch][col*rows + row]`.
    pub dbfs: Vec<Vec<f32>>,
    pub max: f64,
}

/// Per-channel running window state (`make_window` operates on this).
struct WindowState {
    dft_size: i32,
    window: Vec<f64>, // length dft_size + 1
}

impl WindowState {
    /// Port of `make_window`; returns the window-density sum.
    fn make(&mut self, cfg: &Config, end: i32) -> f64 {
        let dft_size = self.dft_size;
        let n0 = 1 + dft_size - end.abs();
        let off = if end < 0 { 0 } else { end } as usize;

        if end != 0 {
            for w in self.window.iter_mut() {
                *w = 0.0;
            }
        }
        for i in 0..n0 as usize {
            self.window[off + i] = 1.0;
        }

        let w = &mut self.window[off..off + n0 as usize];
        match cfg.win_type {
            WindowType::Hann => window::apply_hann(w),
            WindowType::Hamming => window::apply_hamming(w),
            WindowType::Bartlett => window::apply_bartlett(w),
            WindowType::Rectangular => {}
            WindowType::Kaiser => {
                let beta = window::kaiser_beta(
                    (cfg.db_range + cfg.gain) as f64 * (1.1 + cfg.window_adjust / 50.0),
                    0.1,
                );
                window::apply_kaiser(w, beta);
            }
            WindowType::Dolph => {
                window::apply_dolph(
                    w,
                    (cfg.db_range + cfg.gain) as f64 * (1.005 + cfg.window_adjust / 50.0) + 6.0,
                );
            }
        }

        let mut sum = 0.0;
        for i in 0..dft_size as usize {
            sum += self.window[i];
        }
        let n = n0 - 1;
        let scale = 2.0 / sum * sqr(n as f64 / dft_size as f64);
        for i in 0..dft_size as usize {
            self.window[i] *= scale;
        }
        sum
    }
}

#[inline]
fn sqr(x: f64) -> f64 {
    x * x
}

/// One channel's analysis state machine (the per-flow part of `priv_t`).
struct Channel {
    g_dft_size: i32,
    g_step_size: i32,
    g_block_steps: i32,
    rows: i32,
    x_size: i32,
    truncate: bool,

    skip: u64,
    read: i32,
    end: i32,
    end_min: i32,
    last_end: i32,
    block_num: i32,
    cols: i32,
    truncated: bool,
    block_norm: f64,
    max: f64,

    buf: Vec<f64>,
    dft_buf: Vec<f64>,
    magnitudes: Vec<f64>,
    dbfs: Vec<f32>,
    win: WindowState,
    gain_val: f64,

    dft: Dft,
}

impl Channel {
    /// `do_column`: emit one spectrogram column from the accumulated magnitudes.
    /// Returns false when the image is full (the SOX_EOF case).
    fn do_column(&mut self) -> bool {
        if self.cols == self.x_size {
            self.truncated = true;
            return !self.truncate; // EOF (stop) iff truncate requested
        }
        self.cols += 1;
        let base = ((self.cols - 1) * self.rows) as usize;
        self.dbfs.resize((self.cols * self.rows) as usize, 0.0);
        for i in 0..self.rows as usize {
            let db = 10.0 * (self.magnitudes[i] * self.block_norm).log10();
            self.dbfs[base + i] = (db + self.gain_val) as f32;
            if db > self.max {
                self.max = db;
            }
        }
        for m in self.magnitudes.iter_mut() {
            *m = 0.0;
        }
        self.block_num = 0;
        true
    }

    fn process_block(&mut self, cfg: &Config) -> bool {
        // window update
        self.end = self.end.max(self.end_min);
        if self.end != self.last_end {
            self.last_end = self.end;
            self.win.make(cfg, self.end);
        }
        for i in 0..self.g_dft_size as usize {
            self.dft_buf[i] = self.buf[i] * self.win.window[i];
        }
        self.dft.accumulate_power(&self.dft_buf, &mut self.magnitudes);
        self.block_num += 1;
        if self.block_num == self.g_block_steps {
            return self.do_column();
        }
        true
    }

    /// `flow`: feed `len` samples (already in [-1,1) float domain). Honors skip.
    fn flow(&mut self, samples: &[f64], cfg: &Config) {
        let mut idx = 0usize;
        let mut len = samples.len();

        if self.skip > 0 {
            if self.skip as usize >= len {
                self.skip -= len as u64;
                return;
            }
            idx += self.skip as usize;
            len -= self.skip as usize;
            self.skip = 0;
        }

        let step = self.g_step_size;
        let dft = self.g_dft_size;
        while !self.truncated {
            if self.read == step {
                // shift left by step_size
                self.buf.copy_within((step as usize)..(dft as usize), 0);
                self.read = 0;
            }
            while len > 0 && self.read < step {
                let bi = (dft - step + self.read) as usize;
                self.buf[bi] = samples[idx];
                idx += 1;
                len -= 1;
                self.read += 1;
                self.end -= 1;
            }
            if self.read != step {
                break;
            }
            if !self.process_block(cfg) {
                return; // EOF
            }
        }
    }

    /// `drain`: flush trailing partial block with zero padding.
    fn drain(&mut self, cfg: &Config) {
        if self.truncated {
            return;
        }
        let dft = self.g_dft_size;
        let step = self.g_step_size;
        let mut isamp = ((dft - step) / 2) as i64;
        let left_over = (isamp + self.read as i64).rem_euclid(step as i64);
        if left_over >= (step >> 1) as i64 {
            isamp += (step as i64) - left_over;
        }
        self.end = 0;
        self.end_min = -dft;
        let zeros = vec![0.0f64; isamp.max(0) as usize];
        // flow returns SUCCESS unless it hit EOF; we approximate by checking
        // truncated afterwards (matching the C `== SOX_SUCCESS && block_num`).
        let before_truncated = self.truncated;
        self.flow(&zeros, cfg);
        let success = !self.truncated || before_truncated;
        if success && self.block_num != 0 {
            self.block_norm *= self.g_block_steps as f64 / self.block_num as f64;
            self.do_column();
        }
    }
}

pub fn process(cfg: Config, rate: f64, channels: u32, samples: &[i32]) -> Result<Spectrogram, String> {
    let chans = channels as usize;
    if chans == 0 {
        return Err("input has no channels".into());
    }
    let total_len = samples.len() as u64;
    let per_chan_len = total_len / channels as u64;

    let geom = compute_geometry(&cfg, rate, channels, total_len)?;

    // Channels are fully independent, so process them in parallel. Each task
    // gets its own FFT plan + work buffers; the output is identical to doing
    // them sequentially.
    let per_channel: Vec<ChannelOutput> = (0..chans)
        .into_par_iter()
        .map(|ch| {
            // deinterleave this channel to the [-1, 1) float domain
            let mut mono = Vec::with_capacity(per_chan_len as usize);
            let mut f = ch;
            while f < samples.len() {
                mono.push(samples[f] as f64 * (1.0 / (0x7FFF_FFFFu32 as f64 + 1.0)));
                f += chans;
            }

            let dft = Dft::new(geom.dft_size as usize);
            let mut channel = make_channel(&cfg, &geom, dft);
            channel.flow(&mono, &cfg);
            channel.drain(&cfg);

            ChannelOutput {
                dbfs: channel.dbfs,
                max: channel.max,
                cols: channel.cols,
            }
        })
        .collect();

    // SoX uses channel 0's max for -n normalisation (autogain = -p->max where p
    // is flow 0's priv), so we take that channel's max specifically. All channels
    // produce the same column count.
    let ch0_max = per_channel
        .first()
        .map_or(-(cfg.db_range as f64), |c| c.max);
    let cols = per_channel.first().map_or(0, |c| c.cols);
    let dbfs_all: Vec<Vec<f32>> = per_channel.into_iter().map(|c| c.dbfs).collect();

    Ok(Spectrogram {
        chans,
        rows: geom.rows,
        cols,
        step_size: geom.step_size,
        block_steps: geom.block_steps,
        rate,
        dbfs: dbfs_all,
        max: ch0_max,
        cfg,
    })
}

/// Per-channel analysis result, collected from the parallel pass.
struct ChannelOutput {
    dbfs: Vec<f32>,
    max: f64,
    cols: i32,
}

/// `sox_sample_t` (i32, full-scale ±2^31) → the [-1, 1) float domain SoX's DSP
/// works in. Same scale used by the batch `process` path above.
const SAMPLE_SCALE: f64 = 1.0 / (0x7FFF_FFFFu32 as f64 + 1.0);

/// Streaming variant of [`process`]: the geometry is fixed up front (the total
/// length must be known), then sample chunks are fed in with [`push`](StreamProcessor::push)
/// and columns are produced incrementally. Used by the GUI to render a
/// spectrogram progressively as the file decodes. The final pixels are identical
/// to the batch path for the same input.
pub struct StreamProcessor {
    cfg: Config,
    geom: Geometry,
    chans: usize,
    rate: f64,
    channels: Vec<Channel>,
}

impl StreamProcessor {
    /// `total_len` is the interleaved sample count of the whole input (frames ×
    /// channels) — required to fix the time geometry before any samples arrive.
    pub fn new(cfg: Config, rate: f64, channels: u32, total_len: u64) -> Result<Self, String> {
        let chans = channels as usize;
        if chans == 0 {
            return Err("input has no channels".into());
        }
        let geom = compute_geometry(&cfg, rate, channels, total_len)?;
        let chan_states = (0..chans)
            .map(|_| make_channel(&cfg, &geom, Dft::new(geom.dft_size as usize)))
            .collect();
        Ok(StreamProcessor {
            cfg,
            geom,
            chans,
            rate,
            channels: chan_states,
        })
    }

    /// Feed an interleaved chunk of `sox_sample_t` samples. Channels are advanced
    /// in parallel (each owns its own FFT plan + buffers).
    pub fn push(&mut self, interleaved: &[i32]) {
        let chans = self.chans;
        let cfg = &self.cfg;
        self.channels.par_iter_mut().enumerate().for_each(|(ch, c)| {
            let mono: Vec<f64> = interleaved[ch..]
                .iter()
                .step_by(chans)
                .map(|&s| s as f64 * SAMPLE_SCALE)
                .collect();
            c.flow(&mono, cfg);
        });
    }

    /// Flush the trailing partial block on every channel (the `drain` step).
    pub fn finish(&mut self) {
        let cfg = &self.cfg;
        self.channels.par_iter_mut().for_each(|c| c.drain(cfg));
    }

    /// Columns produced so far (all channels advance in lockstep on the same
    /// input, so channel 0 is representative).
    pub fn cols_done(&self) -> i32 {
        self.channels.first().map_or(0, |c| c.cols)
    }

    pub fn x_size(&self) -> i32 {
        self.geom.x_size
    }
    pub fn rows(&self) -> i32 {
        self.geom.rows
    }
    pub fn chans(&self) -> usize {
        self.chans
    }
    pub fn rate(&self) -> f64 {
        self.rate
    }
    pub fn step_size(&self) -> i32 {
        self.geom.step_size
    }
    pub fn block_steps(&self) -> i32 {
        self.geom.block_steps
    }
    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// One channel's accumulated dBfs columns so far (column-major, length
    /// `cols_done * rows`).
    pub fn channel_dbfs(&self, ch: usize) -> &[f32] {
        &self.channels[ch].dbfs
    }

}

fn compute_geometry(
    cfg: &Config,
    rate: f64,
    channels: u32,
    total_len: u64,
) -> Result<Geometry, String> {
    let mut duration = 0.0f64;
    let mut start_time = 0.0f64;
    let mut pixels_per_sec = cfg.pixels_per_sec;
    let mut skip = 0u64;

    if let Some(ds) = &cfg.duration_str {
        let d = timeparse::parse_samples(rate, ds, b't')?;
        duration = d as f64 / rate;
    }

    if let Some(ss) = &cfg.start_time_str {
        let in_length = total_len / channels as u64; // per-channel length (known for WAV)
        let d = timeparse::parse_position(rate, ss, 0, in_length, b'=')?;
        start_time = d as f64 / rate;
        skip = d;
    }

    let mut x_size = cfg.x_size0;

    loop {
        if pixels_per_sec == 0.0 && x_size != 0 && duration != 0.0 {
            pixels_per_sec = (5000.0f64).min(x_size as f64 / duration);
        } else if x_size == 0 && pixels_per_sec != 0.0 && duration != 0.0 {
            x_size = (MAX_X_SIZE as f64).min((pixels_per_sec * duration + 0.5).floor()) as i32;
        }

        if duration == 0.0 {
            // length is always known for a WAV file
            duration = total_len as f64 / (rate * channels as f64);
            duration -= start_time;
            if duration <= 0.0 {
                duration = 1.0;
            }
            continue;
        } else if x_size == 0 {
            x_size = 800;
            continue;
        } else if pixels_per_sec == 0.0 {
            pixels_per_sec = 100.0;
            continue;
        }
        break;
    }

    let dft_size: i32;
    if cfg.y_size != 0 {
        dft_size = 2 * (cfg.y_size - 1);
    } else {
        let y = 32.max((if cfg.big_y_size != 0 { cfg.big_y_size } else { 550 }) / channels as i32 - 2);
        let mut d = 128;
        while d <= y {
            d <<= 1;
        }
        dft_size = d;
    }

    let rows = (dft_size >> 1) + 1;

    // window density via a throwaway window state
    let mut win = WindowState {
        dft_size,
        window: vec![0.0; (dft_size + 1) as usize],
    };
    let actual = win.make(cfg, 0);

    let mut step_size = if cfg.slack_overlap {
        ((actual * dft_size as f64).sqrt() + 0.5) as i32
    } else {
        (actual + 0.5) as i32
    };
    let mut block_steps = (rate / pixels_per_sec).max(1.0) as i32;
    step_size = (block_steps as f64 / (block_steps as f64 / step_size as f64).ceil() + 0.5) as i32;
    block_steps = (block_steps as f64 / step_size as f64 + 0.5).floor() as i32;
    let block_norm = 1.0 / block_steps as f64;

    if std::env::var("SPEK_DEBUG").is_ok() {
        eprintln!(
            "duration_pps={pixels_per_sec} x_size={x_size} dft_size={dft_size} actual={actual} step_size={step_size} block_steps={block_steps}"
        );
    }

    Ok(Geometry {
        dft_size,
        step_size,
        block_steps,
        block_norm,
        rows,
        x_size,
        skip,
    })
}

fn make_channel(cfg: &Config, geom: &Geometry, dft: Dft) -> Channel {
    let dft_size = geom.dft_size;
    let mut win = WindowState {
        dft_size,
        window: vec![0.0; (dft_size + 1) as usize],
    };
    win.make(cfg, 0); // last_end = 0

    Channel {
        g_dft_size: dft_size,
        g_step_size: geom.step_size,
        g_block_steps: geom.block_steps,
        rows: geom.rows,
        x_size: geom.x_size,
        truncate: cfg.truncate,
        skip: geom.skip,
        read: (geom.step_size - dft_size) / 2,
        end: dft_size,
        end_min: 0,
        last_end: 0,
        block_num: 0,
        cols: 0,
        truncated: false,
        block_norm: geom.block_norm,
        max: -(cfg.db_range as f64),
        buf: vec![0.0; dft_size as usize],
        dft_buf: vec![0.0; dft_size as usize],
        magnitudes: vec![0.0; ((dft_size >> 1) + 1) as usize],
        dbfs: Vec::new(),
        win,
        gain_val: cfg.gain as f64,
        dft,
    }
}
