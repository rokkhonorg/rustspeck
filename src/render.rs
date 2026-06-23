//! Rendering — a faithful port of the `stop` function of `spectrogram.c`:
//! palette generation, the embedded zlib-compressed bitmap font, axis drawing
//! and 8-bit palette PNG output.

use std::io::{Read, Write};

use crate::spectrogram::{Config, Spectrogram, StreamProcessor};
use crate::tables::{ALT_PALETTE, FIXED_FONT_ZLIB};

// Layout constants (#defines in spectrogram.c)
const BELOW: i32 = 48;
const LEFT: i32 = 58;
const BETWEEN: i32 = 37;
const SPECTRUM_WIDTH: i32 = 14;
const RIGHT: i32 = 35;

const FONT_X: i32 = 5;
const FONT_Y: i32 = 12;
const FONT_BIG_X: i32 = FONT_X + 1; // font_X in C

// Fixed palette slots
const BACKGROUND: u8 = 0;
const TEXT: u8 = 1;
const LABELS: u8 = 2;
const GRID: u8 = 3;
const FIXED_PALETTE: i32 = 4;

const ALT_PALETTE_LEN: i32 = (ALT_PALETTE.len() / 3) as i32;

struct Image {
    cols: i32,
    rows: i32,
    pixels: Vec<u8>, // y=0 at bottom, pixel(x,y) = pixels[y*cols + x]
    font: Vec<u8>,   // 96 * font_y bitmap rows
}

impl Image {
    #[inline]
    fn set(&mut self, x: i32, y: i32, c: u8) {
        if x >= 0 && x < self.cols && y >= 0 && y < self.rows {
            self.pixels[(y * self.cols + x) as usize] = c;
        }
    }

    /// print_at_ with orientation 0 (horizontal text, rows go upward as i grows
    /// down from baseline y).
    fn print_at(&mut self, mut x: i32, y: i32, c: u8, text: &str) {
        for ch in text.bytes() {
            let glyph = glyph_index(ch);
            let mut pos = glyph * FONT_Y as usize;
            for i in 0..FONT_Y {
                let mut line = self.font[pos] as u32;
                pos += 1;
                for j in 0..FONT_X {
                    if line & 0x80 != 0 {
                        self.set(x + j, y - i, c);
                    }
                    line <<= 1;
                }
            }
            x += FONT_BIG_X;
        }
    }

    /// print_at_ with orientation 1 (vertical text).
    fn print_up(&mut self, x: i32, mut y: i32, c: u8, text: &str) {
        for ch in text.bytes() {
            let glyph = glyph_index(ch);
            let mut pos = glyph * FONT_Y as usize;
            for i in 0..FONT_Y {
                let mut line = self.font[pos] as u32;
                pos += 1;
                for j in 0..FONT_X {
                    if line & 0x80 != 0 {
                        self.set(x + i, y + j, c);
                    }
                    line <<= 1;
                }
            }
            y += FONT_BIG_X;
        }
    }
}

fn glyph_index(ch: u8) -> usize {
    let c = if ch < b' ' || ch > b'~' { b'~' + 1 } else { ch };
    (c - b' ') as usize
}

/// colour(): map a dBfs value to a palette index.
fn colour(cfg: &Config, x: f64) -> u8 {
    let sp = cfg.spectrum_points;
    let c: i32 = if x < -(cfg.db_range as f64) {
        0
    } else if x >= 0.0 {
        sp - 1
    } else {
        (1.0 + (1.0 + x / cfg.db_range as f64) * (sp - 2) as f64) as i32
    };
    (FIXED_PALETTE + c) as u8
}

/// make_palette(): fill RGB triples for the 4 fixed slots + spectrum colours.
fn make_palette(cfg: &Config) -> Vec<u8> {
    const BLACK: [u8; 3] = [0x00, 0x00, 0x00];
    const DGREY: [u8; 3] = [0x3f, 0x3f, 0x3f];
    const MGREY: [u8; 3] = [0x7f, 0x7f, 0x7f];
    const LGREY: [u8; 3] = [0xbf, 0xbf, 0xbf];
    const WHITE: [u8; 3] = [0xff, 0xff, 0xff];
    const LBGND: [u8; 3] = [0xdd, 0xd8, 0xd0];
    const MBGND: [u8; 3] = [0xdf, 0xdf, 0xdf];

    let sp = cfg.spectrum_points;
    let total = (FIXED_PALETTE + sp) as usize;
    let mut pal = vec![0u8; total * 3];

    let mut put = |idx: i32, rgb: [u8; 3]| {
        let o = (idx * 3) as usize;
        pal[o] = rgb[0];
        pal[o + 1] = rgb[1];
        pal[o + 2] = rgb[2];
    };

    if cfg.light_background {
        put(0, if cfg.monochrome { MBGND } else { LBGND });
        put(1, BLACK);
        put(2, DGREY);
        put(3, DGREY);
    } else {
        put(0, BLACK);
        put(1, WHITE);
        put(2, LGREY);
        put(3, MGREY);
    }

    for i in 0..sp {
        let mut c = [0.0f64; 3];
        let x = i as f64 / (sp - 1) as f64;
        let at = if cfg.light_background { sp - 1 - i } else { i };

        if cfg.monochrome {
            c[0] = x;
            c[1] = x;
            c[2] = x;
            if cfg.high_colour {
                c[((1 + cfg.perm) % 3) as usize] = if x < 0.4 { 0.0 } else { 5.0 / 3.0 * (x - 0.4) };
                if cfg.perm < 3 {
                    c[((2 + cfg.perm) % 3) as usize] =
                        if x < 0.4 { 0.0 } else { 5.0 / 3.0 * (x - 0.4) };
                }
            }
            let o = ((FIXED_PALETTE + at) * 3) as usize;
            pal[o] = (0.5 + 255.0 * c[0]) as u8;
            pal[o + 1] = (0.5 + 255.0 * c[1]) as u8;
            pal[o + 2] = (0.5 + 255.0 * c[2]) as u8;
            continue;
        }

        if cfg.high_colour {
            const STATES: [[i32; 7]; 3] = [
                [4, 5, 0, 0, 2, 1, 1],
                [0, 0, 2, 1, 1, 3, 2],
                [4, 1, 1, 3, 0, 0, 2],
            ];
            let phase_num = ((7.0 * x) as i32).min(6);
            for j in 0..3 {
                c[j] = match STATES[j][phase_num as usize] {
                    0 => 0.0,
                    1 => 1.0,
                    2 => ((7.0 * x - phase_num as f64) * std::f64::consts::PI / 2.0).sin(),
                    3 => ((7.0 * x - phase_num as f64) * std::f64::consts::PI / 2.0).cos(),
                    4 => 7.0 * x - phase_num as f64,
                    5 => 1.0 - (7.0 * x - phase_num as f64),
                    _ => 0.0,
                };
            }
        } else if cfg.alt_palette {
            let n = (i as f64 / (sp - 1) as f64 * (ALT_PALETTE_LEN - 1) as f64 + 0.5) as i32;
            c[0] = ALT_PALETTE[(3 * n) as usize] as f64 / 255.0;
            c[1] = ALT_PALETTE[(3 * n + 1) as usize] as f64 / 255.0;
            c[2] = ALT_PALETTE[(3 * n + 2) as usize] as f64 / 255.0;
        } else {
            c[0] = if x < 0.13 {
                0.0
            } else if x < 0.73 {
                ((x - 0.13) / 0.60 * std::f64::consts::PI / 2.0).sin()
            } else {
                1.0
            };
            c[1] = if x < 0.60 {
                0.0
            } else if x < 0.91 {
                ((x - 0.60) / 0.31 * std::f64::consts::PI / 2.0).sin()
            } else {
                1.0
            };
            c[2] = if x < 0.60 {
                0.5 * ((x - 0.00) / 0.60 * std::f64::consts::PI).sin()
            } else if x < 0.78 {
                0.0
            } else {
                (x - 0.78) / 0.22
            };
        }

        let perm = cfg.perm;
        let ri = (perm % 3) as usize;
        let gi = ((1 + perm + (perm % 2)) % 3) as usize;
        let bi = ((2 + perm - (perm % 2)) % 3) as usize;
        let o = ((FIXED_PALETTE + at) * 3) as usize;
        pal[o] = (0.5 + 255.0 * c[ri]) as u8;
        pal[o + 1] = (0.5 + 255.0 * c[gi]) as u8;
        pal[o + 2] = (0.5 + 255.0 * c[bi]) as u8;
    }

    pal
}

/// axis(): returns (step, limit, prefix-char-string).
fn axis(to_in: f64, max_steps: i32) -> (i32, f64, String) {
    let mut to = to_in;
    let mut scale = 1.0f64;
    let mut step = (1.0f64).max(10.0 * to);
    let mut prefix_num = 0i32;

    if max_steps != 0 {
        let mut log_10 = f64::INFINITY;
        to *= 10.0;
        let min_step = to / max_steps as f64;
        let mut i = 5;
        while i != 0 {
            let tryv = (min_step * i as f64).log10().ceil();
            if tryv <= log_10 {
                log_10 = tryv;
                step = 10f64.powf(log_10) / i as f64;
                log_10 -= if i > 1 { 1.0 } else { 0.0 };
            }
            i >>= 1;
        }
        prefix_num = (log_10 / 3.0).floor() as i32;
        scale = 10f64.powf(-3.0 * prefix_num as f64);
    }

    // prefix = "pnum-kMGTPE" + prefix_num + (prefix_num? 4 : 11)
    const PREFIX: &[u8] = b"pnum-kMGTPE";
    let idx = if prefix_num != 0 { prefix_num + 4 } else { 11 };
    let prefix = if idx >= 0 && (idx as usize) < PREFIX.len() {
        (PREFIX[idx as usize] as char).to_string()
    } else {
        String::new()
    };

    let limit = to * scale;
    let step_i = (step * scale + 0.5) as i32;
    (step_i, limit, prefix)
}

/// C printf "%g" with default precision 6.
fn fmt_g(val: f64, prec: usize) -> String {
    let p = if prec == 0 { 1 } else { prec };
    if val == 0.0 {
        return "0".to_string();
    }
    let exp = val.abs().log10().floor() as i32;
    if exp >= -4 && exp < p as i32 {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_trailing(&format!("{:.*}", decimals, val))
    } else {
        let s = format!("{:.*e}", p - 1, val);
        // Normalise to C style: mantissa with stripped zeros, exp with sign and
        // at least two digits.
        if let Some((m, e)) = s.split_once('e') {
            let m = strip_trailing(m);
            let ei: i32 = e.parse().unwrap_or(0);
            format!("{}e{}{:02}", m, if ei < 0 { '-' } else { '+' }, ei.abs())
        } else {
            s
        }
    }
}

fn strip_trailing(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        let t = t.trim_end_matches('.');
        t.to_string()
    } else {
        s.to_string()
    }
}

/// Right-justify `s` into a field of `width` using spaces (C "%Ng").
fn pad_left(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        format!("{}{}", " ".repeat(width - s.len()), s)
    }
}

/// Compose the full spectrogram image (bands + axes/legend, or just the bands in
/// `raw` mode) into an **indexed** buffer in PNG row order (top = highest
/// frequency) plus its RGB `palette`. Both the PNG and RGBA renderers build on
/// this, so they always produce pixel-identical output. Returns
/// `(cols, rows, indices, palette)`.
fn compose_indexed(spec: &Spectrogram) -> Result<(u32, u32, Vec<u8>, Vec<u8>), String> {
    let cfg = &spec.cfg;
    let chans = spec.chans as i32;
    let p_rows = spec.rows;
    let p_cols = spec.cols;
    let c_rows = p_rows * chans + chans - 1;
    let rows = if cfg.raw {
        c_rows
    } else {
        BELOW + c_rows + 30 + if cfg.title.is_some() { 20 } else { 0 }
    };
    let cols = if cfg.raw {
        p_cols
    } else {
        LEFT + p_cols + BETWEEN + SPECTRUM_WIDTH + RIGHT
    };

    let font = inflate_font()?;
    let palette = make_palette(cfg);

    let mut img = Image {
        cols,
        rows,
        pixels: vec![BACKGROUND; (cols * rows) as usize],
        font,
    };

    let tick_len = 3 - cfg.no_axes as i32;

    // --- Spectrogram bands ---
    let autogain = if cfg.normalize { -spec.max } else { 0.0 };

    for k in 0..chans {
        let dbfs = &spec.dbfs[k as usize];
        let base = (!cfg.raw as i32) * BELOW + (chans - 1 - k) * (p_rows + 1);

        for j in 0..p_rows {
            for i in 0..p_cols {
                let v = dbfs[(i * p_rows + j) as usize] as f64 + autogain;
                let col = colour(cfg, v);
                img.set((!cfg.raw as i32) * LEFT + i, base + j, col);
            }
            if !cfg.raw && !cfg.no_axes {
                img.set(LEFT - 1, base + j, GRID);
                img.set(LEFT + p_cols, base + j, GRID);
            }
        }

        if !cfg.raw && !cfg.no_axes {
            for i in -1..=p_cols {
                img.set(LEFT + i, base - 1, GRID);
                img.set(LEFT + i, base + p_rows, GRID);
            }
        }
    }

    if !cfg.raw {
        draw_legends(&mut img, spec, cfg, c_rows, tick_len, autogain);
    }

    // Flip rows so PNG top = highest y (top-down order).
    let mut top_down = vec![0u8; (cols * rows) as usize];
    for y in 0..rows {
        let src = (y * cols) as usize;
        let dst = ((rows - 1 - y) * cols) as usize;
        top_down[dst..dst + cols as usize].copy_from_slice(&img.pixels[src..src + cols as usize]);
    }

    Ok((cols as u32, rows as u32, top_down, palette))
}

pub fn render_png<W: Write>(spec: &Spectrogram, mut out: W) -> Result<(), String> {
    let (cols, rows, top_down, palette) = compose_indexed(spec)?;

    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, cols, rows);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_palette(palette);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("PNG header error: {e}"))?;
        writer
            .write_image_data(&top_down)
            .map_err(|e| format!("PNG write error: {e}"))?;
    }
    out.write_all(&buf).map_err(|e| format!("write error: {e}"))?;
    Ok(())
}

/// Render the spectrogram to an [`image::RgbaImage`] — the same pixels as
/// [`render_png`], expanded from the indexed palette into straight RGBA8 (opaque
/// alpha). Handy when a caller wants the decoded image to resize, composite or
/// hand to another image pipeline rather than encoded PNG bytes.
pub fn render_rgba(spec: &Spectrogram) -> Result<image::RgbaImage, String> {
    let (cols, rows, indices, palette) = compose_indexed(spec)?;

    let mut rgba = vec![0u8; indices.len() * 4];
    for (px, &idx) in rgba.chunks_exact_mut(4).zip(indices.iter()) {
        let p = idx as usize * 3;
        px[0] = palette[p];
        px[1] = palette[p + 1];
        px[2] = palette[p + 2];
        px[3] = 255;
    }
    image::RgbaImage::from_raw(cols, rows, rgba)
        .ok_or_else(|| "internal error: RGBA buffer size mismatch".to_string())
}

/// Build one channel's heatmap (cols × rows, high frequency at top) as a top-down
/// BGRA8 buffer. Each channel is a separate image so the GUI can lay them out
/// with a real background gap between them rather than baking it into one image.
fn channel_bgra(spec: &Spectrogram, k: usize, palette: &[u8]) -> (usize, usize, Vec<u8>) {
    let cfg = &spec.cfg;
    let autogain = if cfg.normalize { -spec.max } else { 0.0 };
    let tex_rows = gui_tex_rows(spec.rows, spec.chans);
    dbfs_to_bgra(cfg, &spec.dbfs[k], spec.rows, spec.cols, spec.cols, tex_rows, autogain, palette)
}

/// Per-channel texture height for the GUI. We keep full frequency resolution in
/// the dBfs data and the PNG/CLI output, but the GPU textures are capped so all
/// channels' textures fit together in floem/vger's shared atlas, which only
/// holds a handful of large regions (a 2400-wide region takes a full atlas row).
/// Without this, a multi-channel file with a tall FFT overflows the atlas: some
/// channels never paint and the atlas thrashes (re-uploading every frame, which
/// is what makes resizing stutter). The bands are small on screen, so capping
/// the texture height is visually imperceptible.
fn gui_tex_rows(rows: i32, chans: usize) -> i32 {
    // Vertical texture budget shared across channels. Keeps the atlas square
    // side (2 * max dimension) within GPU limits and leaves room for the legend.
    const TOTAL_BUDGET: i32 = 3600;
    let cap = (TOTAL_BUDGET / chans.max(1) as i32).max(1);
    rows.min(cap)
}

/// Colour one channel's dBfs columns into a top-down **BGRA8** buffer of width
/// `total_cols × rows` (high frequency at top), ready for direct GPU upload
/// (gpui's `RenderImage` stores BGRA, so we emit that byte order here rather than
/// swapping in a second pass on the UI thread). `palette` is the precomputed RGB
/// palette ([`make_palette`]); the caller passes it in so it isn't rebuilt per
/// channel/frame. Columns `0..cols_filled` are taken from `dbfs` (column-major:
/// `dbfs[col*rows + row]`); any remaining columns up to `total_cols` are left as
/// the background colour, which is how the GUI grows the image left-to-right
/// while a file streams in.
pub fn dbfs_to_bgra(
    cfg: &Config,
    dbfs: &[f32],
    rows: i32,
    cols_filled: i32,
    total_cols: i32,
    tex_rows: i32,
    autogain: f64,
    palette: &[u8],
) -> (usize, usize, Vec<u8>) {
    let w = total_cols.max(1) as usize;
    // Output texture height; never upsample past the real row count. When
    // `tex_rows == rows` this is an exact 1:1 copy (the mapping below reduces to
    // `j = rows - 1 - r`); when smaller, frequency bins are nearest-sampled.
    let h = tex_rows.clamp(1, rows.max(1)) as usize;
    // Background colour in BGRA byte order.
    let bg = [palette[2], palette[1], palette[0]];

    let mut buf = vec![0u8; w * h * 4];
    // Background fill first (covers the not-yet-computed columns).
    for px in buf.chunks_exact_mut(4) {
        px[0] = bg[0];
        px[1] = bg[1];
        px[2] = bg[2];
        px[3] = 0xff;
    }
    let filled = cols_filled.min(total_cols);
    let rows_span = (rows - 1).max(0) as f64;
    let denom = (h as i32 - 1).max(1) as f64;
    for r in 0..h {
        // Output row r (0 = top = Nyquist) → source frequency bin j (0 = DC).
        let j = if h == 1 {
            rows - 1
        } else {
            ((1.0 - r as f64 / denom) * rows_span).round() as i32
        }
        .clamp(0, rows - 1);
        let ro = r * w * 4;
        for i in 0..filled {
            let v = dbfs[(i * rows + j) as usize] as f64 + autogain;
            let idx = colour(cfg, v) as usize * 3;
            let o = ro + i as usize * 4;
            buf[o] = palette[idx + 2]; // B
            buf[o + 1] = palette[idx + 1]; // G
            buf[o + 2] = palette[idx]; // R
        }
    }
    (w, h, buf)
}

fn encode_rgba_png(w: usize, h: usize, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, w as u32, h as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("PNG header error: {e}"))?;
        writer
            .write_image_data(rgba)
            .map_err(|e| format!("PNG write error: {e}"))?;
    }
    Ok(buf)
}

/// Raw BGRA8 per channel (top channel first) as `(width, height, pixels)`. The
/// GUI uploads these directly to GPU textures (no PNG round-trip). The palette is
/// built once here and shared across channels.
pub fn channel_images(spec: &Spectrogram) -> Vec<(usize, usize, Vec<u8>)> {
    let palette = make_palette(&spec.cfg);
    (0..spec.chans).map(|k| channel_bgra(spec, k, &palette)).collect()
}

/// Per-channel partial BGRA snapshot from a streaming processor: each image is
/// the full target width (`x_size`), with only the columns produced so far
/// painted and the rest left as background. Normalisation is intentionally not
/// applied here (the global max isn't known mid-stream); the GUI uses the
/// default, non-normalised config. The palette is built once and shared across
/// channels (it's constant for a given config).
pub fn stream_channel_images(proc: &StreamProcessor) -> Vec<(usize, usize, Vec<u8>)> {
    let cfg = proc.cfg();
    let rows = proc.rows();
    let total = proc.x_size();
    let filled = proc.cols_done();
    let tex_rows = gui_tex_rows(rows, proc.chans());
    let palette = make_palette(cfg);
    (0..proc.chans())
        .map(|k| dbfs_to_bgra(cfg, proc.channel_dbfs(k), rows, filled, total, tex_rows, 0.0, &palette))
        .collect()
}

/// Vertical dBFS colour-scale legend (0 dBFS at top → `-db_range` at bottom)
/// matching the GUI's default palette, as PNG bytes. Stretched in the UI.
pub fn legend_png(db_range: i32) -> Result<Vec<u8>, String> {
    let mut cfg = Config::default();
    cfg.db_range = db_range;
    let cfg = cfg.finalize()?;
    let palette = make_palette(&cfg);
    let w = 24usize;
    let h = 256usize;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        let frac = y as f64 / (h - 1) as f64; // 0 at top .. 1 at bottom
        let dbfs = -(cfg.db_range as f64) * frac; // 0 dBFS .. -range
        let idx = colour(&cfg, dbfs) as usize * 3;
        for x in 0..w {
            let o = (y * w + x) * 4;
            rgba[o] = palette[idx];
            rgba[o + 1] = palette[idx + 1];
            rgba[o + 2] = palette[idx + 2];
            rgba[o + 3] = 0xff;
        }
    }
    encode_rgba_png(w, h, &rgba)
}

fn draw_legends(
    img: &mut Image,
    spec: &Spectrogram,
    cfg: &Config,
    c_rows: i32,
    tick_len: i32,
    autogain: f64,
) {
    let chans = spec.chans as i32;
    let p_rows = spec.rows;
    let p_cols = spec.cols;
    let cols = img.cols;

    // Title + footer comment
    if let Some(title) = &cfg.title {
        let i = title.len() as i32 * FONT_BIG_X;
        if i < cols + 1 {
            img.print_at((cols - i) / 2, img.rows - FONT_Y, TEXT, title);
        }
    }
    if cfg.comment.len() as i32 * FONT_BIG_X < cols + 1 {
        let comment = cfg.comment.clone();
        img.print_at(1, FONT_Y, TEXT, &comment);
    }

    let secs = |cols_: i32| -> f64 {
        cols_ as f64 * spec.step_size as f64 * spec.block_steps as f64 / spec.rate
    };

    // --- X-axis ---
    let (step, limit, prefix) = axis(secs(p_cols), p_cols / (FONT_BIG_X * 9 / 2));
    let label = format!("Time ({}s)", first_char(&prefix));
    img.print_at(
        LEFT + (p_cols - FONT_BIG_X * label.len() as i32) / 2,
        24,
        TEXT,
        &label,
    );

    let mut i = 0i32;
    while i as f64 <= limit {
        let x = if limit != 0.0 {
            (i as f64 / limit * p_cols as f64 + 0.5) as i32
        } else {
            0
        };
        for y in 0..tick_len {
            img.set(LEFT - 1 + x, BELOW - 1 - y, GRID);
            img.set(LEFT - 1 + x, BELOW + c_rows + y, GRID);
        }
        if !(step == 5 && i % 10 != 0) {
            let text = fmt_g(0.1 * i as f64, 6);
            let xx = LEFT + x - 3 * text.len() as i32;
            img.print_at(xx, BELOW - 6, LABELS, &text);
            img.print_at(xx, BELOW + c_rows + 14, LABELS, &text);
        }
        i += step.max(1);
    }

    // --- Y-axis ---
    let (step, limit, prefix) = axis(spec.rate / 2.0, (p_rows - 1) / ((FONT_Y * 3 + 1) >> 1));
    let label = format!("Frequency ({}Hz)", first_char(&prefix));
    img.print_up(10, BELOW + (c_rows - FONT_BIG_X * label.len() as i32) / 2, TEXT, &label);

    for k in 0..chans {
        let base = BELOW + k * (p_rows + 1);
        let mut i = 0i32;
        while i as f64 <= limit {
            let y = if limit != 0.0 {
                (i as f64 / limit * (p_rows - 1) as f64 + 0.5) as i32
            } else {
                0
            };
            for x in 0..tick_len {
                img.set(LEFT - 1 - x, base + y, GRID);
                img.set(LEFT + p_cols + x, base + y, GRID);
            }
            if !((step == 5 && i % 10 != 0) || (i == 0 && k != 0 && chans > 1)) {
                let left_text = if i != 0 {
                    pad_left(&fmt_g(0.1 * i as f64, 6), 5)
                } else {
                    "   DC".to_string()
                };
                img.print_at(LEFT - 4 - FONT_BIG_X * 5, base + y + 5, LABELS, &left_text);
                let right_text = if i != 0 {
                    fmt_g(0.1 * i as f64, 6)
                } else {
                    "DC".to_string()
                };
                img.print_at(LEFT + p_cols + 6, base + y + 5, LABELS, &right_text);
            }
            i += step.max(1);
        }
    }

    // --- Z-axis (spectrum legend) ---
    let k = 400.min(c_rows);
    let base = BELOW + (c_rows - k) / 2;
    img.print_at(cols - RIGHT - 2 - FONT_BIG_X, base - 13, TEXT, "dBFS");
    for j in 0..k {
        let b = colour(cfg, cfg.db_range as f64 * (j as f64 / (k - 1) as f64 - 1.0));
        for i in 0..SPECTRUM_WIDTH {
            img.set(cols - RIGHT - 1 - i, base + j, b);
        }
    }
    let step = 10 * (cfg.db_range as f64 / 10.0 * (FONT_Y + 2) as f64 / (k - 1) as f64).ceil() as i32;
    let mut i = 0i32;
    while i <= cfg.db_range {
        let y = (i as f64 / cfg.db_range as f64 * (k - 1) as f64 + 0.5) as i32;
        let val = i - cfg.gain - cfg.db_range - (autogain + 0.5) as i32;
        let text = format!("{:+}", val);
        img.print_at(cols - RIGHT + 1, base + y + 5, LABELS, &text);
        i += step.max(1);
    }
}

fn first_char(s: &str) -> &str {
    // C uses "%.1s" — at most the first character.
    if s.is_empty() {
        s
    } else {
        &s[..1]
    }
}

fn inflate_font() -> Result<Vec<u8>, String> {
    let mut decoder = flate2::read::ZlibDecoder::new(&FIXED_FONT_ZLIB[..]);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| format!("font inflate error: {e}"))?;
    Ok(out)
}
