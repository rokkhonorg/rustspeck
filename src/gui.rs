//! A Spek-like GUI built on gpui: drag-and-drop an audio file and see its
//! spectrogram with frequency/time axes, a dBFS colour legend and the filename.
//!
//! The spectrogram heatmap is computed once per file on a background thread and
//! GPU-scaled to fill the centre area, so resizing is cheap. Axes/labels are
//! real gpui text elements (JetBrains Mono) drawn in gutters around the heatmap,
//! so they stay crisp regardless of how the heatmap is scaled. Tick positions
//! are computed from the live size of the heatmap area on each render.
//!
//! This is a straight port of the floem viewer (`gpui-port` branch): identical
//! content, layout and styling, just expressed with gpui's element/entity model
//! instead of floem's reactive signals. The heatmap is drawn with a `canvas`
//! element that blits one `RenderImage` per channel into its band via
//! `Window::paint_image`, mirroring the floem `draw_img` path.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use lofty::file::{FileType, TaggedFileExt};
use lofty::prelude::{Accessor, AudioFile};

use gpui::prelude::*;
use gpui::{
    canvas, div, img, point, px, rgb, size, App, Application, AsyncApp, Bounds, Context, Corners,
    Div, ExternalPaths, FocusHandle, KeyDownEvent, ObjectFit, Pixels, Render, RenderImage,
    SharedString, TitlebarOptions, Window, WindowBounds, WindowOptions,
};
use image::{Frame, RgbaImage};
use smallvec::SmallVec;
use strum::IntoEnumIterator;

use crate::audio::{self, Audio};
use crate::render;
use crate::spectrogram::{self, Config, StreamProcessor};
use crate::window::WindowType;

const FONT: &str = "JetBrains Mono";
const PLACEHOLDER: &str = "Drag and drop an audio file.";

/// The monospace font for axis/legend labels, embedded so the viewer renders
/// identically regardless of which fonts are installed on the machine.
const FONT_BYTES: &[u8] = include_bytes!("../assets/JetBrainsMono.ttf");

/// The proportional UI font for all other text (title rows, status/placeholder),
/// also embedded. Set on the root element so it cascades to every child; the
/// monospace labels override it explicitly.
const UI_FONT: &str = "Geist";
const UI_FONT_BYTES: &[u8] = include_bytes!("../assets/Geist.ttf");

// Fixed render resolution; gpui GPU-scales it to fill the centre area.
const RENDER_COLS: i32 = 2400;
const RENDER_HEIGHT: i32 = 1320;

// Gutter sizes (logical px).
const FREQ_W: f64 = 72.0;
const TIME_H: f64 = 34.0;
const LEGEND_W: f64 = 84.0;
const LEGEND_GAP_PX: f64 = 12.0; // gap between the spectrogram and the legend bar
const TITLE_H: f64 = 74.0; // two rows: "Artist — Title" + technical info
const CHANNEL_GAP_PX: f64 = 8.0; // layout gap between stacked channel images

// Colours.
const BG: u32 = 0x12_12_16;
const GUTTER_BG: u32 = 0x1a_1a_20;
const LABEL: u32 = 0xb8_b8_c4;
const TITLE: u32 = 0xe8_e8_f0;

const WIN_W: f64 = 1100.0;
const WIN_H: f64 = 680.0;

/// `f64` logical px → gpui `Pixels`.
#[inline]
fn p(x: f64) -> Pixels {
    px(x as f32)
}

/// The analysis parameters the user can cycle live from the keyboard. `Copy` so
/// the worker thread can take it by value.
#[derive(Clone, Copy)]
struct Analysis {
    /// FFT size override (DFT points). `None` = derive height automatically.
    fft_size: Option<i32>,
    win_type: WindowType,
}

/// FFT sizes the `f` / `Shift+F` keys cycle through (powers of two, 128 … 16384).
const FFT_SIZES: [i32; 8] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

/// Smallest cycle size strictly larger than `cur` (the `f` step). Because it's
/// relative to the current size, the first press steps up from whatever the
/// automatic geometry chose rather than jumping to the bottom. Wraps to the
/// smallest size once past the largest.
fn fft_up(cur: i32) -> i32 {
    FFT_SIZES.iter().copied().find(|&s| s > cur).unwrap_or(FFT_SIZES[0])
}

/// Largest cycle size strictly smaller than `cur` (the `Shift+F` step). Wraps to
/// the largest size once below the smallest.
fn fft_down(cur: i32) -> i32 {
    FFT_SIZES
        .iter()
        .copied()
        .rev()
        .find(|&s| s < cur)
        .unwrap_or(FFT_SIZES[FFT_SIZES.len() - 1])
}

/// Next window function in the cycle (`strum::EnumIter`, wrapping around).
fn next_window(current: WindowType) -> WindowType {
    WindowType::iter()
        .cycle()
        .skip_while(|&w| w != current)
        .nth(1)
        .expect("EnumIter is non-empty")
}

/// Metadata needed to draw the axes (everything is Copy, so it's Send for the
/// worker→UI hand-off).
#[derive(Clone, Copy, Default)]
struct SpecMeta {
    rate: f64,
    channels: i32,
    cols: i32,
    step_size: i32,
    block_steps: i32,
    db_range: i32,
    dft_size: i32,
}

impl SpecMeta {
    fn from(spec: &spectrogram::Spectrogram) -> Self {
        SpecMeta {
            rate: spec.rate,
            channels: spec.chans as i32,
            cols: spec.cols,
            step_size: spec.step_size,
            block_steps: spec.block_steps,
            db_range: spec.cfg.db_range,
            dft_size: (spec.rows - 1) * 2, // rows = dft/2 + 1
        }
    }
    fn time_span(&self) -> f64 {
        self.cols as f64 * self.step_size as f64 * self.block_steps as f64 / self.rate
    }
    fn nyquist(&self) -> f64 {
        self.rate / 2.0
    }
}

/// Two-row header text shown above the spectrogram.
#[derive(Clone, Default)]
struct TrackInfo {
    line1: String, // "Artist — Title" (or a filename fallback)
    line2: String, // "FLAC · 2 ch · 44.1 kHz · 16-bit · 1024-pt · Hann"
}

/// The active load generation. Each drag-and-drop (or initial load) bumps it;
/// the in-flight worker for an older generation aborts at its next checkpoint and
/// the UI drops any of its already-queued messages, so a newly dropped file
/// cleanly supersedes one that's still rendering.
static LOAD_GEN: AtomicU64 = AtomicU64::new(0);

fn is_current(my_gen: u64) -> bool {
    LOAD_GEN.load(Ordering::SeqCst) == my_gen
}

/// Build one `Arc<RenderImage>` per channel from raw BGRA buffers — the byte
/// order gpui's `RenderImage` stores. The renderer (`render::*_images`) already
/// emits BGRA, so there's no per-frame channel swap here. A fresh `RenderImage`
/// (hence a fresh GPU texture id) is created per call, which is what lets a
/// streaming load replace the previous frame's texture.
fn make_images(buffers: Vec<(usize, usize, Vec<u8>)>) -> Vec<Arc<RenderImage>> {
    buffers
        .into_iter()
        .map(|(w, h, bgra)| {
            let buf = RgbaImage::from_raw(w as u32, h as u32, bgra)
                .expect("bgra buffer length must equal w*h*4");
            let frame = Frame::new(buf);
            let frames: SmallVec<[Frame; 1]> = SmallVec::from_buf([frame]);
            Arc::new(RenderImage::new(frames))
        })
        .collect()
}

/// Messages sent from the decode/analyse worker thread to the UI.
enum LoadMsg {
    /// A load just began — show "Loading …" and clear the old image.
    Start(String),
    /// Geometry/metadata known: draw the axes (before any pixels exist).
    Meta(SpecMeta, TrackInfo),
    /// A progressive snapshot (full target width, partially filled).
    Frame(Vec<Arc<RenderImage>>),
    /// The final, complete image.
    Done(Vec<Arc<RenderImage>>),
    /// Decode/analyse failed.
    Err(String),
}

/// Channel the worker thread pushes [`LoadMsg`]s into; a `cx.spawn` task on the
/// UI thread drains it and applies each one in order.
type Sender = async_channel::Sender<(u64, LoadMsg)>;

/// The root view: holds all the state the previous floem build kept in signals.
struct SpekApp {
    tx: Sender,
    analysis: Analysis,
    path: Option<PathBuf>, // the current track, kept so f/w can re-analyse it
    focus_handle: FocusHandle, // so the root element receives key events
    images: Vec<Arc<RenderImage>>, // one image per channel
    status: SharedString,
    info: TrackInfo,
    meta: Option<SpecMeta>,
    legend: Arc<RenderImage>, // dBFS colour-scale gradient bar
}

impl SpekApp {
    fn new(
        tx: Sender,
        legend: Arc<RenderImage>,
        fft_size: Option<i32>,
        focus_handle: FocusHandle,
    ) -> Self {
        SpekApp {
            tx,
            analysis: Analysis {
                fft_size,
                win_type: WindowType::Hann,
            },
            path: None,
            focus_handle,
            images: Vec::new(),
            status: SharedString::from(PLACEHOLDER),
            info: TrackInfo::default(),
            meta: None,
            legend,
        }
    }

    /// Spawn the background worker that decodes `path` and streams the
    /// spectrogram to the UI through the channel. Claims a new generation, which
    /// supersedes any in-flight load.
    fn load(&mut self, path: PathBuf, _cx: &mut Context<Self>) {
        self.path = Some(path.clone());
        let my_gen = LOAD_GEN.fetch_add(1, Ordering::SeqCst) + 1;
        let tx = self.tx.clone();
        let analysis = self.analysis;
        std::thread::spawn(move || {
            let emit = |msg: LoadMsg| {
                let _ = tx.send_blocking((my_gen, msg));
            };
            if let Err(e) = worker(&path, analysis, my_gen, &emit) {
                emit(LoadMsg::Err(format!(
                    "Failed to load {}: {e}",
                    file_name(&path)
                )));
            }
        });
    }

    /// Re-run the analysis for the current track (after `f`/`w` changed a
    /// parameter). No-op when no track is loaded yet.
    fn reanalyse(&mut self, cx: &mut Context<Self>) {
        if let Some(path) = self.path.clone() {
            self.load(path, cx);
        }
    }

    /// The FFT size currently in effect: the explicit override if set, else the
    /// size the automatic height-derived geometry produced for the loaded track.
    fn current_fft(&self) -> i32 {
        self.analysis
            .fft_size
            .or_else(|| self.meta.map(|m| m.dft_size))
            .unwrap_or(FFT_SIZES[3])
    }

    /// `f` / `Shift+F`: step the FFT size up (or `down`) from the current size.
    /// Only acts when a track is present.
    fn cycle_fft(&mut self, down: bool, cx: &mut Context<Self>) {
        if self.path.is_none() {
            return;
        }
        let cur = self.current_fft();
        self.analysis.fft_size = Some(if down { fft_down(cur) } else { fft_up(cur) });
        self.reanalyse(cx);
    }

    /// `w`: cycle the window function. Only acts when a track is present.
    fn cycle_window(&mut self, cx: &mut Context<Self>) {
        if self.path.is_none() {
            return;
        }
        self.analysis.win_type = next_window(self.analysis.win_type);
        self.reanalyse(cx);
    }

    /// Apply one worker message on the UI thread. Drops messages from a
    /// superseded load (a newer file was dropped while this one was still
    /// rendering), so only the active load paints.
    fn handle(&mut self, load_gen: u64, msg: LoadMsg, cx: &mut Context<Self>) {
        if load_gen != LOAD_GEN.load(Ordering::SeqCst) {
            return;
        }
        match msg {
            LoadMsg::Start(name) => {
                self.status = SharedString::from(format!("Loading {name}…"));
                self.images.clear();
            }
            LoadMsg::Meta(m, ti) => {
                self.meta = Some(m);
                self.info = ti;
            }
            LoadMsg::Frame(imgs) | LoadMsg::Done(imgs) => {
                self.images = imgs;
            }
            LoadMsg::Err(e) => {
                self.status = SharedString::from(e);
                self.images.clear();
            }
        }
        cx.notify();
    }
}

impl Render for SpekApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Centre (heatmap) area size in logical px — the live window size minus
        // the fixed gutters. Tick positions are derived from this each render.
        let vp = window.viewport_size();
        let (area_w, area_h) = centre_area(f32::from(vp.width) as f64, f32::from(vp.height) as f64);

        // --- Title row ---
        let title = if self.info.line1.is_empty() {
            div()
                .h(p(TITLE_H))
                .w_full()
                .flex_shrink_0()
                .bg(rgb(GUTTER_BG))
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(TITLE))
                .text_size(px(14.0))
                .child("RustSpeck")
        } else {
            // Metadata rows inherit the proportional UI font (Geist) from the
            // root; only the axis/legend labels are monospace. Left-aligned,
            // indented so the text starts at the spectrogram's left edge.
            div()
                .h(p(TITLE_H))
                .w_full()
                .flex_shrink_0()
                .bg(rgb(GUTTER_BG))
                .flex()
                .flex_col()
                .items_start()
                // Top-anchored (not centred) so the block sits a little lower
                // from the top edge, with the two rows close together.
                .justify_start()
                .pl(p(FREQ_W))
                .pt(px(20.0))
                .gap(px(3.0))
                .child(
                    div()
                        .text_color(rgb(TITLE))
                        .text_size(px(16.0))
                        .child(SharedString::from(self.info.line1.clone())),
                )
                .child(
                    div()
                        .text_color(rgb(LABEL))
                        .text_size(px(12.0))
                        .child(SharedString::from(self.info.line2.clone())),
                )
        };

        // --- Frequency gutter (left) ---
        // Content is absolutely positioned (zero intrinsic size), so flex_shrink_0
        // reserves the gutter's fixed width; the heatmap (flex_1 + min 0) is the
        // only thing that gives when the window shrinks.
        let freq_axis = div()
            .w(p(FREQ_W))
            .h_full()
            .flex_shrink_0()
            .bg(rgb(GUTTER_BG))
            .relative()
            .children(freq_axis_labels(self.meta, area_h));

        // --- Heatmap (centre) ---
        let centre_base = div()
            .flex_1()
            .h_full()
            .min_w(px(0.0))
            .min_h(px(0.0));
        let centre = if self.images.is_empty() {
            centre_base
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(LABEL))
                .child(self.status.clone())
        } else {
            // Draw each channel's image into its band. The renderer caches each
            // texture by id and GPU-scales it on resize, so dragging is cheap.
            let imgs = self.images.clone();
            centre_base.child(
                canvas(
                    |_bounds, _window, _cx| {},
                    move |bounds, _state, window, _cx| {
                        let n = imgs.len();
                        if n == 0 {
                            return;
                        }
                        let gap = CHANNEL_GAP_PX as f32;
                        let total = f32::from(bounds.size.height) - (n as f32 - 1.0) * gap;
                        let band_h = (total / n as f32).max(1.0);
                        for (k, im) in imgs.iter().enumerate() {
                            let top = f32::from(bounds.origin.y) + k as f32 * (band_h + gap);
                            let rect = Bounds {
                                origin: point(bounds.origin.x, px(top)),
                                size: size(bounds.size.width, px(band_h)),
                            };
                            let _ = window.paint_image(
                                rect,
                                Corners::default(),
                                im.clone(),
                                0,
                                false,
                            );
                        }
                    },
                )
                .size_full(),
            )
        };

        // --- Legend gutter (right) ---
        let bar = img(self.legend.clone())
            .w(px(16.0))
            .h_full()
            .object_fit(ObjectFit::Fill);
        let scale = div()
            .flex_1()
            .h_full()
            .relative()
            .children(legend_labels(self.meta, area_h));
        let legend_gutter = div()
            .w(p(LEGEND_W))
            .h_full()
            .flex_shrink_0()
            .bg(rgb(GUTTER_BG))
            .flex()
            .flex_row()
            // Pad left so the colour bar doesn't butt up against the spectrogram.
            .pl(p(LEGEND_GAP_PX))
            .child(bar)
            .child(scale);

        let row = div()
            .flex()
            .flex_row()
            .w_full()
            .flex_1()
            .min_h(px(0.0))
            .child(freq_axis)
            .child(centre)
            .child(legend_gutter);

        // --- Time axis (bottom) ---
        let bottom = div()
            .h(p(TIME_H))
            .w_full()
            .flex_shrink_0()
            .bg(rgb(GUTTER_BG))
            .flex()
            .flex_row()
            .child(div().w(p(FREQ_W)).flex_shrink_0())
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.0))
                    .h_full()
                    .relative()
                    .children(time_axis_labels(self.meta, area_w)),
            )
            .child(div().w(p(LEGEND_W)).flex_shrink_0());

        // gpui bubbles drop events up, so a single root-level handler catches a
        // file dropped anywhere in the window (no transparent overlay needed).
        div()
            .id("root")
            .track_focus(&self.focus_handle)
            .size_full()
            .bg(rgb(BG))
            .font_family(UI_FONT)
            .flex()
            .flex_col()
            // f steps the FFT size up, Shift+F down; w cycles the window function
            // (only with a track loaded). Shift is allowed (it's f's reverse
            // modifier); ignore key repeats and the other modifier combos so
            // holding a key or pressing e.g. Ctrl+F doesn't fire.
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _window, cx| {
                let m = &ev.keystroke.modifiers;
                if ev.is_held || m.control || m.alt || m.platform || m.function {
                    return;
                }
                match ev.keystroke.key.as_str() {
                    "f" => this.cycle_fft(m.shift, cx),
                    "w" => this.cycle_window(cx),
                    _ => {}
                }
            }))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                if let Some(path) = paths.paths().first() {
                    this.load(path.clone(), cx);
                }
            }))
            .child(title)
            .child(row)
            .child(bottom)
    }
}

pub fn run(initial: Option<PathBuf>, fft_size: Option<i32>) -> Result<(), String> {
    Application::new().run(move |cx: &mut App| {
        // Register the bundled monospace font so `.font_family(FONT)` resolves to
        // it whether or not JetBrains Mono is installed system-wide.
        let _ = cx
            .text_system()
            .add_fonts(vec![Cow::Borrowed(FONT_BYTES), Cow::Borrowed(UI_FONT_BYTES)]);

        let (tx, rx) = async_channel::unbounded::<(u64, LoadMsg)>();
        let legend = make_legend();

        let bounds = Bounds::centered(None, size(px(WIN_W as f32), px(WIN_H as f32)), cx);
        let handle = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    titlebar: Some(TitlebarOptions {
                        title: Some(SharedString::from("RustSpeck")),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                |window, cx| {
                    let focus = cx.focus_handle();
                    let view =
                        cx.new(|_cx| SpekApp::new(tx.clone(), legend.clone(), fft_size, focus.clone()));
                    // Focus the root so it receives key events (f/w) from the start.
                    focus.focus(window);
                    view
                },
            )
            .expect("failed to open window");

        // UI sink: drain the worker channel on the UI thread and apply each
        // message in order, never coalescing (the channel is unbounded and every
        // message gets its own `handle`).
        cx.spawn(async move |cx: &mut AsyncApp| {
            while let Ok((load_gen, msg)) = rx.recv().await {
                let _ = handle.update(cx, |app, _window, cx| app.handle(load_gen, msg, cx));
            }
        })
        .detach();

        // Load any file given on the command line — streamed like a drag-and-drop.
        if let Some(path) = initial {
            let _ = handle.update(cx, |app, _window, cx| app.load(path, cx));
        }

        cx.activate(true);
    });
    Ok(())
}

/// Build the dBFS colour-scale gradient bar as a single GPU image. The shared
/// renderer emits it as an RGBA PNG; we decode it once at startup and swap to
/// BGRA (unlike the channel images, which the renderer already emits as BGRA).
fn make_legend() -> Arc<RenderImage> {
    let png = render::legend_png(120).unwrap_or_default();
    let (w, h, mut bgra) = match image::load_from_memory(&png) {
        Ok(decoded) => {
            let buf = decoded.to_rgba8();
            (buf.width() as usize, buf.height() as usize, buf.into_raw())
        }
        Err(_) => (1, 1, vec![0, 0, 0, 0xff]),
    };
    // legend_png is RGBA; RenderImage wants BGRA. One-time swap at startup.
    for px in bgra.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    make_images(vec![(w, h, bgra)])
        .pop()
        .expect("one legend image")
}

/// Centre (heatmap) area size = window size minus the fixed gutters.
fn centre_area(win_w: f64, win_h: f64) -> (f64, f64) {
    (
        (win_w - FREQ_W - LEGEND_W).max(1.0),
        (win_h - TITLE_H - TIME_H).max(1.0),
    )
}

/// Target progressive-render rate while a file streams in. gpui repaints on
/// demand (and the platform vsync-caps to the display), so this just bounds how
/// often the worker pushes a new snapshot to the UI.
const TARGET_FPS: u64 = 10;

/// Publish a progressive frame at most `TARGET_FPS`×/sec, and only
/// when new columns have actually appeared since the last one.
const PUBLISH_INTERVAL: Duration = Duration::from_micros(1_000_000 / TARGET_FPS);

/// Decode `path` and feed the spectrogram incrementally, emitting progressive
/// frames as columns are produced. When the container doesn't report a length,
/// falls back to a single non-progressive render.
fn worker(
    path: &Path,
    analysis: Analysis,
    my_gen: u64,
    sink: &dyn Fn(LoadMsg),
) -> Result<(), String> {
    sink(LoadMsg::Start(file_name(path)));

    let mut dec = audio::open(path)?;
    let rate = dec.rate();
    let channels = dec.channels();
    let cfg = gui_config(analysis)?;

    let Some(total_len) = dec.total_len() else {
        // Length unknown: decode fully, then render once (no progressive fill).
        let mut samples = Vec::new();
        while let Some(chunk) = dec.next_chunk()? {
            if !is_current(my_gen) {
                return Ok(()); // superseded by a newer load
            }
            samples.extend_from_slice(&chunk);
        }
        let audio = Audio {
            rate,
            channels,
            samples,
        };
        let (rgbas, meta) = build_image(&audio, analysis)?;
        sink(LoadMsg::Meta(
            meta,
            track_info(path, rate, channels, &meta, analysis.win_type),
        ));
        sink(LoadMsg::Done(make_images(rgbas)));
        return Ok(());
    };

    let mut proc = StreamProcessor::new(cfg, rate, channels, total_len)?;
    // Axes can be drawn immediately: the geometry (full time/frequency span) is
    // fixed up front, before any pixels exist.
    let meta = meta_from_proc(&proc);
    sink(LoadMsg::Meta(
        meta,
        track_info(path, rate, channels, &meta, analysis.win_type),
    ));

    // Feed roughly 100 ms of audio per push so the per-channel parallel FFT has
    // a worthwhile batch, while keeping the UI updating several times a second.
    let batch = ((rate as usize * channels as usize) / 10).max(channels as usize);
    let mut pending: Vec<i32> = Vec::with_capacity(batch * 2);
    let mut last_publish = Instant::now();
    let mut last_cols = 0;

    let mut publish = |proc: &StreamProcessor, force: bool| {
        let cols = proc.cols_done();
        if cols > 0 && (force || (cols != last_cols && last_publish.elapsed() >= PUBLISH_INTERVAL)) {
            sink(LoadMsg::Frame(make_images(render::stream_channel_images(proc))));
            last_publish = Instant::now();
            last_cols = cols;
        }
    };

    while let Some(chunk) = dec.next_chunk()? {
        if !is_current(my_gen) {
            return Ok(()); // superseded by a newer load
        }
        pending.extend_from_slice(&chunk);
        if pending.len() >= batch {
            proc.push(&pending);
            pending.clear();
            publish(&proc, false);
        }
    }
    if !is_current(my_gen) {
        return Ok(());
    }
    if !pending.is_empty() {
        proc.push(&pending);
    }
    proc.finish();

    // Final image, rendered at the full target width (x_size) — the SAME
    // dimensions as the streaming frames — so the renderer recycles those
    // textures in place.
    sink(LoadMsg::Done(make_images(render::stream_channel_images(&proc))));
    Ok(())
}

/// The Config the GUI renders with: a fixed high render resolution that gpui
/// GPU-scales to the window, optionally overriding the FFT size.
fn gui_config(analysis: Analysis) -> Result<Config, String> {
    let mut cfg = Config::default();
    cfg.x_size0 = RENDER_COLS;
    cfg.win_type = analysis.win_type;
    match analysis.fft_size {
        // y_size sets dft per channel directly: dft = 2*(y_size-1) = n.
        Some(n) => cfg.y_size = n / 2 + 1,
        None => cfg.big_y_size = RENDER_HEIGHT,
    }
    cfg.finalize()
}

/// Axis metadata from a stream processor, with the column count set to the full
/// target width so the time axis spans the whole file while it fills in.
fn meta_from_proc(p: &StreamProcessor) -> SpecMeta {
    SpecMeta {
        rate: p.rate(),
        channels: p.chans() as i32,
        cols: p.x_size(),
        step_size: p.step_size(),
        block_steps: p.block_steps(),
        db_range: p.cfg().db_range,
        dft_size: (p.rows() - 1) * 2,
    }
}

/// Batch render (used by the unknown-length fallback above).
fn build_image(
    a: &Audio,
    analysis: Analysis,
) -> Result<(Vec<(usize, usize, Vec<u8>)>, SpecMeta), String> {
    let cfg = gui_config(analysis)?;
    let spec = spectrogram::process(cfg, a.rate, a.channels, &a.samples)?;
    let meta = SpecMeta::from(&spec);
    let rgbas = render::channel_images(&spec);
    Ok((rgbas, meta))
}

/// Two display rows: "Artist — Title" and a technical summary. Tags/format come
/// from lofty; rate/channels come from the decoded audio; window size/function
/// from our analysis. Never fails — falls back to the filename and extension.
fn track_info(
    path: &Path,
    rate: f64,
    channels: u32,
    meta: &SpecMeta,
    win_type: WindowType,
) -> TrackInfo {
    let mut artist = None;
    let mut title = None;
    let mut bit_depth = None;
    let mut format = None;

    if let Ok(tagged) = lofty::read_from_path(path) {
        let props = tagged.properties();
        bit_depth = props.bit_depth();
        format = Some(file_type_name(tagged.file_type()));
        if let Some(tag) = tagged.primary_tag().or_else(|| tagged.first_tag()) {
            artist = tag.artist().map(|c| c.trim().to_string()).filter(|s| !s.is_empty());
            title = tag.title().map(|c| c.trim().to_string()).filter(|s| !s.is_empty());
        }
    }

    let fmt = format.unwrap_or_else(|| {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_uppercase())
            .unwrap_or_else(|| "Audio".into())
    });

    let line1 = match (artist, title) {
        (Some(a), Some(t)) => format!("{a} — {t}"),
        (None, Some(t)) => t,
        (Some(a), None) => format!("{a} — {}", file_stem(path)),
        (None, None) => file_name(path),
    };

    let sr_khz = rate / 1000.0;
    let sr = if (sr_khz.fract()).abs() < 1e-6 {
        format!("{} kHz", sr_khz as i64)
    } else {
        format!("{sr_khz:.1} kHz")
    };
    let depth = bit_depth
        .map(|b| format!("{b}-bit"))
        .unwrap_or_else(|| "—".into());
    let line2 = format!(
        "{fmt} · {channels} ch · {sr} · {depth} · {}-pt · {win_type}",
        meta.dft_size
    );

    TrackInfo { line1, line2 }
}

fn file_type_name(ft: FileType) -> String {
    match ft {
        FileType::Flac => "FLAC",
        FileType::Mpeg => "MP3",
        FileType::Wav => "WAV",
        FileType::Aiff => "AIFF",
        FileType::Vorbis => "OGG Vorbis",
        FileType::Opus => "Opus",
        FileType::Mp4 => "MP4/AAC",
        FileType::Aac => "AAC",
        FileType::Ape => "APE",
        FileType::WavPack => "WavPack",
        FileType::Mpc => "Musepack",
        FileType::Speex => "Speex",
        _ => "Audio",
    }
    .to_string()
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn file_stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// --- Axis label / grid construction ---

fn freq_axis_labels(meta: Option<SpecMeta>, area_h: f64) -> Vec<Div> {
    let mut views = Vec::new();
    if let Some(m) = meta {
        let nyq = m.nyquist();
        // Bands match the centre heatmap: equal flex bands separated by gaps.
        let n = m.channels.max(1) as f64;
        let band_h = ((area_h - (n - 1.0) * CHANNEL_GAP_PX) / n).max(1.0);
        let step = nice_step(nyq, freq_target(band_h));
        for k in 0..m.channels {
            let band_top = k as f64 * (band_h + CHANNEL_GAP_PX);
            // Frequencies to label: DC (0) at the bottom, then nice steps, and
            // always the Nyquist peak at the top. Drop the final step if it would
            // collide with the peak label.
            let mut freqs = Vec::new();
            let mut f = 0.0;
            while f < nyq - step * 0.5 {
                freqs.push(f);
                f += step;
            }
            freqs.push(nyq);
            for f in freqs {
                let y = band_top + (1.0 - f / nyq) * band_h;
                // Keep each label inside its own band so DC sits at the band's
                // bottom edge and the peak at its top, never bleeding into a gap.
                let hi = (band_top + band_h - 14.0).max(band_top);
                let top = (y - 7.0).clamp(band_top, hi);
                // Right-aligned, hugging the plot.
                views.push(tick_label(0.0, top, FREQ_W - 10.0, Align::End, fmt_freq(f)));
            }
        }
    }
    views
}

fn time_axis_labels(meta: Option<SpecMeta>, area_w: f64) -> Vec<Div> {
    let mut views = Vec::new();
    if let Some(m) = meta {
        let span = m.time_span();
        if span > 0.0 {
            let step = nice_step(span, time_target(area_w));
            let w = 48.0_f64;
            let mut t = 0.0;
            while t <= span + 1e-6 {
                let frac = (t / span).clamp(0.0, 1.0);
                let x = frac * area_w;
                // Anchor the tick value at x: the first tick is flush-left so 0
                // sits at the true left edge, the last flush-right, rest centred.
                let (left, align) = if frac <= 1e-6 {
                    (0.0, Align::Start)
                } else if frac >= 1.0 - 1e-6 {
                    ((area_w - w).max(0.0), Align::End)
                } else {
                    ((x - w / 2.0).clamp(0.0, (area_w - w).max(0.0)), Align::Center)
                };
                views.push(tick_label(left, 9.0, w, align, fmt_time(t)));
                t += step;
            }
        }
    }
    views
}

fn legend_labels(meta: Option<SpecMeta>, area_h: f64) -> Vec<Div> {
    let mut labels = Vec::new();
    if let Some(m) = meta {
        let range = m.db_range as f64;
        let mut d = 0.0;
        while d <= range + 1e-6 {
            let y = (d / range) * area_h; // 0 dB at top
            let top = (y - 7.0).clamp(0.0, (area_h - 14.0).max(0.0));
            let txt = if d <= 1e-6 {
                "0".to_string()
            } else {
                format!("-{}", d as i64)
            };
            labels.push(tick_label(6.0, top, LEGEND_W - 24.0, Align::Start, txt));
            d += 20.0;
        }
    }
    labels
}

/// Horizontal alignment of a tick label inside its (absolutely-positioned) box.
#[derive(Clone, Copy)]
enum Align {
    Start,
    Center,
    End,
}

/// One absolutely-positioned monospace tick label.
fn tick_label(left: f64, top: f64, w: f64, align: Align, text: String) -> Div {
    let d = div()
        .absolute()
        .left(p(left))
        .top(p(top))
        .w(p(w))
        .h(p(14.0))
        .flex()
        .items_center()
        .font_family(FONT)
        .text_size(px(11.0))
        .text_color(rgb(LABEL))
        .child(SharedString::from(text));
    match align {
        Align::Start => d.justify_start(),
        Align::Center => d.justify_center(),
        Align::End => d.justify_end(),
    }
}

// --- tick math ---

/// Roughly one frequency tick per ~50px of band height (so small windows aren't
/// crowded), clamped to a sane range.
fn freq_target(band_h: f64) -> usize {
    ((band_h / 50.0) as usize).clamp(2, 12)
}

/// Roughly one time tick per ~90px of width.
fn time_target(area_w: f64) -> usize {
    ((area_w / 90.0) as usize).clamp(2, 12)
}

/// A "nice" tick step (1/2/5 × 10^n) so that `max` spans roughly `target` ticks.
fn nice_step(max: f64, target: usize) -> f64 {
    if max <= 0.0 || target == 0 {
        return 1.0;
    }
    let rough = max / target as f64;
    let mag = 10f64.powf(rough.log10().floor());
    let norm = rough / mag;
    let nice = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * mag
}

fn fmt_freq(hz: f64) -> String {
    if hz >= 1000.0 {
        let k = hz / 1000.0;
        if (k - k.round()).abs() < 1e-6 {
            format!("{}k", k.round() as i64)
        } else {
            format!("{k:.1}k")
        }
    } else {
        format!("{}", hz.round() as i64)
    }
}

fn fmt_time(s: f64) -> String {
    if s < 60.0 {
        if (s - s.round()).abs() < 1e-6 {
            format!("{}s", s.round() as i64)
        } else {
            format!("{s:.1}s")
        }
    } else {
        let m = (s / 60.0).floor() as i64;
        let sec = (s - m as f64 * 60.0).round() as i64;
        format!("{m}:{sec:02}")
    }
}
