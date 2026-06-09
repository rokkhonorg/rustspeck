//! A Spek-like GUI built on floem: drag-and-drop an audio file and see its
//! spectrogram with frequency/time axes, a dBFS colour legend and the filename.
//!
//! The spectrogram heatmap is computed once per file on a background thread and
//! GPU-scaled to fill the centre area, so resizing is cheap. Axes/labels are
//! real floem text views (JetBrains Mono) drawn in gutters around the heatmap,
//! so they stay crisp regardless of how the heatmap is scaled. Tick positions
//! are computed reactively from the live size of the heatmap area.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lofty::file::{FileType, TaggedFileExt};
use lofty::prelude::{Accessor, AudioFile};

use floem::action::debounce_action;
use floem::event::listener;
use floem::ext_event::{register_ext_trigger, ExtSendTrigger};
use floem::kurbo::{Rect, Size};
use floem::peniko::{Blob, Color, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality};
use floem::reactive::{RwSignal, Scope, SignalGet, SignalUpdate};
use floem::unit::UnitExt;
use floem::views::{canvas, dyn_view, img, Decorators, Empty, Label, Stack};
use floem::window::WindowConfig;
use floem::{Application, IntoView, Renderer, View};
use floem_renderer::Img;

use crate::audio::{self, Audio};
use crate::render;
use crate::spectrogram::{self, Config, Spectrogram, StreamProcessor};

const FONT: &str = "JetBrains Mono";
const PLACEHOLDER: &str = "Drag and drop an audio file.";

// Fixed render resolution; floem GPU-scales it to fill the centre area.
const RENDER_COLS: i32 = 2400;
const RENDER_HEIGHT: i32 = 1320;

// Gutter sizes (logical px).
const FREQ_W: f64 = 72.0;
const TIME_H: f64 = 34.0;
const LEGEND_W: f64 = 84.0;
const LEGEND_GAP_PX: f64 = 12.0; // gap between the spectrogram and the legend bar
const TITLE_H: f64 = 66.0; // two rows: "Artist — Title" + technical info
const CHANNEL_GAP_PX: f64 = 8.0; // layout gap between stacked channel images

// Colours.
const BG: Color = Color::from_rgb8(0x12, 0x12, 0x16);
const GUTTER_BG: Color = Color::from_rgb8(0x1a, 0x1a, 0x20);
const LABEL: Color = Color::from_rgb8(0xb8, 0xb8, 0xc4);
const TITLE: Color = Color::from_rgb8(0xe8, 0xe8, 0xf0);

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
    fn from(spec: &Spectrogram) -> Self {
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

/// One channel's spectrogram as a GPU-uploadable image brush, with a stable
/// cache key so the renderer uploads the texture once and reuses it.
#[derive(Clone)]
struct ChannelTex {
    brush: ImageBrush,
    hash: Arc<[u8]>,
}

static NEXT_GEN: AtomicU64 = AtomicU64::new(0);

/// The active load generation. Each drag-and-drop (or initial load) bumps it;
/// the in-flight worker for an older generation aborts at its next checkpoint and
/// the UI sink drops any of its already-queued messages, so a newly dropped file
/// cleanly supersedes one that's still rendering.
static LOAD_GEN: AtomicU64 = AtomicU64::new(0);

fn is_current(my_gen: u64) -> bool {
    LOAD_GEN.load(Ordering::SeqCst) == my_gen
}

/// Build one `ImageBrush` per channel directly from raw RGBA (no PNG round-trip).
fn make_texs(rgbas: Vec<(usize, usize, Vec<u8>)>) -> Vec<ChannelTex> {
    let gen_id = NEXT_GEN.fetch_add(1, Ordering::Relaxed);
    rgbas
        .into_iter()
        .enumerate()
        .map(|(k, (w, h, rgba))| {
            let blob = Blob::new(Arc::new(rgba));
            let brush = ImageBrush::new(ImageData {
                data: blob,
                format: ImageFormat::Rgba8,
                alpha_type: ImageAlphaType::Alpha,
                width: w as u32,
                height: h as u32,
            })
            .with_quality(ImageQuality::High);
            // Stable, unique key: (load generation, channel index).
            let mut key = gen_id.to_le_bytes().to_vec();
            key.push(k as u8);
            ChannelTex {
                brush,
                hash: Arc::from(key),
            }
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
    Frame(Vec<(usize, usize, Vec<u8>)>),
    /// The final, complete image.
    Done(Vec<(usize, usize, Vec<u8>)>),
    /// Decode/analyse failed.
    Err(String),
}

/// A thread-safe sink that marshals [`LoadMsg`]s onto the UI thread and runs
/// `handler` for each, in order, never dropping a message (it drains a queue).
///
/// floem's `create_ext_action` is one-shot and `update_signal_from_channel`
/// funnels through a single-slot signal (which can coalesce away intermediate
/// values); a streaming load needs every Meta/Done delivered, so we keep our own
/// queue and wake the UI with an `ExtSendTrigger`.
fn make_sink(handler: impl Fn(u64, LoadMsg) + 'static) -> Arc<dyn Fn(u64, LoadMsg) + Send + Sync> {
    let cx = Scope::new();
    let trigger = cx.enter(ExtSendTrigger::new);
    let queue: Arc<Mutex<VecDeque<(u64, LoadMsg)>>> = Arc::new(Mutex::new(VecDeque::new()));
    {
        let queue = queue.clone();
        cx.create_effect(move |_| {
            trigger.track();
            // Drain everything queued since the last wake.
            loop {
                let next = queue.lock().unwrap().pop_front();
                match next {
                    Some((load_gen, msg)) => handler(load_gen, msg),
                    None => break,
                }
            }
        });
    }
    Arc::new(move |load_gen: u64, msg: LoadMsg| {
        queue.lock().unwrap().push_back((load_gen, msg));
        register_ext_trigger(trigger);
    })
}

type Sink = Arc<dyn Fn(u64, LoadMsg) + Send + Sync>;

const WIN_W: f64 = 1100.0;
const WIN_H: f64 = 680.0;

pub fn run(initial: Option<PathBuf>, fft_size: Option<i32>) -> Result<(), String> {
    Application::new()
        .window(
            move |_| build_ui(initial, fft_size),
            Some(
                WindowConfig::default()
                    .size(Size::new(WIN_W, WIN_H))
                    .title("RustSpeck"),
            ),
        )
        .run();
    Ok(())
}

/// Build the whole view tree. Must run inside the window closure so its signals
/// and views are created within floem's runtime scope (otherwise nothing mounts).
fn build_ui(initial: Option<PathBuf>, fft_size: Option<i32>) -> impl IntoView {
    let image = RwSignal::new(Vec::<ChannelTex>::new()); // one image brush per channel
    let status = RwSignal::new(PLACEHOLDER.to_string());
    let info = RwSignal::new(TrackInfo::default());
    let meta = RwSignal::new(None::<SpecMeta>);
    // Centre (heatmap) area size in logical px — derived from the window size
    // minus the fixed gutters. Initialised from the configured window size so
    // axis labels are placed correctly before the first resize event.
    let area = RwSignal::new(centre_area(WIN_W, WIN_H));
    // Raw window size, written cheaply on every resize event; a debounced effect
    // (below) turns it into `area` once the drag settles, so we don't rebuild the
    // axis labels on every event.
    let raw_size = RwSignal::new((WIN_W as u32, WIN_H as u32));
    let legend = RwSignal::new(render::legend_png(120).unwrap_or_default());
    // Root view id, set after the tree is built; used to force a full repaint
    // when the axis views are rebuilt (on resize / worker update).
    let repaint = RwSignal::new(None::<floem::ViewId>);

    // UI sink: the worker thread sends LoadMsgs here; the handler runs on the UI
    // thread and fans each one out to the signals above (which drive the views).
    let sink = make_sink(move |load_gen, msg| {
        let cur = LOAD_GEN.load(Ordering::SeqCst);
        match &msg {
            LoadMsg::Start(n) => eprintln!("[ui] gen{load_gen}/{cur} Start {n}"),
            LoadMsg::Meta(m, _) => {
                eprintln!("[ui] gen{load_gen}/{cur} Meta ch={} cols={}", m.channels, m.cols)
            }
            LoadMsg::Frame(r) => eprintln!(
                "[ui] gen{load_gen}/{cur} Frame n={} dim0={:?}",
                r.len(),
                r.first().map(|(w, h, _)| (*w, *h))
            ),
            LoadMsg::Done(r) => eprintln!(
                "[ui] gen{load_gen}/{cur} Done n={} dim0={:?}",
                r.len(),
                r.first().map(|(w, h, _)| (*w, *h))
            ),
            LoadMsg::Err(e) => eprintln!("[ui] gen{load_gen}/{cur} Err {e}"),
        }
        // Drop messages from a superseded load (a newer file was dropped while
        // this one was still rendering), so only the active load paints.
        if load_gen != cur {
            eprintln!("[ui] DROPPED (stale)");
            return;
        }
        match msg {
            LoadMsg::Start(name) => {
                status.set(format!("Loading {name}…"));
                image.set(Vec::new());
            }
            LoadMsg::Meta(m, ti) => {
                meta.set(Some(m));
                info.set(ti);
            }
            LoadMsg::Frame(rgbas) | LoadMsg::Done(rgbas) => {
                image.set(make_texs(rgbas));
            }
            LoadMsg::Err(e) => {
                status.set(e);
                image.set(Vec::new());
            }
        }
        if let Some(id) = repaint.get_untracked() {
            id.request_all();
        }
    });

    // Load any file given on the command line — streamed like a drag-and-drop.
    if let Some(p) = initial {
        load_async(p, sink.clone(), fft_size);
    }

    // --- Heatmap (centre) — a canvas drawing each channel's image into its band.
    // Drawing the RGBA brushes directly skips the PNG encode/decode round-trip;
    // the renderer caches each texture by its hash and GPU-scales on resize.
    let centre = dyn_view(move || {
        let texs = image.get();
        if texs.is_empty() {
            Label::derived(move || status.get())
                .style(|s| s.size_full().items_center().justify_center().color(LABEL))
                .into_any()
        } else {
            let n = texs.len();
            canvas(move |cx, size| {
                eprintln!("[canvas] n={n} size={}x{}", size.width, size.height);
                let gap = CHANNEL_GAP_PX;
                let band_h = ((size.height - (n as f64 - 1.0) * gap) / n as f64).max(1.0);
                for (k, t) in texs.iter().enumerate() {
                    let top = k as f64 * (band_h + gap);
                    let rect = Rect::new(0.0, top, size.width, top + band_h);
                    cx.draw_img(
                        Img {
                            img: t.brush.clone(),
                            hash: &t.hash,
                        },
                        rect,
                    );
                }
            })
            .style(|s| s.size_full())
            .into_any()
        }
    })
    .style(|s| s.flex_grow(1.0).height_full().min_width(0.0).min_height(0.0));

    // --- Axis gutters ---
    // flex_shrink(0): the gutters' content is absolutely positioned (zero
    // intrinsic size), so without this flexbox would shrink them to nothing and
    // the heatmap would expand over them. Fixed-size gutters reserve their space;
    // the heatmap (flex_grow + min_width 0) is the only thing that gives.
    let freq_axis = dyn_view(move || freq_axis_labels(meta.get(), area.get().1))
        .style(|s| s.width(FREQ_W.pt()).height_full().flex_shrink(0.0).background(GUTTER_BG));

    let legend_gutter = dyn_view(move || legend_view(meta.get(), area.get().1, legend.get()))
        .style(|s| s.width(LEGEND_W.pt()).height_full().flex_shrink(0.0).background(GUTTER_BG));

    let row = Stack::horizontal((freq_axis, centre, legend_gutter))
        .style(|s| s.flex_grow(1.0).width_full().min_width(0.0).min_height(0.0));

    let time_axis = dyn_view(move || time_axis_labels(meta.get(), area.get().0))
        .style(|s| s.size_full());
    let bottom = Stack::horizontal((
        Empty::new().style(|s| s.width(FREQ_W.pt()).flex_shrink(0.0)),
        time_axis.style(|s| s.flex_grow(1.0).min_width(0.0).height_full()),
        Empty::new().style(|s| s.width(LEGEND_W.pt()).flex_shrink(0.0)),
    ))
    .style(|s| s.height(TIME_H.pt()).width_full().flex_shrink(0.0).background(GUTTER_BG));

    let title = dyn_view(move || {
        let ti = info.get();
        if ti.line1.is_empty() {
            Label::derived(|| "RustSpeck".to_string())
                .style(|s| {
                    s.size_full()
                        .items_center()
                        .justify_center()
                        .color(TITLE)
                        .font_family(FONT.to_string())
                        .font_size(14.0)
                })
                .into_any()
        } else {
            Stack::vertical((
                // Metadata rows use the default proportional font (only the
                // axis/legend labels are monospace).
                Label::new(ti.line1).style(|s| s.color(TITLE).font_size(16.0)),
                Label::new(ti.line2).style(|s| s.color(LABEL).font_size(12.0)),
            ))
            // Left-aligned, indented so the text starts at the spectrogram's
            // left edge (just past the frequency gutter).
            .style(|s| {
                s.size_full()
                    .items_start()
                    .justify_center()
                    .padding_left(FREQ_W.pt())
                    .row_gap(8.0.pt())
            })
            .into_any()
        }
    })
    .style(|s| {
        s.height(TITLE_H.pt())
            .width_full()
            .flex_shrink(0.0)
            .background(GUTTER_BG)
    });

    let content = Stack::vertical((title, row, bottom)).style(|s| s.size_full().background(BG));
    // Repaint target: the visible content subtree (worker updates / resize).
    repaint.set(Some(content.id()));

    // Recompute the centre area (which repositions every axis label) only once
    // resizing has paused for a beat. Rebuilding dozens of labels — and the
    // glyph re-shaping/atlas churn that comes with it — on every WindowResized
    // event is what makes dragging janky on multi-channel files. The heatmap
    // itself still scales live during the drag (its canvas is size_full).
    debounce_action(raw_size, Duration::from_millis(120), move || {
        let (w, h) = raw_size.get_untracked();
        let sz = centre_area(w as f64, h as f64);
        if area.get_untracked() != sz {
            area.set(sz);
            if let Some(id) = repaint.get_untracked() {
                id.request_all();
            }
        }
    });

    // floem dispatches file-drag events only to the *topmost view under the
    // cursor* (Phases::TARGET, no bubbling). So a transparent full-window overlay
    // on top catches drops anywhere, regardless of which gutter/image is beneath.
    let drop_catcher = Empty::new()
        .style(|s| s.absolute().inset(0.0).size_full())
        .on_event_stop(listener::FileDragDrop, move |_cx, ev| {
            if let Some(path) = ev.paths.first() {
                load_async(path.clone(), sink.clone(), fft_size);
            }
        });

    Stack::vertical((content, drop_catcher))
        .style(|s| s.size_full())
        .on_event_cont(listener::WindowResized, move |_cx, size| {
            // Cheap: just record the size. The expensive recalc (axis-label
            // rebuild + repaint) is debounced above so it runs once, after the
            // drag settles, instead of on every event.
            raw_size.set((size.width.round() as u32, size.height.round() as u32));
        })
}

/// Centre (heatmap) area size = window size minus the fixed gutters.
fn centre_area(win_w: f64, win_h: f64) -> (f64, f64) {
    (
        (win_w - FREQ_W - LEGEND_W).max(1.0),
        (win_h - TITLE_H - TIME_H).max(1.0),
    )
}

/// Spawn the background worker that decodes `path` and streams the spectrogram
/// to the UI through `sink`.
fn load_async(path: PathBuf, sink: Sink, fft_size: Option<i32>) {
    // Claim a new generation: this supersedes any in-flight load.
    let my_gen = LOAD_GEN.fetch_add(1, Ordering::SeqCst) + 1;
    std::thread::spawn(move || {
        let emit = move |msg: LoadMsg| sink(my_gen, msg);
        if let Err(e) = worker(&path, fft_size, my_gen, &emit) {
            emit(LoadMsg::Err(format!(
                "Failed to load {}: {e}",
                file_name(&path)
            )));
        }
    });
}

/// Publish a progressive frame at most ~16×/sec, and only when new columns have
/// actually appeared since the last one.
const PUBLISH_INTERVAL: Duration = Duration::from_millis(60);

/// Decode `path` and feed the spectrogram incrementally, emitting progressive
/// frames as columns are produced. When the container doesn't report a length,
/// falls back to a single non-progressive render.
fn worker(
    path: &Path,
    fft_size: Option<i32>,
    my_gen: u64,
    sink: &dyn Fn(LoadMsg),
) -> Result<(), String> {
    sink(LoadMsg::Start(file_name(path)));

    let mut dec = audio::open(path)?;
    let rate = dec.rate();
    let channels = dec.channels();
    let cfg = gui_config(fft_size)?;

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
        let (rgbas, meta) = build_image(&audio, fft_size)?;
        sink(LoadMsg::Meta(meta, track_info(path, rate, channels, &meta)));
        sink(LoadMsg::Done(rgbas));
        return Ok(());
    };

    let mut proc = StreamProcessor::new(cfg, rate, channels, total_len)?;
    // Axes can be drawn immediately: the geometry (full time/frequency span) is
    // fixed up front, before any pixels exist.
    let meta = meta_from_proc(&proc);
    sink(LoadMsg::Meta(meta, track_info(path, rate, channels, &meta)));

    // Feed roughly 100 ms of audio per push so the per-channel parallel FFT has
    // a worthwhile batch, while keeping the UI updating several times a second.
    let batch = ((rate as usize * channels as usize) / 10).max(channels as usize);
    let mut pending: Vec<i32> = Vec::with_capacity(batch * 2);
    let mut last_publish = Instant::now();
    let mut last_cols = 0;

    let mut publish = |proc: &StreamProcessor, force: bool| {
        let cols = proc.cols_done();
        if cols > 0 && (force || (cols != last_cols && last_publish.elapsed() >= PUBLISH_INTERVAL)) {
            sink(LoadMsg::Frame(render::stream_channel_images(proc)));
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

    // Final, exact image (matches the batch/CLI render pixel-for-pixel).
    let spec = proc.into_spectrogram();
    sink(LoadMsg::Done(render::channel_images(&spec)));
    Ok(())
}

/// The Config the GUI renders with: a fixed high render resolution that floem
/// GPU-scales to the window, optionally overriding the FFT size.
fn gui_config(fft_size: Option<i32>) -> Result<Config, String> {
    let mut cfg = Config::default();
    cfg.x_size0 = RENDER_COLS;
    match fft_size {
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
    fft_size: Option<i32>,
) -> Result<(Vec<(usize, usize, Vec<u8>)>, SpecMeta), String> {
    let cfg = gui_config(fft_size)?;
    let spec = spectrogram::process(cfg, a.rate, a.channels, &a.samples)?;
    let meta = SpecMeta::from(&spec);
    let rgbas = render::channel_images(&spec);
    Ok((rgbas, meta))
}

/// Two display rows: "Artist — Title" and a technical summary. Tags/format come
/// from lofty; rate/channels come from the decoded audio; window size/function
/// from our analysis. Never fails — falls back to the filename and extension.
fn track_info(path: &Path, rate: f64, channels: u32, meta: &SpecMeta) -> TrackInfo {
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
        "{fmt} · {channels} ch · {sr} · {depth} · {}-pt · Hann",
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

fn freq_axis_labels(meta: Option<SpecMeta>, area_h: f64) -> impl IntoView {
    let mut views = Vec::new();
    if let Some(m) = meta {
        let nyq = m.nyquist();
        // Bands match the centre v_stack: equal flex bands separated by gaps.
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
                let txt = fmt_freq(f);
                views.push(Label::new(txt).style(move |s| {
                    s.absolute()
                        .inset_top(top.pt())
                        .inset_left(0.0.pt())
                        .width((FREQ_W - 10.0).pt()) // right-aligned, hugging the plot
                        .height(14.0.pt())
                        .justify_end()
                        .font_family(FONT.to_string())
                        .font_size(11.0)
                        .color(LABEL)
                }));
            }
        }
    }
    Stack::from_iter(views).style(|s| s.size_full())
}

fn time_axis_labels(meta: Option<SpecMeta>, area_w: f64) -> impl IntoView {
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
                let txt = fmt_time(t);
                // Anchor the tick value at x: the first tick is flush-left so 0
                // sits at the true left edge, the last flush-right, rest centred.
                let (left, align) = if frac <= 1e-6 {
                    (0.0, 0u8)
                } else if frac >= 1.0 - 1e-6 {
                    ((area_w - w).max(0.0), 2u8)
                } else {
                    ((x - w / 2.0).clamp(0.0, (area_w - w).max(0.0)), 1u8)
                };
                views.push(Label::new(txt).style(move |s| {
                    let s = s
                        .absolute()
                        .inset_left(left.pt())
                        .inset_top(9.0.pt())
                        .width(w.pt())
                        .height(14.0.pt())
                        .font_family(FONT.to_string())
                        .font_size(11.0)
                        .color(LABEL);
                    match align {
                        0 => s.justify_start(),
                        2 => s.justify_end(),
                        _ => s.justify_center(),
                    }
                }));
                t += step;
            }
        }
    }
    Stack::from_iter(views).style(|s| s.size_full())
}

fn legend_view(meta: Option<SpecMeta>, area_h: f64, gradient: Vec<u8>) -> impl IntoView {
    let bar = img(move || gradient.clone()).style(|s| s.width(16.0.pt()).height_full());

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
            labels.push(Label::new(txt).style(move |s| {
                s.absolute()
                    .inset_top(top.pt())
                    .inset_left(6.0.pt())
                    .width((LEGEND_W - 24.0).pt())
                    .height(14.0.pt())
                    .font_family(FONT.to_string())
                    .font_size(11.0)
                    .color(LABEL)
            }));
            d += 20.0;
        }
    }
    let scale = Stack::from_iter(labels).style(|s| s.flex_grow(1.0).height_full());

    // Pad left so the colour bar doesn't butt up against the spectrogram.
    Stack::horizontal((bar, scale)).style(|s| s.size_full().padding_left(LEGEND_GAP_PX.pt()))
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
