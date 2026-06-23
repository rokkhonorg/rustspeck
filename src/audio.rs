//! Audio decoding via Symphonia (pure-Rust), producing samples in SoX's native
//! `sox_sample_t` (i32) domain.
//!
//! Symphonia decodes WAV/FLAC/MP3/OGG/AAC/ALAC/AIFF/CAF/… (whatever features
//! are enabled). Whatever the source format, every decoded sample is bridged
//! through SoX's exact conversions from `sox.h`, so that lossless integer
//! sources (WAV/FLAC) remain bit-identical to running
//! `sox in -n spectrogram`, and float sources match SoX's float→sample rule.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::sample::{i24, u24};
// `Audio` is Symphonia's planar-buffer trait (provides `plane`); imported
// anonymously so it doesn't clash with our own `Audio` struct below.
use symphonia::core::audio::{Audio as _, GenericAudioBufferRef};
use symphonia::core::codecs::audio::{AudioDecoder, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Decoded audio with samples interleaved in the i32 `sox_sample_t` domain.
pub struct Audio {
    pub rate: f64,
    pub channels: u32,
    /// Interleaved samples: frame f channel c is at `samples[f * channels + c]`.
    pub samples: Vec<i32>,
}

impl Audio {
    /// Total per-channel frame count.
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
}

/// A streaming decoder: open once, then pull interleaved `sox_sample_t` chunks.
///
/// Symphonia (pure Rust) is tried first. When it can't handle the container or
/// codec and the `ffmpeg` feature is enabled, the file is decoded via FFmpeg
/// instead and served from an in-memory buffer ([`Buffered`]). Either way
/// `rate`/`channels` are known after `open`, and `total_len` is the interleaved
/// stream length when known (always, for the FFmpeg path), which the
/// spectrogram needs to fix its geometry.
pub enum Decoder {
    Symphonia(SymphoniaDecoder),
    Buffered(Buffered),
}

impl Decoder {
    pub fn rate(&self) -> f64 {
        match self {
            Decoder::Symphonia(d) => d.rate as f64,
            Decoder::Buffered(d) => d.rate as f64,
        }
    }
    pub fn channels(&self) -> u32 {
        match self {
            Decoder::Symphonia(d) => d.channels,
            Decoder::Buffered(d) => d.channels,
        }
    }
    /// Interleaved sample count of the whole stream, if known.
    pub fn total_len(&self) -> Option<u64> {
        match self {
            Decoder::Symphonia(d) => d.total_frames.map(|f| f * d.channels as u64),
            Decoder::Buffered(d) => Some(d.total),
        }
    }

    /// Decode and return the next chunk's interleaved samples, or `None` at the
    /// end of the stream.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<i32>>, String> {
        match self {
            Decoder::Symphonia(d) => d.next_chunk(),
            Decoder::Buffered(d) => Ok(d.next_chunk()),
        }
    }
}

/// The Symphonia-backed streaming decoder.
pub struct SymphoniaDecoder {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    rate: u32,
    channels: u32,
    total_frames: Option<u64>,
    /// First decoded chunk, buffered during `open` to learn `rate`/`channels`.
    pending_first: Option<Vec<i32>>,
}

impl SymphoniaDecoder {
    /// Decode and return the next packet's interleaved samples, or `None` at the
    /// end of the stream. Corrupt packets are skipped.
    fn next_chunk(&mut self) -> Result<Option<Vec<i32>>, String> {
        if let Some(first) = self.pending_first.take() {
            return Ok(Some(first));
        }
        loop {
            let packet = match self.format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => return Ok(None),
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(SymphoniaError::ResetRequired) => return Ok(None),
                Err(e) => return Err(format!("error reading stream: {e}")),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(decoded) => {
                    let mut chunk = Vec::new();
                    append_samples(decoded, &mut chunk);
                    return Ok(Some(chunk));
                }
                // A single corrupt packet shouldn't abort the whole decode.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(e) => return Err(format!("decode error: {e}")),
            }
        }
    }
}

/// A decoder serving from a fully decoded in-memory buffer (the FFmpeg fallback
/// path; we read all of FFmpeg's PCM output up front and hand it over in one
/// chunk).
pub struct Buffered {
    rate: u32,
    channels: u32,
    /// Interleaved sample count, kept stable for `total_len` after the samples
    /// are taken by `next_chunk`.
    total: u64,
    samples: Option<Vec<i32>>,
}

impl Buffered {
    fn next_chunk(&mut self) -> Option<Vec<i32>> {
        self.samples.take()
    }
}

/// Open a file for streaming decode. Tries Symphonia first; if it can't handle
/// the container/codec and the `ffmpeg` feature is enabled, falls back to FFmpeg.
pub fn open<P: AsRef<Path>>(path: P) -> Result<Decoder, String> {
    let path = path.as_ref();
    match open_symphonia(path) {
        Ok(d) => Ok(Decoder::Symphonia(d)),
        // A genuine I/O problem (missing/unreadable file) won't be helped by a
        // different decoder, so surface it directly rather than masking it behind
        // the fallback's error.
        Err(OpenError::Io(e)) => Err(e),
        Err(OpenError::Unsupported(e)) => fallback(path, e),
    }
}

/// Fallback for a format/codec Symphonia couldn't handle: decode with the system
/// `ffmpeg` binary if it's available. When `ffmpeg` isn't on `PATH`, surface the
/// original Symphonia error (plus a hint) rather than the missing-binary error.
fn fallback(path: &Path, symphonia_err: String) -> Result<Decoder, String> {
    match ffmpeg_decode::decode(path) {
        Ok(buffered) => Ok(Decoder::Buffered(buffered)),
        Err(ffmpeg_decode::Error::NotFound) => Err(format!(
            "{symphonia_err} (install ffmpeg and put it on PATH to decode more formats)"
        )),
        Err(ffmpeg_decode::Error::Failed(fe)) => {
            Err(format!("{symphonia_err}; ffmpeg fallback also failed: {fe}"))
        }
    }
}

/// Why [`open_symphonia`] failed, so [`open`] can decide whether a fallback is
/// worth attempting.
enum OpenError {
    /// The file couldn't be opened/read — not a codec problem.
    Io(String),
    /// Symphonia doesn't support this container/codec (or couldn't get audio out
    /// of it) — a different decoder might.
    Unsupported(String),
}

/// Open via Symphonia, decoding the first packet eagerly to learn the
/// authoritative sample rate / channel count.
fn open_symphonia(path: &Path) -> Result<SymphoniaDecoder, OpenError> {
    let file = File::open(path).map_err(|e| OpenError::Io(format!("cannot open file: {e}")))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| OpenError::Unsupported(format!("unsupported or unrecognised audio format: {e}")))?;

    let (track_id, total_frames, mut decoder) = {
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.as_ref().is_some_and(|p| p.is_audio()))
            .ok_or_else(|| OpenError::Unsupported("no decodable audio track found".to_string()))?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| OpenError::Unsupported("track has no audio codec parameters".to_string()))?;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| OpenError::Unsupported(format!("no decoder for this codec: {e}")))?;
        (track.id, track.num_frames, decoder)
    };

    // Decode packets until the first one for our track yields audio, so we know
    // the true rate/channels and can buffer that chunk for the first read.
    let mut rate = 0u32;
    let mut channels = 0u32;
    let mut pending_first: Option<Vec<i32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(OpenError::Unsupported(format!("error reading stream: {e}"))),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                rate = spec.rate();
                channels = spec.channels().count() as u32;
                let mut chunk = Vec::new();
                append_samples(decoded, &mut chunk);
                pending_first = Some(chunk);
                break;
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(e) => return Err(OpenError::Unsupported(format!("decode error: {e}"))),
        }
    }

    if rate == 0 || channels == 0 {
        return Err(OpenError::Unsupported(
            "could not determine sample rate / channel count".into(),
        ));
    }

    Ok(SymphoniaDecoder {
        format,
        decoder,
        track_id,
        rate,
        channels,
        total_frames,
        pending_first,
    })
}

/// Decode an entire file into memory (batch path, used by the CLI). Built on the
/// streaming [`Decoder`].
pub fn read<P: AsRef<Path>>(path: P) -> Result<Audio, String> {
    let mut dec = open(path)?;
    let rate = dec.rate();
    let channels = dec.channels();
    let mut samples: Vec<i32> = Vec::new();
    if let Some(n) = dec.total_len() {
        samples.reserve(n as usize);
    }
    while let Some(chunk) = dec.next_chunk()? {
        samples.extend_from_slice(&chunk);
    }
    Ok(Audio {
        rate,
        channels,
        samples,
    })
}

/// Decode fallback that shells out to the system `ffmpeg`/`ffprobe` binaries for
/// containers/codecs Symphonia can't handle.
///
/// `ffprobe` reports the native rate/channels; `ffmpeg` transcodes the audio to
/// raw interleaved **signed-32 little-endian** PCM on stdout. S32 spans the full
/// i32 range — exactly the `sox_sample_t` domain — so the bytes map straight to
/// samples with no scaling. The whole stream is buffered (the fallback is a cold
/// path), so no FFmpeg dev libraries or build-time linking are involved; it just
/// needs the binaries on `PATH` at runtime.
mod ffmpeg_decode {
    use std::path::Path;
    use std::process::Command;

    use super::Buffered;

    /// Why the FFmpeg fallback didn't produce audio.
    pub enum Error {
        /// The `ffmpeg`/`ffprobe` binary isn't installed / not on `PATH`.
        NotFound,
        /// The binary ran but couldn't decode the file.
        Failed(String),
    }

    pub fn decode(path: &Path) -> Result<Buffered, Error> {
        let (rate, channels) = probe(path)?;

        // Raw interleaved s32le at the native rate/layout (no -ar/-ac), straight
        // to stdout. `-v error` keeps stderr to genuine errors only.
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-map", "a:0", "-f", "s32le", "-c:a", "pcm_s32le", "-"])
            .output()
            .map_err(spawn_error)?;
        if !out.status.success() {
            return Err(Error::Failed(format!(
                "ffmpeg: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }

        // s32le bytes -> sox_sample_t i32 (chunks_exact drops any partial tail).
        let samples: Vec<i32> = out
            .stdout
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if samples.is_empty() {
            return Err(Error::Failed("ffmpeg produced no audio samples".into()));
        }

        Ok(Buffered {
            rate,
            channels,
            total: samples.len() as u64,
            samples: Some(samples),
        })
    }

    /// Read the first audio stream's sample rate and channel count via `ffprobe`.
    fn probe(path: &Path) -> Result<(u32, u32), Error> {
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "default=noprint_wrappers=1:nokey=0",
            ])
            .arg(path)
            .output()
            .map_err(spawn_error)?;
        if !out.status.success() {
            return Err(Error::Failed(format!(
                "ffprobe: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }

        // Parse `sample_rate=...` / `channels=...` lines (order-independent).
        let text = String::from_utf8_lossy(&out.stdout);
        let mut rate = 0u32;
        let mut channels = 0u32;
        for line in text.lines() {
            if let Some(v) = line.trim().strip_prefix("sample_rate=") {
                rate = v.trim().parse().unwrap_or(0);
            } else if let Some(v) = line.trim().strip_prefix("channels=") {
                channels = v.trim().parse().unwrap_or(0);
            }
        }
        if rate == 0 || channels == 0 {
            return Err(Error::Failed(
                "ffprobe: could not read sample rate / channel count".into(),
            ));
        }
        Ok((rate, channels))
    }

    /// Map a spawn failure to `NotFound` (binary missing) vs a real `Failed`.
    fn spawn_error(e: std::io::Error) -> Error {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::NotFound
        } else {
            Error::Failed(format!("could not run ffmpeg: {e}"))
        }
    }
}

/// Interleave a decoded planar buffer into `out`, converting each sample to
/// `sox_sample_t` using SoX's exact per-format rule.
fn append_samples(decoded: GenericAudioBufferRef<'_>, out: &mut Vec<i32>) {
    let chans = decoded.spec().channels().count();
    let frames = decoded.frames();
    out.reserve(frames * chans);

    macro_rules! go {
        ($buf:expr, $conv:expr) => {{
            let buf = $buf;
            let planes: Vec<&[_]> = (0..chans).map(|c| buf.plane(c).unwrap_or(&[])).collect();
            for f in 0..frames {
                for plane in &planes {
                    out.push($conv(plane[f]));
                }
            }
        }};
    }

    match decoded {
        GenericAudioBufferRef::U8(b) => go!(b, |s: u8| unsigned_to_sample(8, s as u32)),
        GenericAudioBufferRef::U16(b) => go!(b, |s: u16| unsigned_to_sample(16, s as u32)),
        GenericAudioBufferRef::U24(b) => go!(b, |s: u24| unsigned_to_sample(24, s.0)),
        GenericAudioBufferRef::U32(b) => go!(b, |s: u32| unsigned_to_sample(32, s)),
        GenericAudioBufferRef::S8(b) => go!(b, |s: i8| signed_to_sample(8, s as i32)),
        GenericAudioBufferRef::S16(b) => go!(b, |s: i16| signed_to_sample(16, s as i32)),
        GenericAudioBufferRef::S24(b) => go!(b, |s: i24| signed_to_sample(24, s.0)),
        GenericAudioBufferRef::S32(b) => go!(b, |s: i32| signed_to_sample(32, s)),
        GenericAudioBufferRef::F32(b) => go!(b, |s: f32| float_to_sample(s as f64)),
        GenericAudioBufferRef::F64(b) => go!(b, |s: f64| float_to_sample(s)),
    }
}

// --- SoX sox.h conversions (preserved so output stays bit-exact with SoX) ---

const SOX_SAMPLE_MAX: i64 = 0x7FFF_FFFF;
const SOX_SAMPLE_MIN: i64 = -0x8000_0000;
const SOX_SAMPLE_NEG: i32 = i32::MIN;

/// `SOX_SIGNED_TO_SAMPLE(bits, d)` = `(sox_sample_t)d << (32 - bits)`
fn signed_to_sample(bits: u32, d: i32) -> i32 {
    ((d as u32).wrapping_shl(32 - bits)) as i32
}

/// `SOX_UNSIGNED_TO_SAMPLE(bits, d)` = signed_to_sample(bits, d) ^ SOX_SAMPLE_NEG
fn unsigned_to_sample(bits: u32, d: u32) -> i32 {
    signed_to_sample(bits, d as i32) ^ SOX_SAMPLE_NEG
}

/// `SOX_FLOAT_64BIT_TO_SAMPLE(d, clips)`
fn float_to_sample(d: f64) -> i32 {
    let tmp = d * (SOX_SAMPLE_MAX as f64 + 1.0);
    let v: f64 = if tmp < 0.0 {
        if tmp <= SOX_SAMPLE_MIN as f64 - 0.5 {
            return SOX_SAMPLE_MIN as i32;
        }
        tmp - 0.5
    } else if tmp >= SOX_SAMPLE_MAX as f64 + 0.5 {
        return SOX_SAMPLE_MAX as i32;
    } else {
        tmp + 0.5
    };
    v as i32 // C cast truncates toward zero, matching `as i32` in range
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed16_conversion_matches_sox() {
        assert_eq!(signed_to_sample(16, 1), 1 << 16);
        assert_eq!(signed_to_sample(16, -1), -(1 << 16));
        assert_eq!(signed_to_sample(16, i16::MAX as i32), (i16::MAX as i32) << 16);
        assert_eq!(signed_to_sample(16, i16::MIN as i32), i32::MIN);
    }

    #[test]
    fn unsigned8_conversion_matches_sox() {
        assert_eq!(unsigned_to_sample(8, 128), 0);
        assert_eq!(unsigned_to_sample(8, 0), i32::MIN);
        assert_eq!(unsigned_to_sample(8, 255), 0x7F00_0000);
    }

    #[test]
    fn float_conversion_rounds_and_clips() {
        assert_eq!(float_to_sample(0.0), 0);
        assert_eq!(float_to_sample(1.0), SOX_SAMPLE_MAX as i32);
        assert_eq!(float_to_sample(-1.0), SOX_SAMPLE_MIN as i32);
        let half = 1.0 / (SOX_SAMPLE_MAX as f64 + 1.0);
        assert_eq!(float_to_sample(half), 1);
    }
}
