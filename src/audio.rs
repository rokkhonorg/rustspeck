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
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
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

pub fn read<P: AsRef<Path>>(path: P) -> Result<Audio, String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| format!("cannot open file: {e}"))?;
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
        .map_err(|e| format!("unsupported or unrecognised audio format: {e}"))?;

    // Pick the first decodable audio track and build its decoder. Scoped so the
    // immutable borrow of `format` is released before the decode loop below.
    let (track_id, mut decoder) = {
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.as_ref().is_some_and(|p| p.is_audio()))
            .ok_or_else(|| "no decodable audio track found".to_string())?;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| "track has no audio codec parameters".to_string())?;
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(params, &AudioDecoderOptions::default())
            .map_err(|e| format!("no decoder for this codec: {e}"))?;
        (track.id, decoder)
    };

    let mut rate = 0u32;
    let mut channels = 0u32;
    let mut samples: Vec<i32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break, // clean end of stream
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("error reading stream: {e}")),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                if rate == 0 {
                    let spec = decoded.spec();
                    rate = spec.rate();
                    channels = spec.channels().count() as u32;
                }
                append_samples(decoded, &mut samples);
            }
            // A single corrupt packet shouldn't abort the whole decode.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break
            }
            Err(e) => return Err(format!("decode error: {e}")),
        }
    }

    if rate == 0 || channels == 0 {
        return Err("could not determine sample rate / channel count".into());
    }

    Ok(Audio {
        rate: rate as f64,
        channels,
        samples,
    })
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
