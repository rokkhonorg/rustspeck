---
name: port-status
description: Status of the SoX spectrogram -> Rust port (rustspek), what's verified and what's left
metadata:
  type: project
---

Porting SoX's `spectrogram` effect to Rust (WAV input only). Source of truth: `sox-code/` (version **14.4.3git** per configure.ac). Installed SoX binary is **14.4.2 (32-bit, x87)** at `C:\Program Files (x86)\sox-14-4-2\sox.exe`.

**Modules (all written, builds clean 0 warnings, 10 unit tests pass):** `fft.rs` (**rustfft** — single `Dft` type, `accumulate_power`; replaced hand-ported Ooura rdft AND slow non-p2 DFT), `audio.rs` (**symphonia** — replaced the hand-rolled `wav.rs`; decodes all symphonia formats, bridges every sample through SoX's `signed/unsigned/float_to_sample`), `window.rs`, `timeparse.rs` (-d/-S), `spectrogram.rs` (Config/geometry/flow/drain), `render.rs` (palette/font/axes/PNG), `tables.rs` (FIXED_FONT_ZLIB + ALT_PALETTE), `main.rs` (clap CLI, humanized long flags).

**Deps:** clap (derive), png, flate2, rustfft, symphonia (features=["all"]) — user adds via `cargo add` themselves (see [[deps-and-edition]]).

**Ongoing effort: replacing C-style ported code with proper Rust libraries.**
- FFT → rustfft (2026-06-08): output *identical* to prior Ooura build (0 px diff), still 0-diff vs SoX in raw mode incl. non-p2 sizes. rustfft ~1e-13 diffs are below the colour-quantization step.
- WAV → symphonia 0.6 (2026-06-08): bit-exact preserved for ALL lossless paths — verified 0 raw-mode diffs vs SoX for 8/16/24/32-bit + 32f WAV AND FLAC. Adds MP3/OGG/AAC/ALAC/etc. (MP3 couldn't be round-trip tested: SoX 14.4.2 has no LAME encoder). Symphonia 0.6 API notes: `get_probe().probe(&hint, mss, FormatOptions, MetadataOptions)` (by value) → `Box<dyn FormatReader>`; `next_packet()->Result<Option<Packet>>`; buffers are `GenericAudioBufferRef` (variants U8..F64); planar access via the `Audio` trait `.plane(c)`; `i24/u24` are tuple structs (`.0`); decoder via `get_codecs().make_audio_decoder(params.audio(), &AudioDecoderOptions::default())`.

**Assessment of remaining C-style code (told user 2026-06-08):** `timeparse.rs` is NOT a good library candidate — SoX's `mm:ss`/sample-count/`=+-` anchor grammar matches no popular crate; leave it or do an idiomatic rewrite. `window.rs` must stay (SoX-specific Kaiser-beta poly + Dolph). `render.rs`/`tables.rs` font+palette must stay for pixel fidelity. Next genuine win if asked: error handling `Result<_,String>` → thiserror/anyhow.

**Verification vs installed SoX (test/ has mono.wav, stereo.wav + generators):**
- `-r` raw mode (pure spectrogram data): **bit-exact**, 0/410400 px diff. 36/39 x-values in a sweep are 0-diff.
- Full images: identical except divergences traced to the 14.4.2 binary being a **32-bit x87 build** (80-bit intermediates). Two confirmed signatures, both "true value just below an integer, x87 truncates down, IEEE-754 double rounds up": (1) Z-legend gradient single row, `colour()` `83.999…` vs `84.0` (~14 px); (2) `step_size`/`block_steps` at specific x (e.g. 700: `44100/pps` = 188.99999999999999233 -> x87 188, double 189) which cascades. NOT a port bug — a modern 64-bit SoX build would match rustspek.
- `-A` alt palette differs because 14.4.2 alt_palette had 169 entries vs 14.4.3git source's 168 (genuine version difference).
- `-n` normalize: 14.4.2 binary lacks it (`invalid option -n`); it's a 14.4.3git feature, implemented per source.

**Left to do (was interrupted):** confirm 24-bit / 32-float / 8-bit WAV reading end-to-end (had just generated mono24/mono32f/mono8 via sox when stopped); write a README. `SPEK_DEBUG=1` env var prints geometry (mirrors sox -V).
