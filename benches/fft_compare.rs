//! Micro-benchmark: real-input FFT (realfft) vs full complex FFT (rustfft) for
//! the spectrogram's power-spectrum kernel.
//!
//! Both variants are reproduced here so they can be timed side-by-side in one
//! process (the real `fft::Dft` lives inside the bin crate and isn't importable
//! from a bench). `RustfftDft` mirrors the pre-change implementation; `RealfftDft`
//! mirrors the current one. Each computes `sum |X[k]|^2` for `k in 0..=n/2` of a
//! real frame — the exact work `Channel::process_block` does per block.
//!
//! Run with `cargo bench --bench fft_compare`.

use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

/// Pre-change: full length-n complex FFT with a zeroed imaginary part.
struct RustfftDft {
    n: usize,
    fft: Arc<dyn Fft<f64>>,
    buf: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl RustfftDft {
    fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        RustfftDft {
            n,
            fft,
            buf: vec![Complex::new(0.0, 0.0); n],
            scratch,
        }
    }

    fn accumulate_power(&mut self, input: &[f64], out: &mut [f64]) {
        for (slot, &x) in self.buf.iter_mut().zip(input) {
            slot.re = x;
            slot.im = 0.0;
        }
        self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
        for (k, o) in out.iter_mut().enumerate().take(self.n / 2 + 1) {
            let c = self.buf[k];
            *o += c.re * c.re + c.im * c.im;
        }
    }
}

/// Current: real-to-complex transform (length-n/2 complex FFT + recombination).
struct RealfftDft {
    n: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    input: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
}

impl RealfftDft {
    fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(n);
        let input = r2c.make_input_vec();
        let spectrum = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();
        RealfftDft {
            n,
            r2c,
            input,
            spectrum,
            scratch,
        }
    }

    fn accumulate_power(&mut self, input: &[f64], out: &mut [f64]) {
        self.input.copy_from_slice(input);
        self.r2c
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .unwrap();
        for (o, c) in out.iter_mut().zip(self.spectrum.iter()).take(self.n / 2 + 1) {
            *o += c.re * c.re + c.im * c.im;
        }
    }
}

/// A deterministic real frame, roughly windowed so it looks like real input.
fn make_frame(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let win = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos(); // Hann
            win * ((13.0 * t).sin() + 0.5 * (47.0 * t + 1.0).cos() + 0.2 * (113.0 * t).sin())
        })
        .collect()
}

/// Median nanoseconds per call, auto-tuning the iteration count to a stable
/// window and taking the median of several trials.
fn time_ns(mut call: impl FnMut()) -> f64 {
    for _ in 0..64 {
        call(); // warm up
    }
    let bench_once = |iters: u64, call: &mut dyn FnMut()| -> f64 {
        let t = Instant::now();
        for _ in 0..iters {
            call();
        }
        t.elapsed().as_nanos() as f64 / iters as f64
    };
    // Grow iters until a single batch takes >= 120 ms, so timer granularity is
    // negligible.
    let mut iters = 256u64;
    loop {
        let t = Instant::now();
        for _ in 0..iters {
            call();
        }
        if t.elapsed() >= Duration::from_millis(120) {
            break;
        }
        iters = iters.saturating_mul(2);
    }
    let mut trials: Vec<f64> = (0..5).map(|_| bench_once(iters, &mut call)).collect();
    trials.sort_by(|a, b| a.partial_cmp(b).unwrap());
    trials[trials.len() / 2]
}

fn main() {
    // Sizes that actually occur: GUI dft_size is ~1024–2048 (and up to 16384 with
    // -F); 1536/6144 exercise mixed-radix (non-power-of-two) paths.
    let sizes = [512usize, 1024, 1536, 2048, 4096, 6144, 8192, 16384];

    println!(
        "{:>7}  {:>14}  {:>14}  {:>9}",
        "n", "rustfft ns", "realfft ns", "speedup"
    );
    println!("{}", "-".repeat(52));

    for &n in &sizes {
        let frame = make_frame(n);
        let mut out = vec![0.0f64; n / 2 + 1];

        let mut rf = RustfftDft::new(n);
        let rustfft_ns = time_ns(|| {
            rf.accumulate_power(black_box(&frame), black_box(&mut out));
        });

        let mut re = RealfftDft::new(n);
        let realfft_ns = time_ns(|| {
            re.accumulate_power(black_box(&frame), black_box(&mut out));
        });

        println!(
            "{:>7}  {:>14.1}  {:>14.1}  {:>8.2}x",
            n,
            rustfft_ns,
            realfft_ns,
            rustfft_ns / realfft_ns
        );
    }
}
