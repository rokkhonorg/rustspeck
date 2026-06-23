//! Forward DFT power spectrum, backed by `realfft` (a real-input FFT layer over
//! `rustfft`).
//!
//! SoX's `spectrogram.c` only ever needs the *power spectrum* of a windowed
//! real frame (`|X[k]|^2` for `k` in `0..=n/2`), accumulated across the blocks
//! that make up one output column. The original C used Ooura's `fft4g` for
//! power-of-2 sizes and a slow O(n^2) DFT otherwise.
//!
//! Because the input is real, the spectrum is conjugate-symmetric and the upper
//! half is redundant — we already only read bins `0..=n/2`. So instead of a full
//! length-`n` complex FFT with a zero imaginary part, we use a real-to-complex
//! transform, which computes those `n/2+1` bins via a length-`n/2` complex FFT
//! plus an O(n) recombination — roughly half the work and memory. The output is
//! the same DFT bins (within floating-point tolerance), so the spectrogram is
//! unchanged.
//!
//! Note: results match the C to within normal floating-point tolerance rather
//! than bit-for-bit — the FFT uses a different algorithm and operation order.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

/// Forward real-input DFT for a fixed transform length. Build once per length
/// and reuse; the internal buffers are reused across calls.
pub struct Dft {
    n: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    input: Vec<f64>,            // length n (overwritten each call)
    spectrum: Vec<Complex<f64>>, // length n/2 + 1
    scratch: Vec<Complex<f64>>,
}

impl Dft {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "DFT length must be >= 2");
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(n);
        let input = r2c.make_input_vec(); // length n
        let spectrum = r2c.make_output_vec(); // length n/2 + 1
        let scratch = r2c.make_scratch_vec();
        Dft {
            n,
            r2c,
            input,
            spectrum,
            scratch,
        }
    }

    /// Transform the real `input` (length == n) and add `|X[k]|^2` into
    /// `out[k]` for `k` in `0..=n/2`. `out` is accumulated into, not cleared —
    /// mirroring how the spectrogram sums several blocks per column.
    pub fn accumulate_power(&mut self, input: &[f64], out: &mut [f64]) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert!(out.len() >= self.n / 2 + 1);

        // realfft consumes (overwrites) its input buffer, so transform a copy and
        // leave the caller's slice intact.
        self.input.copy_from_slice(input);
        self.r2c
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .expect("realfft: buffer sizes are fixed at construction");

        for (o, c) in out.iter_mut().zip(self.spectrum.iter()).take(self.n / 2 + 1) {
            *o += c.re * c.re + c.im * c.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Naive DFT magnitude-squared reference.
    fn naive_power(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut out = vec![0.0; n / 2 + 1];
        for (k, o) in out.iter_mut().enumerate() {
            let mut re = 0.0;
            let mut im = 0.0;
            for (i, &x) in input.iter().enumerate() {
                let ang = 2.0 * PI * k as f64 * i as f64 / n as f64;
                re += x * ang.cos();
                im += x * ang.sin();
            }
            *o = re * re + im * im;
        }
        out
    }

    fn check(n: usize) {
        let input: Vec<f64> = (0..n)
            .map(|i| (0.3 * i as f64).sin() + 0.5 * (0.13 * i as f64 + 1.0).cos())
            .collect();
        let reference = naive_power(&input);

        let mut dft = Dft::new(n);
        let mut power = vec![0.0; n / 2 + 1];
        dft.accumulate_power(&input, &mut power);

        for (k, (&got, &want)) in power.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "n={n} bin {k}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn power_of_two_sizes_match_naive_dft() {
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 1024] {
            check(n);
        }
    }

    #[test]
    fn non_power_of_two_sizes_match_naive_dft() {
        // e.g. y_size = 64 -> dft_size = 126; also a few odd/mixed-radix sizes
        for &n in &[126usize, 100, 200, 384, 546] {
            check(n);
        }
    }

    #[test]
    fn accumulates_without_clearing() {
        let n = 32;
        let input: Vec<f64> = (0..n).map(|i| (0.2 * i as f64).sin()).collect();
        let mut dft = Dft::new(n);
        let mut a = vec![0.0; n / 2 + 1];
        dft.accumulate_power(&input, &mut a);
        let mut b = a.clone();
        dft.accumulate_power(&input, &mut b);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((y - 2.0 * x).abs() <= 1e-9 * (1.0 + x.abs()));
        }
    }
}
