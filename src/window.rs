//! Window functions — faithful port of the relevant parts of
//! `effects_i_dsp.c` (Hann/Hamming/Bartlett/Kaiser/Dolph + Bessel/Kaiser-beta).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display, strum::EnumIter)]
pub enum WindowType {
    Hann,
    Hamming,
    Bartlett,
    Rectangular,
    Kaiser,
    Dolph,
}

/// `lsx_bessel_I_0`
pub fn bessel_i0(x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    let x2 = x / 2.0;
    let mut i = 1.0;
    loop {
        let y = x2 / i;
        i += 1.0;
        let last_sum = sum;
        term *= y * y;
        sum += term;
        if sum == last_sum {
            break;
        }
    }
    sum
}

/// `lsx_apply_hann`
pub fn apply_hann(h: &mut [f64]) {
    let m = (h.len() - 1) as f64;
    for (i, hi) in h.iter_mut().enumerate() {
        let x = 2.0 * PI * i as f64 / m;
        *hi *= 0.5 - 0.5 * x.cos();
    }
}

/// `lsx_apply_hamming`
pub fn apply_hamming(h: &mut [f64]) {
    let m = (h.len() - 1) as f64;
    for (i, hi) in h.iter_mut().enumerate() {
        let x = 2.0 * PI * i as f64 / m;
        *hi *= 0.53836 - 0.46164 * x.cos();
    }
}

/// `lsx_apply_bartlett`
pub fn apply_bartlett(h: &mut [f64]) {
    let m = (h.len() - 1) as f64;
    for (i, hi) in h.iter_mut().enumerate() {
        *hi *= 2.0 / m * (m / 2.0 - (i as f64 - m / 2.0).abs());
    }
}

/// `lsx_kaiser_beta`
pub fn kaiser_beta(att: f64, tr_bw: f64) -> f64 {
    if att >= 60.0 {
        const COEFS: [[f64; 4]; 10] = [
            [-6.784957e-10, 1.02856e-05, 0.1087556, -0.8988365 + 0.001],
            [-6.897885e-10, 1.027433e-05, 0.10876, -0.8994658 + 0.002],
            [-1.000683e-09, 1.030092e-05, 0.1087677, -0.9007898 + 0.003],
            [-3.654474e-10, 1.040631e-05, 0.1087085, -0.8977766 + 0.006],
            [8.106988e-09, 6.983091e-06, 0.1091387, -0.9172048 + 0.015],
            [9.519571e-09, 7.272678e-06, 0.1090068, -0.9140768 + 0.025],
            [-5.626821e-09, 1.342186e-05, 0.1083999, -0.9065452 + 0.05],
            [-9.965946e-08, 5.073548e-05, 0.1040967, -0.7672778 + 0.085],
            [1.604808e-07, -5.856462e-05, 0.1185998, -1.34824 + 0.1],
            [-1.511964e-07, 6.363034e-05, 0.1064627, -0.9876665 + 0.18],
        ];
        let realm = (tr_bw / 0.0005).ln() / 2f64.ln();
        let n = COEFS.len() as i32;
        let i0 = range_limit(realm as i32, 0, n - 1) as usize;
        let i1 = range_limit(1 + realm as i32, 0, n - 1) as usize;
        let c0 = &COEFS[i0];
        let c1 = &COEFS[i1];
        let b0 = ((c0[0] * att + c0[1]) * att + c0[2]) * att + c0[3];
        let b1 = ((c1[0] * att + c1[1]) * att + c1[2]) * att + c1[3];
        return b0 + (b1 - b0) * (realm - realm as i32 as f64);
    }
    if att > 50.0 {
        return 0.1102 * (att - 8.7);
    }
    if att > 20.96 {
        return 0.58417 * (att - 20.96).powf(0.4) + 0.07886 * (att - 20.96);
    }
    0.0
}

/// `lsx_apply_kaiser`
pub fn apply_kaiser(h: &mut [f64], beta: f64) {
    let m = (h.len() - 1) as f64;
    let denom = bessel_i0(beta);
    for (i, hi) in h.iter_mut().enumerate() {
        let x = 2.0 * i as f64 / m - 1.0;
        *hi *= bessel_i0(beta * (1.0 - x * x).sqrt()) / denom;
    }
}

/// `lsx_apply_dolph`
pub fn apply_dolph(h: &mut [f64], att: f64) {
    let big_n = h.len() as i32;
    let nf = big_n as f64;
    let b0 = ((10f64.powf(att / 20.0)).acosh() / (nf - 1.0)).cosh();
    let c = 1.0 - 1.0 / (b0 * b0);
    let mut norm = 0.0;
    let mut i = (big_n - 1) / 2;
    while i >= 0 {
        let fi = i as f64;
        let mut sum = if i == 0 { 1.0 } else { 0.0 };
        let mut t = 1.0;
        let mut b = 1.0;
        let mut j = 1i32;
        // C for-loop: cond `j <= i && sum != t` checked at top.
        while j <= i && sum != t {
            let fj = j as f64;
            t = sum;
            b *= c * (nf - fi - fj) * (1.0 / fj);
            sum += b;
            // increment expression
            b *= (fi - fj) * (1.0 / fj);
            j += 1;
        }
        sum /= nf - 1.0 - fi;
        norm = if norm != 0.0 { norm } else { sum };
        sum /= norm;
        h[i as usize] *= sum;
        h[(big_n - 1 - i) as usize] *= sum;
        i -= 1;
    }
}

fn range_limit(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_endpoints_and_peak() {
        let mut h = vec![1.0; 9];
        apply_hann(&mut h);
        assert!(h[0].abs() < 1e-12);
        assert!(h[8].abs() < 1e-12);
        assert!((h[4] - 1.0).abs() < 1e-12); // centre of Hann is 1
    }

    #[test]
    fn bessel_i0_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        // I0(1) ~ 1.2660658777520084
        assert!((bessel_i0(1.0) - 1.2660658777520084).abs() < 1e-10);
    }

    #[test]
    fn windows_are_symmetric() {
        for apply in [apply_hann as fn(&mut [f64]), apply_hamming, apply_bartlett] {
            let mut h = vec![1.0; 17];
            apply(&mut h);
            for i in 0..h.len() {
                assert!((h[i] - h[h.len() - 1 - i]).abs() < 1e-12, "asymmetry at {i}");
            }
        }
    }

    #[test]
    fn dolph_and_kaiser_run_and_are_symmetric() {
        let mut h = vec![1.0; 33];
        apply_kaiser(&mut h, kaiser_beta(120.0, 0.1));
        for i in 0..h.len() {
            assert!((h[i] - h[h.len() - 1 - i]).abs() < 1e-9);
        }
        let mut h = vec![1.0; 33];
        apply_dolph(&mut h, 126.0);
        for i in 0..h.len() {
            assert!((h[i] - h[h.len() - 1 - i]).abs() < 1e-9);
        }
    }
}
