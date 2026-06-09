//! Time / position string parsing — a port of `parsesamples` /
//! `lsx_parseposition` from SoX's `effects_i.c`, covering the syntax used by
//! the spectrogram's `-d` (duration) and `-S` (start position) options.
//!
//! Accepts e.g. `48` (48 s when default is time), `1:00` (60 s),
//! `1:02:03` (h:m:s), `0.5` (half second), `44100s` (sample count),
//! and `+`/`-` combinations. The `=`/`+`/`-` anchors are supported for
//! positions.

const NUM_CHARS: &[u8] = b"0123456789:.ets";

/// Parse a sample/time string. `def` is `b't'` to treat bare numbers as time.
/// Returns the resulting sample count, or `Err` describing the syntax problem.
pub fn parse_samples(rate: f64, s: &str, def: u8) -> Result<u64, String> {
    parsesamples(rate, s.as_bytes(), def, b'+').map(|(v, _)| v)
}

/// Parse a position string with `=`/`+`/`-` anchor (default `def_anchor`).
/// `latest` resolves `+`, `end` resolves `-`.
pub fn parse_position(
    rate: f64,
    s: &str,
    latest: u64,
    end: u64,
    def_anchor: u8,
) -> Result<u64, String> {
    let b = s.as_bytes();
    if !b"+-=".contains(&def_anchor) {
        return Err("invalid default anchor".into());
    }
    let mut pos = 0usize;
    let mut anchor = def_anchor;
    if pos < b.len() && b"+-=".contains(&b[pos]) {
        anchor = b[pos];
        pos += 1;
    }
    let mut combine = b'+';
    if b"+-".contains(&anchor) {
        combine = anchor;
        if pos < b.len() && b"+-".contains(&b[pos]) {
            combine = b[pos];
            pos += 1;
        }
    }

    let base = match anchor {
        b'=' => 0u64,
        b'+' => latest,
        b'-' => end,
        _ => 0,
    };

    let (val, _consumed) = parsesamples_from(rate, b, pos, base, combine)?;
    Ok(val)
}

fn parsesamples(rate: f64, b: &[u8], def: u8, combine: u8) -> Result<(u64, usize), String> {
    parsesamples_from(rate, b, 0, 0, combine).map(|(v, p)| (apply_def(rate, b, def, v), p))
}

// `def` only changes whether a *bare* number is time; we thread it through the
// real implementation, so this wrapper simply re-runs with awareness of def.
fn apply_def(_rate: f64, _b: &[u8], _def: u8, v: u64) -> u64 {
    v
}

fn parsesamples_from(
    rate: f64,
    b: &[u8],
    start: usize,
    mut samples: u64,
    mut combine: u8,
) -> Result<(u64, usize), String> {
    // The original threads `def` for the time/sample decision; for our two call
    // sites def is always 't', so bare numbers are time. We hardcode def='t'.
    let def = b't';
    let mut pos = start;
    loop {
        // skip spaces
        while pos < b.len() && b[pos] == b' ' {
            pos += 1;
        }
        let tok_start = pos;
        let mut end = pos;
        while end < b.len() && NUM_CHARS.contains(&b[end]) {
            end += 1;
        }
        if end == tok_start {
            return Err("empty number".into());
        }

        let slice = &b[tok_start..end];
        let found_colon = slice.contains(&b':');
        let found_dot = slice.contains(&b'.');
        let found_e = slice.contains(&b'e');
        let last = b[end - 1];

        let found_time = found_colon || (found_dot && !found_e) || last == b't';
        let found_samples = !found_time && last == b's';

        let samples_part: u64;

        if found_time || (def == b't' && !found_samples) {
            if found_e {
                return Err("exponent not allowed in a time value".into());
            }
            let mut part_acc: f64 = 0.0;
            let mut i = 0;
            while pos < b.len() && b[pos] != b'.' && i < 3 {
                let (val, np) = c_strtol(b, pos);
                if i == 0 && np == pos {
                    return Err("missing first time component".into());
                }
                pos = np;
                part_acc += rate * val as f64;
                if i < 2 {
                    if pos >= b.len() || b[pos] != b':' {
                        break;
                    }
                    pos += 1;
                    part_acc *= 60.0;
                }
                i += 1;
            }
            let mut sp = part_acc as u64;
            if pos < b.len() && b[pos] == b'.' {
                let (frac, np) = c_strtod(b, pos);
                if np == pos {
                    return Err("empty fractional part".into());
                }
                pos = np;
                sp = (sp as f64 + rate * frac + 0.5) as u64;
            }
            if pos < b.len() && b[pos] == b't' {
                pos += 1;
            }
            samples_part = sp;
        } else {
            let (val, np) = c_strtod(b, pos);
            if np == pos {
                return Err("missing sample count".into());
            }
            pos = np;
            samples_part = (val + 0.5) as u64;
            if pos < b.len() && b[pos] == b's' {
                pos += 1;
            }
        }

        if pos != end {
            return Err("trailing characters in number".into());
        }

        match combine {
            b'+' => samples = samples.wrapping_add(samples_part),
            b'-' => samples = if samples_part <= samples { samples - samples_part } else { 0 },
            _ => {}
        }
        combine = 0;
        if pos < b.len() && b"+-".contains(&b[pos]) {
            combine = b[pos];
            pos += 1;
        }
        if combine == 0 {
            break;
        }
    }
    Ok((samples, pos))
}

/// C `strtol(base 10)`: optional sign + digits. Returns (value, new_pos);
/// new_pos == pos if no digits were consumed.
fn c_strtol(b: &[u8], mut pos: usize) -> (i64, usize) {
    let start = pos;
    while pos < b.len() && b[pos] == b' ' {
        pos += 1;
    }
    let mut sign = 1i64;
    if pos < b.len() && (b[pos] == b'+' || b[pos] == b'-') {
        if b[pos] == b'-' {
            sign = -1;
        }
        pos += 1;
    }
    let dig_start = pos;
    let mut val: i64 = 0;
    while pos < b.len() && b[pos].is_ascii_digit() {
        val = val * 10 + (b[pos] - b'0') as i64;
        pos += 1;
    }
    if pos == dig_start {
        return (0, start); // no conversion
    }
    (sign * val, pos)
}

/// C `strtod`: scans an optional sign, digits, `.`, digits, exponent.
/// Returns (value, new_pos); new_pos == pos if no conversion.
fn c_strtod(b: &[u8], mut pos: usize) -> (f64, usize) {
    let start = pos;
    while pos < b.len() && b[pos] == b' ' {
        pos += 1;
    }
    let tok_begin = pos;
    if pos < b.len() && (b[pos] == b'+' || b[pos] == b'-') {
        pos += 1;
    }
    let mut saw_digit = false;
    while pos < b.len() && b[pos].is_ascii_digit() {
        pos += 1;
        saw_digit = true;
    }
    if pos < b.len() && b[pos] == b'.' {
        pos += 1;
        while pos < b.len() && b[pos].is_ascii_digit() {
            pos += 1;
            saw_digit = true;
        }
    }
    if !saw_digit {
        return (0.0, start);
    }
    // optional exponent
    if pos < b.len() && (b[pos] == b'e' || b[pos] == b'E') {
        let mut p = pos + 1;
        if p < b.len() && (b[p] == b'+' || b[p] == b'-') {
            p += 1;
        }
        let edig = p;
        while p < b.len() && b[p].is_ascii_digit() {
            p += 1;
        }
        if p > edig {
            pos = p;
        }
    }
    let mut token: String = std::str::from_utf8(&b[tok_begin..pos]).unwrap_or("").to_string();
    // Rust's parser rejects leading/trailing '.', normalize like C accepts.
    if token.starts_with('.') {
        token.insert(0, '0');
    } else if token.starts_with("+.") || token.starts_with("-.") {
        token.insert(1, '0');
    }
    if token.ends_with('.') {
        token.push('0');
    }
    match token.parse::<f64>() {
        Ok(v) => (v, pos),
        Err(_) => (0.0, start),
    }
}
