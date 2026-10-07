//! ChaCha20-based deterministic PRNG with Box-Muller normals (§3.5).
//! Method recorded for reproducibility: ChaCha20 (20 rounds), key = SplitMix64(seed) x4, nonce = 0.

pub struct Rng {
    state: [u32; 16],
    buf: [u32; 16],
    idx: usize,
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn qr(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u32; 16];
        state[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
        for i in 0..4 {
            let v = splitmix(&mut sm);
            state[4 + 2 * i] = v as u32;
            state[5 + 2 * i] = (v >> 32) as u32;
        }
        Rng { state, buf: [0; 16], idx: 16 }
    }

    pub fn dump_state(&self) -> ([u32; 16], [u32; 16], usize) {
        (self.state, self.buf, self.idx)
    }

    pub fn from_state(state: [u32; 16], buf: [u32; 16], idx: usize) -> Self {
        Rng { state, buf, idx }
    }

    fn refill(&mut self) {
        let mut w = self.state;
        for _ in 0..10 {
            qr(&mut w, 0, 4, 8, 12);
            qr(&mut w, 1, 5, 9, 13);
            qr(&mut w, 2, 6, 10, 14);
            qr(&mut w, 3, 7, 11, 15);
            qr(&mut w, 0, 5, 10, 15);
            qr(&mut w, 1, 6, 11, 12);
            qr(&mut w, 2, 7, 8, 13);
            qr(&mut w, 3, 4, 9, 14);
        }
        for ((slot, &w_elem), &st_elem) in self.buf.iter_mut().zip(&w).zip(&self.state) {
            *slot = w_elem.wrapping_add(st_elem);
        }
        // 64-bit block counter in words 12..14
        let (lo, carry) = self.state[12].overflowing_add(1);
        self.state[12] = lo;
        if carry {
            self.state[13] = self.state[13].wrapping_add(1);
        }
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 16 {
            self.refill();
        }
        let v = self.buf[self.idx];
        self.idx += 1;
        v
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Standard normal via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = 1.0 - self.next_f32(); // (0, 1]
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let a: Vec<u32> = { let mut r = Rng::new(1); (0..40).map(|_| r.next_u32()).collect() };
        let b: Vec<u32> = { let mut r = Rng::new(1); (0..40).map(|_| r.next_u32()).collect() };
        let c: Vec<u32> = { let mut r = Rng::new(2); (0..40).map(|_| r.next_u32()).collect() };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn normal_statistics_reasonable() {
        let mut r = Rng::new(7);
        let n = 20000;
        let xs: Vec<f32> = (0..n).map(|_| r.next_normal()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.08, "mean={mean} var={var}");
        assert!(xs.iter().all(|x| x.is_finite()));
    }
}
