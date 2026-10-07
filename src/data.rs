//! Deterministic window loader over in-memory documents (§5.1 boundary policy, §5.2 loader state).
//!
//! Each document becomes `BOS + bytes + EOS`, cut into `T+1`-token windows at stride `T`
//! (targets continuous, windows never cross documents), right-padded with PAD.
//! Window order is a seeded Fisher-Yates shuffle per epoch, so a saved
//! `{epoch, position, shuffle_seed}` resumes to the identical next window.

use crate::autodiff::TrainError;
use crate::rng::Rng;
use crate::tokenizer::{BOS, EOS, PAD};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoaderState {
    pub epoch: u64,
    pub position: u64,
    pub shuffle_seed: u64,
}

#[derive(Debug, Clone)]
pub struct WindowLoader {
    windows: Vec<Vec<u32>>,
    order: Vec<usize>,
    state: LoaderState,
}

fn shuffled(len: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut rng = Rng::new(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    for i in (1..len).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

impl WindowLoader {
    /// `docs` hold raw token ids (bytes); boundary tokens are added here.
    pub fn new(docs: &[Vec<u32>], t: usize, seed: u64) -> Result<Self, TrainError> {
        if t == 0 {
            return Err(TrainError::InvalidArgument("window length must be >= 1"));
        }
        let mut windows = Vec::new();
        for doc in docs {
            let mut tok = Vec::with_capacity(doc.len() + 2);
            tok.push(BOS);
            tok.extend_from_slice(doc);
            tok.push(EOS);
            let n = tok.len();
            let mut s = 0;
            while s + 1 < n {
                let mut w: Vec<u32> = tok[s..(s + t + 1).min(n)].to_vec();
                w.resize(t + 1, PAD);
                windows.push(w);
                s += t;
            }
        }
        if windows.is_empty() {
            return Err(TrainError::Data("no windows"));
        }
        let order = shuffled(windows.len(), seed, 0);
        Ok(WindowLoader { windows, order, state: LoaderState { epoch: 0, position: 0, shuffle_seed: seed } })
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
    pub fn state(&self) -> LoaderState {
        self.state
    }

    pub fn restore(&mut self, s: LoaderState) -> Result<(), TrainError> {
        if s.position > self.windows.len() as u64 {
            return Err(TrainError::Data("loader position beyond epoch"));
        }
        self.order = shuffled(self.windows.len(), s.shuffle_seed, s.epoch);
        self.state = s;
        Ok(())
    }

    /// Next window (cursor advances even if the caller later skips the step).
    pub fn next_window(&mut self) -> Vec<u32> {
        if self.state.position as usize >= self.windows.len() {
            self.state.epoch += 1;
            self.state.position = 0;
            self.order = shuffled(self.windows.len(), self.state.shuffle_seed, self.state.epoch);
        }
        let w = self.windows[self.order[self.state.position as usize]].clone();
        self.state.position += 1;
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Vec<u32>> {
        vec![(0..20).collect(), (30..34).collect(), vec![]]
    }

    #[test]
    fn windows_respect_boundaries_and_padding() {
        let l = WindowLoader::new(&docs(), 8, 1).unwrap();
        // doc0: 22 tokens -> starts 0,8,16 (3); doc1: 6 tokens -> 1; doc2: [BOS,EOS] -> 1
        assert_eq!(l.len(), 5);
        assert!(l.windows.iter().all(|w| w.len() == 9));
        assert!(l.windows.iter().any(|w| w[0] == BOS && w[1] == 30));
        assert!(l.windows.iter().all(|w| w.iter().filter(|&&t| t == BOS).count() <= 1));
    }

    #[test]
    fn restore_resumes_identical_stream_across_epochs() {
        let mut a = WindowLoader::new(&docs(), 8, 42).unwrap();
        for _ in 0..7 {
            a.next_window();
        }
        let st = a.state();
        let tail_a: Vec<_> = (0..9).map(|_| a.next_window()).collect();
        let mut b = WindowLoader::new(&docs(), 8, 42).unwrap();
        b.restore(st).unwrap();
        let tail_b: Vec<_> = (0..9).map(|_| b.next_window()).collect();
        assert_eq!(tail_a, tail_b);
        assert!(a.state().epoch >= 2);
        assert!(b.restore(LoaderState { epoch: 0, position: 999, shuffle_seed: 42 }).is_err());
    }
}
