//! signalox — pure-Rust bacterial signal-peptide / cleavage / transmembrane predictor.
//!
//! A compact convolutional model with a 2-state linear-chain CRF cleavage decoder
//! and a per-residue transmembrane head. The architecture is **inspired by DeepSig**
//! (Savojardo, Martelli, Fariselli & Casadio, *Bioinformatics* 2018) — three cascaded
//! convolutional stages that read the protein N-terminus, a classification head that
//! discriminates signal peptides from N-terminal transmembrane segments, and a CRF
//! that locates the cleavage site. It is **not** DeepSig: this is an independent
//! clean-room implementation with our own weights, trained from scratch on public
//! UniProt/SwissProt data (CC-BY 4.0). No DeepSig source or model weights are used.
//!
//! Inference is self-contained and database-free: the trained weights are embedded in
//! the binary ([`weights.json`]) and every operation (one-hot encode, `conv1d`, ReLU,
//! global-average pool, linear heads, CRF Viterbi) is hand-rolled with no ML runtime.
//!
//! ```
//! let m = signalox::Model::embedded();
//! let p = m.predict(b"MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKLGWSQYH");
//! println!("{:?} cleavage={:?}", p.sp_type, p.cleavage);
//! ```

use serde::Deserialize;
use std::sync::OnceLock;

/// Signal-peptide class predicted for a protein's N-terminus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpType {
    /// No signal peptide (soluble / cytoplasmic).
    NoSp,
    /// No signal peptide, but an N-terminal transmembrane helix.
    Tm,
    /// Sec-secreted, signal peptidase I (classical Sec/SPI).
    Sec,
    /// Lipoprotein, signal peptidase II (Sec/SPII).
    Lipo,
    /// Tat-secreted, twin-arginine, signal peptidase I (Tat/SPI).
    Tat,
}

impl SpType {
    fn from_index(i: usize) -> SpType {
        match i {
            0 => SpType::NoSp,
            1 => SpType::Tm,
            2 => SpType::Sec,
            3 => SpType::Lipo,
            4 => SpType::Tat,
            _ => SpType::NoSp,
        }
    }
    /// True for the three signal-peptide-bearing classes (Sec/Lipo/Tat).
    pub fn is_signal(self) -> bool {
        matches!(self, SpType::Sec | SpType::Lipo | SpType::Tat)
    }
    /// Lower-case tag matching bactars' signal-peptide vocabulary.
    pub fn tag(self) -> &'static str {
        match self {
            SpType::NoSp => "none",
            SpType::Tm => "transmembrane",
            SpType::Sec => "sec",
            SpType::Lipo => "lipoprotein",
            SpType::Tat => "tat",
        }
    }
}

/// A prediction for one protein.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Most probable class.
    pub sp_type: SpType,
    /// Softmax probability of `sp_type`.
    pub sp_prob: f32,
    /// Full softmax over {NoSp, Tm, Sec, Lipo, Tat}.
    pub type_probs: [f32; 5],
    /// 1-based cleavage position (residue after which the peptidase cuts), for
    /// signal-peptide classes only; `None` otherwise or when the CRF finds no signal.
    pub cleavage: Option<usize>,
    /// Predicted transmembrane segments as 1-based inclusive `(start, end)` ranges
    /// within the scored N-terminal window.
    pub tm_segments: Vec<(usize, usize)>,
}

/// Raw weight tensors as exported by the training script (`train.py`).
#[derive(Deserialize)]
struct Weights {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "C")]
    c: usize,
    k: usize,
    nsym: usize,
    aa_order: String,
    conv1_w: Vec<Vec<Vec<f32>>>, // [C][nsym][k]
    conv1_b: Vec<f32>,
    conv2_w: Vec<Vec<Vec<f32>>>, // [C][C][k]
    conv2_b: Vec<f32>,
    conv3_w: Vec<Vec<Vec<f32>>>, // [C][C][k]
    conv3_b: Vec<f32>,
    type_w: Vec<Vec<f32>>, // [5][C]
    type_b: Vec<f32>,
    emit_w: Vec<Vec<f32>>, // [2][C]
    emit_b: Vec<f32>,
    tm_w: Vec<Vec<f32>>, // [1][C]
    tm_b: Vec<f32>,
    trans: Vec<Vec<f32>>, // [2][2]
    start: Vec<f32>,      // [2]
    stop: Vec<f32>,       // [2]
}

/// A loaded signalox model.
pub struct Model {
    w: Weights,
    /// aa byte -> channel index (0..nsym-1), or `nsym` sentinel for pad/unknown.
    aa_to_chan: [usize; 256],
}

/// Trained weights, gzip-compressed and embedded at compile time (kept under the
/// crates.io size limit; decompressed once in [`Model::embedded`]).
const EMBEDDED_WEIGHTS_GZ: &[u8] = include_bytes!("../weights.json.gz");

impl Model {
    /// The model backed by the embedded trained weights (parsed once, cached).
    pub fn embedded() -> &'static Model {
        static M: OnceLock<Model> = OnceLock::new();
        M.get_or_init(|| {
            use std::io::Read;
            let mut s = String::new();
            flate2::read::GzDecoder::new(EMBEDDED_WEIGHTS_GZ)
                .read_to_string(&mut s)
                .expect("decompress embedded weights");
            Model::from_json(&s).expect("embedded weights parse")
        })
    }

    /// Load a model from a `weights.json` string.
    pub fn from_json(s: &str) -> Result<Model, String> {
        let w: Weights = serde_json::from_str(s).map_err(|e| e.to_string())?;
        // In training, amino acid index 1..=20 map to one-hot channels 0..=19 (the
        // pad channel 0 is dropped), and the "other/X" index nsym maps to channel
        // nsym-1. Rebuild that byte->channel table from aa_order.
        let mut aa_to_chan = [w.nsym; 256]; // default -> unknown (last channel)
        for (i, c) in w.aa_order.bytes().enumerate() {
            aa_to_chan[c as usize] = i; // channel i (0-based)
            aa_to_chan[c.to_ascii_lowercase() as usize] = i;
        }
        // "other" (index 21 in training = channel nsym-1 = 20)
        Ok(Model { aa_to_chan, w })
    }

    /// Predict for many sequences in parallel (rayon). Equivalent to mapping
    /// [`Model::predict`] over `seqs` — identical results, one [`Prediction`] per input in
    /// order. Use for whole-genome / large batches; forward passes are independent.
    pub fn predict_many(&self, seqs: &[&[u8]]) -> Vec<Prediction> {
        use rayon::prelude::*;
        seqs.par_iter().map(|s| self.predict(s)).collect()
    }

    /// Predict signal-peptide class, cleavage site and TM segments for `aa`.
    pub fn predict(&self, aa: &[u8]) -> Prediction {
        let l = self.w.l;
        let nsym = self.w.nsym;
        // one-hot [nsym][L]; positions past the sequence stay all-zero (pad).
        let mut oh = vec![0.0f32; nsym * l];
        for t in 0..l {
            if t >= aa.len() {
                break;
            }
            let ch = self.aa_to_chan[aa[t] as usize];
            if ch < nsym {
                oh[ch * l + t] = 1.0;
            } else {
                // unknown/other -> last channel (matches training index nsym)
                oh[(nsym - 1) * l + t] = 1.0;
            }
        }
        // conv stack (ReLU) -> h [C][L]
        let h1 = conv1d_relu(&oh, nsym, l, &self.w.conv1_w, &self.w.conv1_b, self.w.c, self.w.k);
        let h2 = conv1d_relu(&h1, self.w.c, l, &self.w.conv2_w, &self.w.conv2_b, self.w.c, self.w.k);
        let h = conv1d_relu(&h2, self.w.c, l, &self.w.conv3_w, &self.w.conv3_b, self.w.c, self.w.k);
        let c = self.w.c;

        // type head: global average pool over L, then linear -> 5 logits
        let mut pooled = vec![0.0f32; c];
        for ci in 0..c {
            let mut s = 0.0;
            for t in 0..l {
                s += h[ci * l + t];
            }
            pooled[ci] = s / l as f32;
        }
        let mut type_logits = [0.0f32; 5];
        for k in 0..5 {
            let mut s = self.w.type_b[k];
            for ci in 0..c {
                s += self.w.type_w[k][ci] * pooled[ci];
            }
            type_logits[k] = s;
        }
        let type_probs = softmax5(&type_logits);
        let mut best = 0usize;
        for k in 1..5 {
            if type_probs[k] > type_probs[best] {
                best = k;
            }
        }
        let sp_type = SpType::from_index(best);

        // emission head [L][2] + CRF Viterbi -> cleavage
        let cleavage = if sp_type.is_signal() {
            let emis = emissions(&h, c, l, &self.w.emit_w, &self.w.emit_b);
            let tags = crf_viterbi(&emis, l, &self.w.trans, &self.w.start, &self.w.stop);
            // cleavage = last S (state 0) position, 1-based
            let mut cl = 0usize;
            for t in 0..l {
                if tags[t] == 0 {
                    cl = t + 1;
                }
            }
            if cl > 0 {
                Some(cl)
            } else {
                None
            }
        } else {
            None
        };

        // TM head: per-position sigmoid > 0.5 -> segments
        let mut tm_segments = Vec::new();
        let mut run_start: Option<usize> = None;
        for t in 0..l {
            let mut s = self.w.tm_b[0];
            for ci in 0..c {
                s += self.w.tm_w[0][ci] * h[ci * l + t];
            }
            let on = s > 0.0; // sigmoid(s) > 0.5  <=>  s > 0
            match (on, run_start) {
                (true, None) => run_start = Some(t),
                (false, Some(st)) => {
                    tm_segments.push((st + 1, t));
                    run_start = None;
                }
                _ => {}
            }
        }
        if let Some(st) = run_start {
            tm_segments.push((st + 1, l));
        }

        Prediction {
            sp_type,
            sp_prob: type_probs[best],
            type_probs,
            cleavage,
            tm_segments,
        }
    }
}

/// `conv1d` with `same` padding (k/2) followed by ReLU. `inp` is `[cin][L]` row-major,
/// `w` is `[cout][cin][k]`; returns `[cout][L]` row-major.
fn conv1d_relu(
    inp: &[f32],
    cin: usize,
    l: usize,
    w: &[Vec<Vec<f32>>],
    b: &[f32],
    cout: usize,
    k: usize,
) -> Vec<f32> {
    let pad = k / 2;
    let mut out = vec![0.0f32; cout * l];
    for co in 0..cout {
        let wco = &w[co];
        let bias = b[co];
        for t in 0..l {
            let mut acc = bias;
            for ci in 0..cin {
                let wci = &wco[ci];
                let base = ci * l;
                for dk in 0..k {
                    // input index for kernel tap dk at output position t
                    let ii = t as isize + dk as isize - pad as isize;
                    if ii >= 0 && (ii as usize) < l {
                        acc += wci[dk] * inp[base + ii as usize];
                    }
                }
            }
            out[co * l + t] = if acc > 0.0 { acc } else { 0.0 };
        }
    }
    out
}

/// Per-position emission scores `[L][2]` from features `h` `[C][L]` and linear `emit`.
fn emissions(h: &[f32], c: usize, l: usize, ew: &[Vec<f32>], eb: &[f32]) -> Vec<[f32; 2]> {
    let mut emis = vec![[0.0f32; 2]; l];
    for t in 0..l {
        for s in 0..2 {
            let mut acc = eb[s];
            for ci in 0..c {
                acc += ew[s][ci] * h[ci * l + t];
            }
            emis[t][s] = acc;
        }
    }
    emis
}

/// 2-state linear-chain CRF Viterbi decode. States: 0 = signal (S), 1 = mature (M).
/// Returns the best state per position.
fn crf_viterbi(emis: &[[f32; 2]], l: usize, trans: &[Vec<f32>], start: &[f32], stop: &[f32]) -> Vec<usize> {
    let mut v = [start[0] + emis[0][0], start[1] + emis[0][1]];
    let mut bp = vec![[0usize; 2]; l];
    for t in 1..l {
        let mut nv = [f32::NEG_INFINITY; 2];
        for j in 0..2 {
            for i in 0..2 {
                let sc = v[i] + trans[i][j];
                if sc > nv[j] {
                    nv[j] = sc;
                    bp[t][j] = i;
                }
            }
            nv[j] += emis[t][j];
        }
        v = nv;
    }
    let mut best = if v[0] + stop[0] >= v[1] + stop[1] { 0 } else { 1 };
    let mut tags = vec![1usize; l];
    tags[l - 1] = best;
    for t in (1..l).rev() {
        best = bp[t][best];
        tags[t - 1] = best;
    }
    tags
}

fn softmax5(x: &[f32; 5]) -> [f32; 5] {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut e = [0.0f32; 5];
    let mut s = 0.0;
    for i in 0..5 {
        e[i] = (x[i] - m).exp();
        s += e[i];
    }
    for i in 0..5 {
        e[i] /= s;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_loads_and_predicts() {
        let m = Model::embedded();
        // A classic Sec signal peptide (E. coli OmpA-like N-terminus).
        let p = m.predict(b"MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKLGWSQYH");
        // just structural sanity: probabilities sum ~1, class in range
        let s: f32 = p.type_probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "probs sum {s}");
        assert!(p.sp_prob > 0.0);
    }
}
