//! signalox CLI — predict signal peptide / cleavage / TM topology for protein FASTA.
//!
//! Usage:
//!   signalox proteins.faa            # tabular output to stdout
//!   signalox < proteins.faa          # read from stdin
//!
//! Columns: id  type  prob  cleavage  tm_segments

use signalox::Model;
use std::io::{self, Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = if args.len() > 1 && args[1] != "-" {
        match std::fs::read_to_string(&args[1]) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("signalox: cannot read {}: {e}", args[1]);
                std::process::exit(1);
            }
        }
    } else {
        let mut s = String::new();
        if io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("signalox: failed to read stdin");
            std::process::exit(1);
        }
        s
    };

    let model = Model::embedded();
    let out = io::stdout();
    let mut w = out.lock();
    let _ = writeln!(w, "#id\ttype\tprob\tcleavage\ttm_segments");
    let recs = fasta_iter(&input);
    let seqs: Vec<&[u8]> = recs.iter().map(|(_, s)| s.as_bytes()).collect();
    let preds = model.predict_many(&seqs); // parallel across sequences
    for ((id, _seq), p) in recs.iter().zip(preds.iter()) {
        let cleave = p.cleavage.map(|c| c.to_string()).unwrap_or_else(|| "-".into());
        let tms = if p.tm_segments.is_empty() {
            "-".to_string()
        } else {
            p.tm_segments
                .iter()
                .map(|(a, b)| format!("{a}-{b}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let _ = writeln!(
            w,
            "{id}\t{}\t{:.3}\t{cleave}\t{tms}",
            p.sp_type.tag(),
            p.sp_prob
        );
    }
}

/// Minimal FASTA parser: yields `(id, sequence)`. `id` is the first whitespace-
/// delimited token of the header; sequence lines are concatenated (whitespace removed).
fn fasta_iter(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut id: Option<String> = None;
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(prev) = id.take() {
                out.push((prev, std::mem::take(&mut seq)));
            }
            id = Some(rest.split_whitespace().next().unwrap_or("").to_string());
        } else {
            seq.extend(line.split_whitespace());
        }
    }
    if let Some(prev) = id.take() {
        out.push((prev, seq));
    }
    out
}
