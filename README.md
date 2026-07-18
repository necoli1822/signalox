# signalox

Pure-Rust bacterial **signal-peptide**, **cleavage-site** and **transmembrane-topology**
predictor. A compact convolutional network with a 2-state linear-chain CRF cleavage
decoder and a per-residue transmembrane head — self-contained, database-free, no ML
runtime, no Python at inference time.

```rust
let m = signalox::Model::embedded();
let p = m.predict(b"MKKTAIAIAVALAGFATVAQAAPKDNTWYTGAKLGWSQYH");
println!("{:?}  prob={:.2}  cleavage={:?}  tm={:?}",
         p.sp_type, p.sp_prob, p.cleavage, p.tm_segments);
```

CLI:

```
signalox proteins.faa
#id       type   prob   cleavage  tm_segments
sp|P0A910 sec    0.98   21        -
```

## What it predicts

For each protein N-terminus, one of five classes:

| class | meaning |
|-------|---------|
| `none` | no signal peptide (soluble / cytoplasmic) |
| `transmembrane` | N-terminal transmembrane helix, no cleaved signal |
| `sec` | Sec-secreted, signal peptidase I (Sec/SPI) |
| `lipoprotein` | lipoprotein, signal peptidase II (Sec/SPII) |
| `tat` | Tat-secreted, twin-arginine (Tat/SPI) |

plus the **cleavage position** (CRF-decoded) for the signal-peptide classes and the
predicted **transmembrane segments**.

## Model & provenance

The architecture is **inspired by DeepSig** (Savojardo, Martelli, Fariselli & Casadio,
*Bioinformatics* 2018): cascaded convolutional stages read the N-terminus, a
classification head discriminates signal peptides from N-terminal transmembrane
segments, and a CRF locates the cleavage site.

This is **not** DeepSig and is not affiliated with or derived from the original DeepSig
software. It is an independent clean-room implementation with its own code and its own
weights, **trained from scratch on public UniProt/SwissProt data** (reviewed bacterial
entries, CC-BY 4.0). No DeepSig source code or trained weights were used.

The trained weights are embedded in the crate (`weights.json`); inference is fully
pure-Rust (`conv1d` / ReLU / global-average pool / linear heads / CRF Viterbi), with no
external database and no neural-network runtime dependency.

## Changelog

### 0.1.0
- Initial release: distilled/trained pure-Rust model, gzip-embedded weights, PyTorch-parity verified.

## License

MIT OR Apache-2.0. Training data derived from UniProt (CC-BY 4.0).
