# CO2 Corrosion in Transport Pipelines — Literature Survey, Physics-Based Dataset, and ML Study

This project studies corrosion-rate prediction for CO2 transport pipelines
(relevant to oil & gas production/transport and to CCS dense-phase CO2
infrastructure). It contains:

```
co2_corrosion_project/
├── references/
│   ├── references.csv      162 catalogued literature sources (titles, links, categories, notes)
│   └── README.md           methodology + IMPORTANT access-limitation statement — read first
├── dataset/
│   ├── generate_dataset.py builds a transparent, physics-based dataset from
│   │                       published empirical CO2-corrosion correlations
│   └── co2_corrosion_dataset.csv   6,000-row generated dataset
├── model/
│   └── train_model.py      dependency-free random-forest regressor + regime analysis
└── paper/
    └── high_level_paper.md the write-up: literature review, methodology,
                            results, and the research gap this project surfaces
```

## tl;dr — what this is and isn't

- **It is**: a literature map (162 references with links), a fully transparent
  and reproducible dataset built by sampling realistic pipeline operating
  conditions and evaluating them with two real, citable, peer-reviewed
  empirical corrosion correlations (de Waard-Milliams 1975, de Waard 1995),
  a from-scratch ML model trained on that dataset, and a paper that uses the
  results to **quantify a real research gap**: published empirical models
  disagree most — and are validated least — in the dense-phase/supercritical
  CO2 regime that defines modern CCS transport pipelines.
- **It is not**: a database of raw experimental measurements scraped from
  paywalled journals. The execution environment used to build this had no
  working full-text fetch capability (every URL, including non-paywalled ones,
  returned `403 Forbidden`) — see `references/README.md` for the full
  transparency statement. Fabricating "real experimental data" under those
  constraints would have produced a worthless and potentially dangerous
  dataset; this project instead documents that limitation honestly and builds
  something defensible around it.

## Reproducing the results

```bash
cd dataset && python3 generate_dataset.py     # regenerates co2_corrosion_dataset.csv
cd ../model && python3 train_model.py         # trains the model, prints regime-wise metrics
```

No external dependencies (no numpy/pandas/sklearn) are required — everything
runs with the Python 3 standard library, so the full pipeline is auditable
end-to-end.

## Start here

Read `paper/high_level_paper.md` for the full write-up, including the
literature review, the dataset/model methodology (with a dedicated
transparency section explaining exactly how and why it was built this way),
results broken down by CO2 phase regime, and recommendations for future work.
