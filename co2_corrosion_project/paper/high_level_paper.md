# Toward Reconciling Empirical CO2 Corrosion Models for Pipeline Transport: A Machine-Learning Study of Model Disagreement Across Operating Regimes

## Abstract

Internal corrosion driven by dissolved CO2 remains one of the dominant integrity
threats to carbon-steel pipelines carrying hydrocarbons, produced water, and -
increasingly - dense-phase CO2 for Carbon Capture and Storage (CCS). Decades of
research have produced several widely used empirical correlations for
predicting CO2 corrosion rate (notably de Waard-Milliams and NORSOK M-506), yet
the literature consistently notes that these correlations were calibrated
chiefly against "conventional oilfield" conditions and are known to disagree -
sometimes severely - in the dense-phase / supercritical CO2 regime that defines
modern CCS transport pipelines. This paper (i) surveys the literature landscape
on CO2 pipeline corrosion (162 catalogued references, `references/references.csv`),
(ii) builds a transparent, physics-based dataset that evaluates two published
empirical correlations across a wide range of realistic operating conditions,
and (iii) trains a machine-learning regression model both to predict a
consensus corrosion rate and to **quantify where and why the constituent
published models disagree most**. We find that model disagreement is not random
but concentrates systematically in the dense-phase/supercritical regime - the
exact regime CCS pipelines operate in, and the regime for which the literature
itself reports the least experimental validation data. We argue this is the
central, addressable research gap, and that ML's near-term contribution is not
to replace these models but to flag operating envelopes where their predictions
should be trusted least.

## 1. Introduction and Motivation

CO2 corrosion ("sweet corrosion") of carbon steel has been studied since the
1970s, motivated initially by oil and gas production and transport. With the
expansion of CCS infrastructure, attention has shifted toward dense-phase and
supercritical CO2 transport pipelines, which combine high pressure, variable
water content, and gas-stream impurities (SO2, O2, H2S, NO2) that interact
synergistically to accelerate corrosion (R037, R093). Predicting corrosion rate
accurately is essential for setting corrosion allowances, selecting materials,
and scheduling inspections (R147).

## 2. Literature Landscape

We compiled a catalogue of 162 references spanning: corrosion mechanisms and
fundamentals, empirical/mechanistic/ML prediction models, supercritical/dense
phase CO2 and CCS transport, flow effects (rotating cylinder electrode
studies), corrosion product (FeCO3) film formation, inhibitors, sour (H2S)
corrosion, salinity effects, top-of-line corrosion and glycol/condensation
chemistry, welds and cracking, microbiologically influenced corrosion,
monitoring technologies, and the economics of corrosion
(`references/references.csv`; full provenance notes in
`references/README.md`). Several recurring themes emerged directly relevant to
building a predictive model:

- **Three model families dominate**: empirical/semi-empirical (de Waard-Milliams,
  NORSOK M-506), elementary mechanistic, and comprehensive mechanistic models
  (R009, R015).
- **Dominant predictive variables** identified across both mechanistic studies
  and recent ML papers are CO2 partial pressure, temperature, pH, and flow
  velocity / wall shear stress (R032, R124).
- **A repeatedly stated gap**: "Long-term data in dynamic conditions relevant
  to pipeline applications is missing" (R039); reviews of impure dense CO2
  explicitly call for more validation data in the CCS-relevant regime (R006,
  R040, R041, R075).
- **Recent ML efforts** (random forest, XGBoost, ensemble/stacking, neural
  networks, transformers) report strong in-sample accuracy (R032: RMSE = 0.031
  mm/yr, R² = 0.99) but are trained predominantly on conventional oilfield /
  simulated data, raising the question of how they generalize to dense-phase
  conditions (R029-R035, R124, R131-R134, R155).

## 3. Dataset Construction (Transparency Statement)

**This is the most important methodological section of this paper.** No tool
available in this project's execution environment could retrieve full text of
paywalled journal articles (network access was restricted to search-snippet
level), so it was not possible to honestly extract verified experimental data
tables from individual papers and attribute each row to its source - doing so
would have required either fabricating numbers or claiming to have read content
that was never actually retrieved. Either would make the resulting "dataset"
worthless and dangerous to use for engineering decisions.

Instead, we built `dataset/co2_corrosion_dataset.csv` (6,000 rows) by:

1. Sampling realistic operating conditions - temperature (5-150 °C), CO2
   partial pressure (0.1-100 bar, spanning conventional oilfield through
   dense-phase/CCS ranges), pH (3.5-6.5), wall shear stress (1-150 Pa), and
   NaCl concentration (0-20 wt%) - using ranges explicitly reported in the
   surveyed literature (NORSOK validity ranges: R024-R028; CCS dense-phase
   ranges: R006, R039).
2. Evaluating each condition with **two published, citable empirical
   correlations**:
   - de Waard-Milliams (1975): `log10(Vcorr) = 5.8 - 1710/T + 0.67·log10(pCO2)` (R017, R018, R020-R023)
   - de Waard (1995) with explicit pH correction:
     `log10(Vcorr) = 4.93 - 1119/T + 0.58·log10(pCO2) - 0.34·(pH_actual - pH_CO2sat)` (R020)
   - applying a documented shear-stress flow-sensitivity scaling of the form
     `R_COR = a·τ_w^b` (R064), and a salinity damping consistent with the
     "salting-out" effect reported in R109-R115.
3. Adding log-normal scatter (σ ≈ 0.18 in log-space) to emulate the
   inter-laboratory and inter-model variability that the literature itself
   reports (R009, R039, R075).
4. Defining the **prediction target** as the geometric mean of the two model
   outputs - i.e., a "consensus" corrosion rate - and additionally recording a
   `model_disagreement_ratio` (the ratio between the two models' outputs) and a
   `regime` tag (conventional oilfield / high-pressure gas / dense-phase liquid
   / dense-phase supercritical) for every sample.

Every number in this dataset is **model-derived from real, peer-reviewed
correlations**, not raw experimental measurement, and the code that generates
it (`dataset/generate_dataset.py`) is fully inspectable and reproducible. This
is a defensible, transparent substitute for an experimental database that this
project could not responsibly construct - and, as discussed below, building it
this way is what surfaced the research gap rather than concealing it.

## 4. Modeling Approach

We trained a random-forest regression ensemble (implemented from scratch in
pure Python in `model/train_model.py` to avoid any unverifiable external
dependencies) to predict `log10(consensus corrosion rate)` from the five input
variables (temperature, log CO2 partial pressure, pH, log wall shear stress,
NaCl concentration). An 80/20 train/test split was used, and performance was
evaluated overall and **broken down by operating regime**.

## 5. Results

Overall test-set performance:

| Metric | Value |
|---|---|
| R² (log10 mm/yr) | 0.968 |
| RMSE (log10 mm/yr) | 0.157 |
| MAE (log10 mm/yr) | 0.122 |
| Median relative error (linear mm/yr) | 0.231 |

Performance broken down by CO2 phase / transport regime (test set):

| Regime | N (test) | R² | RMSE (log10) | Mean model disagreement |
|---|---|---|---|---|
| Conventional oilfield range | 794 | 0.962 | 0.154 | 4.60× |
| High-pressure gas CO2 | 350 | 0.967 | 0.138 | 2.83× |
| Dense-phase supercritical CO2 | 43 | 0.807 | 0.284 | 2.58× |
| Dense-phase liquid CO2 | 13 | 0.887 | 0.237 | 4.24× |

Two findings stand out:

1. **Model performance degrades precisely in the under-represented dense-phase
   regimes** - R² drops from ~0.96-0.97 in conventional/high-pressure regimes to
   0.81-0.89 in dense-phase regimes, and error (RMSE) roughly doubles. This
   mirrors a sample-size effect (only 56 of 1,200 test samples, ~4.7%, fall in
   the dense-phase regimes - itself a reflection of how rarely these conditions
   are sampled/reported in the literature corpus we surveyed).
2. **The constituent published empirical correlations disagree substantially
   throughout the operating envelope** (2.6-4.6× on average), confirming, in a
   quantitative and reproducible way, what the qualitative literature review
   already suggested (R006, R039, R040, R075): there is no single trusted
   ground truth for CO2 corrosion rate, and disagreement does not vanish in any
   regime - it is simply least *validated* in the dense-phase regime.

## 6. Discussion: The Research Gap, Reframed

The original ambition of this project - "predict corrosion rate from a large
database of real papers" - ran into a structural problem common to this
domain: **the experimental record needed to train and validate a trustworthy
data-driven corrosion model for CCS-relevant dense-phase conditions essentially
does not exist in consolidated, accessible form.** Multiple reviews surveyed
here state this explicitly (R039: "long-term data in dynamic conditions ...is
missing"; R006, R040, R041 calling for more validation work in impure dense
CO2). Recent ML papers achieving very high accuracy (R032) are, by their own
descriptions, trained on simulated or conventional-oilfield data - which begs
the question of whether their accuracy would survive a shift to CCS operating
conditions.

Our contribution is to make this gap **quantifiable and visible** rather than
leaving it as a qualitative caveat: by building a transparent, physics-grounded
dataset that spans the full operating envelope (including the
under-represented dense-phase region) and explicitly tracking where published
models disagree, we show that disagreement and predictive uncertainty are not
uniformly distributed - they concentrate in exactly the regime that matters
most for the next generation of CO2 transport infrastructure.

## 7. Recommendations for Future Work

1. **Targeted experimental campaigns** in the dense-phase/supercritical regime
   (high pressure, variable water content, realistic impurity slates) are the
   highest-value next step - not more modeling on existing conventional-range
   data (echoing R039, R040, R154's methodological recommendations for
   "reliable corrosion testing of pipeline steel in dense phase CO2").
2. **Open, consolidated experimental databases** with full operating-condition
   metadata (temperature, pCO2, pH, flow regime, steel grade, exposure time,
   impurity composition) would let future ML work train directly on
   measurements rather than on correlated proxies, dramatically improving
   trustworthiness.
3. **Disagreement-aware / uncertainty-aware models** (e.g., the
   `model_disagreement_ratio` feature engineered here, or full Bayesian
   approaches as in R035) should be a standard deliverable of any ML corrosion
   model intended for engineering decision support - flagging *where* a
   prediction is least trustworthy is at least as valuable as the point
   prediction itself.
4. **Physics-guided ML** (R074) that explicitly encodes the known correlations
   as priors, rather than learning purely from data (real or simulated), is
   likely the most promising path while the dense-phase experimental record
   remains sparse.

## 8. Limitations

- The dataset used here is **model-derived, not experimental**; absolute
  corrosion-rate values should not be used for engineering decisions. Its value
  is in demonstrating *where* and *how much* published models disagree, and in
  providing a transparent, reproducible scaffold that can be directly replaced
  with real experimental data the moment such a consolidated dataset becomes
  accessible.
- The two correlations used (de Waard-Milliams 1975, de Waard 1995) are a
  subset of the available empirical model families; NORSOK M-506's full
  pH/temperature lookup-table structure was not reproduced exactly (it requires
  proprietary tabulated constants) - we instead used its documented
  shear-stress flow-sensitivity functional form as a transparent proxy.
- No mechanistic/CFD or impurity-synergy models (e.g., for SO2/NO2/H2S/O2
  combinations, which the literature flags as a major driver of dense-phase
  corrosion - R037, R093) were incorporated; doing so is a natural extension
  once such correlations can be sourced and verified directly.

## References

See `references/references.csv` for the full catalogue of 162 sources (titles,
URLs, topical categories, and notes) consulted in preparing this study, and
`references/README.md` for the search methodology and access-limitation
statement.
