# References Catalogue — CO2 Pipeline Corrosion

`references.csv` contains 162 catalogued sources on CO2 corrosion of carbon-steel
pipelines (mechanisms, empirical/mechanistic/ML prediction models, supercritical
and dense-phase CO2 / CCS transport, flow effects, corrosion-product films,
inhibitors, sour corrosion, salinity effects, top-of-line corrosion, welds and
cracking, microbiologically influenced corrosion, monitoring technology, and
corrosion economics).

## Methodology

References were collected via web search (titles, authors, journals, DOIs/URLs,
and short descriptive snippets returned by search results) using ~20 topic
queries spanning the subfields above. Each entry records: an ID, title, source
URL/DOI link, a topical category, and a brief note on its relevance.

## IMPORTANT — Access limitations (read before using this catalogue)

The execution environment used to build this catalogue could not fetch and
read full paper text — direct HTTP/API access (CrossRef, OpenAlex,
Semantic Scholar, ScienceDirect, PMC, even Wikipedia) returned `403 Forbidden`
for every URL tried, including non-paywalled ones. Only short search-result
snippets were retrievable. Consequently:

- This catalogue should be treated as a **literature map / starting
  bibliography**, not a verified annotated bibliography — entries have not been
  individually read in full, and the "notes" column reflects search-snippet
  context, not full-text review.
- **No PDFs or full texts are stored in this repository** — only reference
  links (DOIs/URLs), to respect copyright and avoid repository bloat. Several
  entries are explicitly open-access (PMC, MDPI, open ScienceDirect articles,
  open theses) and are flagged as such in the notes where identified.
- A handful of links point to aggregator/secondary sources (ResearchGate,
  Academia.edu, Scribd, patent-image servers) where that was the most relevant
  result the search returned; where possible, the canonical
  publisher/DOI link is also listed as a separate entry.

## How this catalogue was used

The topical clustering of these references directly informed:
1. The selection of the two empirical correlations (de Waard-Milliams,
   de Waard 1995 / NORSOK-family flow scaling) used to build the physics-based
   dataset in `../dataset/`.
2. The realistic operating-condition ranges sampled when generating that
   dataset (temperature, CO2 partial pressure, pH, shear stress, salinity).
3. The framing of the research gap addressed in `../paper/high_level_paper.md`
   — namely, that the literature itself repeatedly identifies a shortage of
   experimental validation data for dense-phase / supercritical CO2 transport
   conditions relevant to CCS pipelines.
