# Cross-corpus validity in automated English writing assessment

This project evaluates whether a proficiency signal learned from CEFR-labelled
Write & Improve essays transfers to scores in the public FCE corpus, and whether
out-of-sample scoring error varies across first-language groups.

The package reproduces the aggregate tables and figures reported in the study.
It intentionally excludes manuscript files, licensed learner text, and
candidate-level prediction exports.

## Data

Download release 2.1 of W&I+LOCNESS and FCE from the official Cambridge
BEA-2019 data release and extract the archives under `dataset_raw/`. The corpora are provided for
non-commercial research and educational use; their licence files govern reuse.
The experiment does not redistribute corpus text.

Expected directories:

```text
dataset_raw/extracted_wilocness/wi+locness/json/
dataset_raw/extracted_fce/fce/json/
```

## Reproduce

Use Python 3.9 or later in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_experiments.py
```

All aggregate tables, diagnostic summaries, and figures are regenerated under
`results/`. Candidate-level prediction exports are created locally for audit
purposes but are ignored by version control. The fixed random seed is recorded
in both the script and the machine-readable summary.

## Analytical design

The workflow performs a writer-grouped W&I split, trains surface, TF–IDF, and
hybrid CEFR models, calibrates the source proficiency signal to the FCE scale,
compares it with target-trained scoring models, and audits signed and absolute
out-of-fold error across sufficiently represented L1 groups. The official FCE
test partition is used once for the final model comparison.

## Outputs retained in the archive

- Aggregate corpus profiles and performance tables.
- Paired bootstrap comparisons and conditional L1 audit summaries.
- Two publication-resolution figures in a consistent visual style.
- The complete analysis program and pinned Python dependencies.

The code is released under the MIT License. The source corpora remain subject
to their original Cambridge research licences.
