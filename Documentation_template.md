# ML Challenge 2026: Business Entity Resolution

Team Name: To be supplied
Team Members: To be supplied
Submission Date: Not yet submitted

## 1. Executive Summary

An offline pipeline retrieves plausible business matches using independent name,
address, and character-trigram searches. A LightGBM classifier combines 64 pairwise
features; validation selects its threshold using the exact macro F0.5 objective.
Implementation is complete for initial experimentation; training has not been run.

## 2. Methodology

### 2.1 Problem Analysis

Training contains 2,206,821 Source 1 records, 5,034,616 Source 2 records, and
5,285,603 Source 3 records. Ground truth contains 7,638,365 positive pairs and
123,247 singletons. Names are highly ambiguous and may change scripts; addresses
are noisy and sometimes absent. Training countries are US and India; test also
contains France. Full counts and explicitly sampled comparisons are retained under
`analysis/reports/` in the working project, outside the required submission package.

### 2.2 Solution Strategy

Approach: blocking plus supervised pair classification. Source 1 defines the
reference groups; multiple matches and empty predictions are supported. The pipeline
does not infer real-world information from external sources or numeric record IDs.

## 3. Candidate Generation

SQLite FTS5 retrieves by name, address, joint evidence, and name character trigrams.
Exact normalized name/address routes supplement these results. Reciprocal-rank
fusion produces the final candidates scored by the model. Country labels are open
strings. Default retrieval limits are unvalidated starting values.

Candidate pairs generated: not measured.
Blocking recall and macro F0.5 ceiling: not measured.
`candidate_pairs.tsv` is generated from the exact inference candidate list.

## 4. Matching Model

Model: LightGBM binary classifier, MIT-licensed library.
The Ubuntu launcher supports native CUDA training on the NVIDIA A30, with an
explicit GPU preflight and separate run/environment. CUDA starts with 63 histogram
bins; CPU defaults to 127. This is a configurable implementation choice, not a
validated optimal setting. GPU throughput/accuracy have not yet been measured.
Features: native/transliterated names and addresses, normalized edit/token/trigram
similarities, legal suffix normalization, rarity-weighted overlap, missingness,
numeric/range agreement, possible postal-code conflicts, and retrieval evidence.

Training uses retrieved positive pairs, hard negatives and sampled remaining
negatives. Holdout-owned targets are excluded from training candidates. Every
reference contributes equal total pair weight. An 80/10/10 reference split separates
training, model/threshold selection, and audit. Unlabeled retrieval corpus statistics
use all target records. No label-based candidate injection occurs.

Threshold selection: exact tied-score sweep maximizing macro F0.5 on tuning data.
Early stopping: macro F0.5 over a configurable threshold grid. Audit data is used
only after selection. Full-data refitting is not automatic.

## 5. Results and Error Analysis

Validation macro F0.5: not measured; no training performed.
Audit macro F0.5: not measured.
Measured runtime/memory: not measured at full scale.
Anticipated errors to investigate: same-name businesses in different locations,
cross-script names, missing addresses, and unseen-country generalization.
These are hypotheses from EDA, not measured model errors.

## 6. Conclusion

The implementation is ready for dependency setup and staged experiments. Neither
accuracy nor optimal hyperparameters have been established. This document must be
updated with actual results and team details before final submission.

## Appendix A. Code Artefacts

See `code/business_entity_resolution/README.md` for pinned dependencies, stage
commands, exact scoring rules, leakage controls, and resource considerations.
Entry point: `python -m ber.cli`. Training only runs via the explicit `train` command.
For Ubuntu automation, the root `run_training.sh` installs dependencies and invokes
all training stages through `ber.server`, including the explicit training command.
Preparation and completed model trials have restart checkpoints. Terminal logs,
input/code fingerprints, and success/failure records are saved for reproducibility.

## Appendix B. Additional Results

Pending execution. Synthetic tests cover scoring, threshold ties, singleton/missing
candidates, split isolation, Unicode handling, output coverage, and invalid IDs.
