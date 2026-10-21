# Exploratory feature statistics

Run the noninteractive script from the repository root after configuring the
private recordings and restoring the original subject key and reviewed annotations
as described in the root README. With root requirements installed, regenerate
the audit inventory and reviewed processing outputs before the statistics:

```powershell
python audit/audit_data.py
python processing/pipeline.py init
python processing/pipeline.py run
python statistics/feature_statistics.py
```

The audit generates the private `audit/recording_inventory.csv` required by the
statistics script. Keep that inventory, review files and statistical outputs
private. Notebooks and the development test script are not included in this folder.
No model training or internet access is needed. Root requirements include the
scientific libraries and Pillow for audit figures.

## Decisions made before this statistical run

- Main analysis uses saved reviewed intervals and documented set exclusions.
  It requires the processing run to match current annotations exactly. If not, 
  rerun `py -3.13 processing/pipeline.py run` before opening the notebook.
- “Early” and “late” are first/last thirds of **elapsed active-set time**, not
  repetition-count thirds or independent physiological fatigue states. The middle third is omitted.
- A complete window must lie inside its third. Windows straddling a boundary do
  not contribute. At least three defined eligible windows per stage/set/feature
  are required; no missing values are imputed and no failed pair becomes zero.
- Calculate the median within each stage of each set. Retain sets with both
  stages, then calculate the late-minus-early difference. Average paired sets
  equally within each participant/leg; average the two legs equally within each
  participant. Both legs are required in the main comparison. Feature-specific
  subject and set counts are exported; thousands of windows are not thousands
  of independent observations. Later sets may start with residual fatigue.
- All 36 descriptors are tested and reported, including known redundant ones.
  RMS/MDF/MNF are initially illustrated because these were previously specified,
  not because they have the smallest p-values. All findings are exploratory:
  trajectories have already been inspected.

## Tests, effect estimates and uncertainty

Primary test: two-sided **exact paired sign test** on participant differences.
Under its null, positive/negative changes among nonzero differences are equally
probable across independent participants. It does not assume a normal or symmetric
distribution of magnitudes, and it does not test the size of the mean difference.
Its deliberate cost is limited power and ignoring magnitude. Differences within
1e-12 of zero count as ties and are omitted from the sign-test denominator.
Both total participant count and the nonzero count are displayed.

Do not automatically switch tests based on a normality p-value or use whichever
test produces significance. Histograms/Q-Q plots are for assessing distributions
and influential values, not selecting the smallest p-value. Wilcoxon signed rank
is a possible separate, prespecified analysis if its symmetry assumption is
defended; it is not silently substituted here.

Effects include mean/median participant change in original feature units,
standardized paired mean change (mean delta / sample SD of deltas), direction
counts and the exact positive-fraction binomial interval. Mean-change uncertainty
uses 10,000 BCa bootstrap resamples of participant summaries, with fixed seeds.
Participants are resampled intact, preserving the paired early/late structure.
With only ~15 convenience-sampled participants, intervals can be unstable and
do not establish external generalization. Degenerate observed differences are
explicitly flagged instead of claiming genuine absence of uncertainty.

Mean-change CIs and sign tests target different quantities, so they can disagree.
The 95% mean-change CIs are pointwise and are **not multiplicity-adjusted**.

Primary p-value correction: Benjamini-Yekutieli (BY) over all 36 main comparisons,
chosen because feature tests are dependent and arbitrary dependence is allowed.
Benjamini-Hochberg (BH) is shown additionally but has stronger dependence
conditions. Never choose between corrections based on the attractive result.
Additional columns adjust the entire set of scenario-feature tests together.
An adjusted p-value is not the probability the null is true or that a feature
will generalize; nonsignificance is not evidence of equivalence or uselessness.

## Sensitivity and redundancy

Main: reviewed intervals. Sensitivities, all retaining the same reviewed set
decisions: whole CSV duration, shrinking each reviewed interval by 250 ms on each
end, and omitting provisional low-frequency/transient flags. The latter is a
stress test, not adoption of those flags as validated exclusion criteria. Do not
choose the interval/QC policy with the strongest p-values. Pair availability can
change, so support and matched-cohort comparisons are exported separately.

Redundancy is Spearman correlation of **participant-level changes**, not pooled
overlapping windows. High absolute correlation (>=0.9) is flagged descriptively,
with paired subject counts. There are no correlation p-values or automatic
predictor removals. Known exact redundancy is recorded (IEMG/MAV for a fixed
window, Hjorth activity/variance). Correlation of changes is not the same as
collinearity of raw model inputs; inspect training-fold predictor redundancy
again during modelling.

## Report use and model selection

Tables, participant plots and `report_notes.txt` can support the description of
feature behaviour and limitations. They do not validate peripheral fatigue.
Stage, contraction phase, quiet signal, artifacts, recording gain and prior-set
fatigue can all affect the contrast. No MVC, force, Borg or video reference exists.

`feature_evidence_worksheet.csv` contains effect sizes, corrected p-values,
sensitivity, redundancy and empty rationale/selection fields. It intentionally
does not output a model-selected feature set. Full-cohort p-value filtering before
LOSO leaks information. Retain the mandated RMS+MDF baseline; freeze an algorithm
for any additional feature selection and fit it inside grouped inner training
validation. Data-guided changes to the overall modelling strategy should also be
disclosed as exploratory. PCA does not establish physiological relevance.

Outputs under `outputs/<analysis_id>/` include paired set/leg/subject tables,
window-stage assignments, all comparisons, main comparisons, a feature evidence
worksheet, subject-change correlation matrices, 36 per-feature delta plots,
paired-stage plots, distribution diagnostics, sensitivity plots and report notes.
The manifest records source hashes, code/configuration, library versions and seeds.
Do not rename these tables `results.csv`: the assignment's model-by-fold file is
a separate later deliverable.

References:

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html
- https://scikit-learn.org/stable/common_pitfalls.html
