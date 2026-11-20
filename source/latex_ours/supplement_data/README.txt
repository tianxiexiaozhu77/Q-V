Supplementary numerical records

The PDF supplement tabulates all fixed-support evaluations, all primary query-path
contrasts, and the main-paper repeated-support few-shot comparisons and protocols.
The main paper focuses its path analysis on AlphaRAD/RSNA and AlphaRAD/ChestXDet10.
Appendix B also reports the complete additional SIIM holdout and six-model cardiac
path experiments, including negative, mixed, and inconclusive contrasts.
The accompanying CSV files preserve numerical precision beyond the PDF display
rounding, including per-support-draw results and the CheXchoNet method comparisons
not tabulated in the PDF.
No patient identifiers, image filenames, or local absolute filesystem paths are included.

Files:
- fixed_support_all.csv: 138 completed tasks across ten models and eleven datasets,
  four branches, 1,032 metric values (120 classification, 12 localization, and six
  segmentation tasks). Method development used only RadZero on ChestXDet10,
  ChestXray14, OpenI, SIIM, and RSNA. OpenI is excluded from the current reporting
  scope, which contains four RadZero development comparisons and 134 subsequent
  comparisons with the fixed method. The development SIIM comparison uses 1,694
  test images; RadZero/SIIM (1,704) is a subsequent evaluation. All other models
  and datasets were evaluated after the method and parameters were fixed; some
  cross-model comparisons reuse datasets from RadZero development.
- evaluated_label_orders.csv: model/dataset label names and their evaluated order.
- path_development_*: AlphaRAD RSNA/ChestXDet10 path intervention.
- path_siim_*: additional same-source SIIM holdout intervention.
- cardiac_paths_*: all six-backbone CheXchoNet path branches and contrasts.
- fewshot_development_*: original 11-method/branch comparison on two backbones
  and two datasets, including Q and the other native branches over five draws.
- fewshot_taskres_sstext_means.csv: eight TaskRes/SS-Text+ summary rows for
  AlphaRAD/BiomedCLIP on RSNA/ChestXDet10 (five-draw means, sample SDs, and
  conditional test-group bootstrap intervals).
- fewshot_taskres_sstext_seeds.csv: all 40 TaskRes/SS-Text+ results from the
  same five support draws (seeds 42--46), with one test view per method.
- fewshot_taskres_sstext_comparisons.csv: all eight Q-minus-TaskRes/SS-Text+
  contrasts reported in the PDF, including mean differences, support-draw SDs,
  gain counts, and conditional paired 95% intervals. Differences are target
  minus reference; negative values and intervals crossing zero are retained.
- fewshot_taskres_sstext_per_class.csv: 44 class-level TaskRes/SS-Text+ summary
  rows (one RSNA class and ten ChestXDet10 classes per model and method), with
  five-draw means, sample SDs, and conditional test-group bootstrap intervals.
- fewshot_cardiac_*: all seven competitors, both view counts, and cardiac tasks.

TaskRes/SS-Text+ records:
These four files supplement the two added methods and the eight reported
Q-minus-adapter contrasts. Method keys taskres and sstext_plus denote the
feature-level OvR ports documented in the PDF; query denotes native Q. Q's
means and per-draw results remain in fewshot_development_means.csv and
fewshot_development_seeds.csv, joined by model, dataset, and seed where present.
The Q results agree exactly between the two saved result sources.
The per_class file contains class summaries, not class-by-draw records.
Intervals use 2,000 test-group bootstrap resamples (bootstrap seed 20261024),
conditional on the five supports, without multiple-comparison correction.
The resampling_unit column identifies patient groups for RSNA and image/exact-
duplicate groups for ChestXDet10. Matching support draws and test views does not
match the internal representation/readout: AlphaRAD Q retains native FLaS,
whereas the added adapters use pooled global features and cosine prediction.

Fixed-support RSNA patient exclusion:
The current fixed_support_all.csv uses 5,297 RSNA test images from 3,713 patients.
The ten original support images are unchanged. The official RSNA-to-NIH mapping
identified 30 other images from seven support patients in the historical
5,327-image test list; those images are excluded from all ten models and four
branches before rescoring the saved predictions. No support/test patients overlap
after this correction. The original prediction and result files remain archived.
Repeated-support and query-path RSNA experiments retain their separately
patient-disjoint supports and original 5,327-image, 3,720-patient test cohort;
their numerical records and intervals are unchanged. Fixed-support and repeated-
support Original scores can therefore differ because their test sets differ.

Conventions:
The method key positive_query denotes Positive-support-only adaptation:
the update uses the normalized positive support-feature mean in place of Q's
normalized positive-minus-negative mean, with the same eta=0.3, query-scale
policy, and native prediction path. Negative text prompts remain unchanged.
Absolute metrics are percentages; differences and SDs are percentage points.
AUPRC is trapezoidal PR-curve area, not average precision.
Means average seed-wise metrics, not predictions. SD is the sample SD (ddof=1).
Intervals are paired test-group bootstrap intervals conditional on fixed supports.
The macro_three_tasks rows summarize correlated labels and add no independent cohort.
The fixed-support historical QaTaCOV19 rows use test-mask-selected thresholds;
the main paper's validation-threshold experiment is a separate protocol.
Current tables include the ten models reported in the manuscript. Original
experiment files and historical verification records remain archived separately.
The main classification table averages ten dataset/protocol groups, retaining both SIIM protocols and
combining the three cardiac labels. The component ablation table instead retains one SIIM protocol per
model and averages nine datasets per model.
