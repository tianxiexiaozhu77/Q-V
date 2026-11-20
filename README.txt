Paper experiment source release draft
====================================

Scope: latex_ours/main_with_supplement.tex (main.tex + sec/supplementary.tex).
This folder organizes the implementation of the tables in that manuscript.
It is a release draft, not a validated end-to-end inference distribution.
Original research files are unchanged. No GPU inference is run by packaging.

Quick start (Python 3.10+, standard library only):
  python tools/rebuild_tables.py --output build
  python tools/verify_manual_tables.py

The first command regenerates the Python-generated tables from saved aggregate
results in a temporary workspace and compares them with manuscript snapshots.
The second checks main Tables 2 and 4 and supplementary TaskRes, segmentation
and sensitivity tables against saved full-precision aggregate CSVs. Neither command trains models or recomputes
metrics from image-level predictions.

Layout
------
TABLE_INDEX.csv / TABLE_INDEX.json: every active table, its manuscript number,
  label, LaTeX source, experiment family, entry point and rendering method.
tables/: one extracted LaTeX environment per active table.
source/: selected original experiment, inference, evaluation and table code,
  preserving the relative workspace layout required by cross-directory imports.
source/latex_ours/supplement_data/: aggregate results and metric definitions.
SOURCE_MANIFEST.json: SHA-256 of collected source and result snapshots.
PORTABILITY_AUDIT.json: source lines with machine paths or local GPU restrictions.
tools/: organization, isolated rebuilding and numerical consistency checks.

Experimental reproduction map
-----------------------------
1. Fixed support: main Tables 1 and 3; Appendix A and fixed-support counts.
   AlphaRAD/RadZero historical four-arm evaluation:
     source/baselines/paper_draft_zh/scripts/replay_phase6.py
     source/experiments/query_search_20261021_152010/code_phase6/
   Eight other backbones:
     source/baselines/scripts/eval_query_refinement.sh
     source/baselines/local_setup/evaluate_query_transfer.py
     source/baselines/<model>/local_setup/query_refinement_adapter.py
   Cardiac extensions:
     source/baselines/local_setup/evaluate_chexchonet.py
   Additional model/dataset runs:
     source/experiments/table2_completion_20261103/
     source/experiments/query_resource_completion_20261103/
   Fixed RSNA patient exclusion and rescoring:
     source/experiments/rsna_fixed_patient_cleanup_20261110/recompute.py
   Aggregation: source/baselines/paper_draft_zh/build_verified_results.py.
   Rendering: source/latex_ours/scripts/build_supplement.py and
     build_classification_table.py.
   The packaged verified_results.json is a reduced rendering snapshot, not the
   original full provenance record. Source hashes of originals are kept separately.

2. Five support seeds: main Table 2, Appendix C means and paired intervals.
   source/experiments/fewshot_multiseed_20261024/
     prepare_protocol.py -> cache_multiseed.py -> evaluate_multiseed.py
     -> summarize_multiseed.py; controller: run_campaign.py.
   TaskRes / SS-Text+ additions:
     source/experiments/fewshot_taskres_sstext_20261109/
     prepare_experiment.py, fit_unit.py, taskres_adapter.py, sstext_adapter.py,
     evaluate.py, run_campaign.py.
   TaskRes and SS-Text+ are binary feature-level ports. AlphaRAD's pooled
   baseline features differ from Q's native readout. They are not exact
   reproductions of the upstream benchmark pipelines.

3. Aggregation/readout interventions: main Table 4 and Appendix B.
   source/experiments/query_path_intervention_20261024/
     intervention.py, infer.py, evaluate.py, run_campaign.py;
     prepare_siim.py, infer_siim.py, evaluate_siim.py, run_siim.py.
   Six-backbone cardiac paths:
     source/experiments/query_path_multibackbone_20261025/
     prepare_protocol.py, infer_*.py, *_intervention.py, summarize.py,
     run_campaign.py.

4. Validation-selected segmentation threshold:
   source/experiments/qata_validation_threshold_20261024/
     prepare_protocol.py -> infer_maps.py -> select_thresholds.py
     -> evaluate_fixed_test.py; controller: run_campaign.py.
   This is distinct from the historical test-mask threshold selection in
   fixed-support Appendix A. Do not pool those protocols.

5. Sensitivity grid:
   source/experiments/sensitivity_grid/run_sensitivity_campaign.py;
   eval_radzero_chexpert.py. Nine predefined eta/crop slices, not a full grid.
   e_sensitivity.tex is maintained separately; build_supplement.py does not
   regenerate it. The corrected 90-row aggregate grid is included as evidence. The older
   sensitivity_parameter_summary.csv is stale and must not be substituted.

Protocol tables describe interfaces, prompts, support sources and development
scope. They do not each have an independent training/evaluation program.
Commented-out tables, the token-MLP paper and unrelated later experiments are
outside the active manuscript scope. fewshot_neighbors is retained as a
build_supplement.py dependency even though its cardiac comparison tables are
currently commented out in the manuscript.

What is validated and what still requires setup
-----------------------------------------------
The table rebuild uses saved aggregate results and needs no ML packages.
Full inference requires original licensed datasets, model checkpoints, upstream
model repositories, frozen support/test manifests and each backbone's compatible
environment. These assets are not included here. Patient/image manifests and
prediction caches are deliberately not bulk-copied into this public draft.
The original local launchers still assume per-model virtual environments; some
controllers restrict physical GPUs to 4--7. Machine-specific paths and policy
lines are enumerated in PORTABILITY_AUDIT.json. Do not interpret these scripts
as portable launchers until you configure paths, environments and device IDs.
Historical JSON manifests authenticate source hashes: editing code or paths
requires recording a new run manifest, not pretending the old hash still applies.

Do not install every backbone into one guessed requirements.txt. Reconstruct
per-model environments from the supplied environment version record and upstream
projects. Third-party code/checkpoint licenses remain applicable; this draft
assigns no new license to the original or third-party code.

Scientific conventions to preserve
----------------------------------
- AUROC / trapezoidal AUPRC; AP is not interchangeable with AUPRC.
- 138 completed fixed-support evaluations: 120 classification, 12 localization,
  6 segmentation. These are distinct from repeated-support comparisons.
- Fixed RSNA uses 5,297 test images after patient exclusion. Already separated
  repeated-support/path experiments keep their 5,327-image test protocol.
- Main Table 1 averages 10 dataset/protocol groups; Table 3 uses 9 datasets per
  model with SIIM 1,704 only and the three cardiac labels combined.
- Five-support-seed intervals condition on seeds 42--46 and are unadjusted for
  multiple comparisons; support SD and test-bootstrap uncertainty differ.
- Q uses one test view; Q+V uses two. Preserve negative and mixed results.

Packaging can be refreshed locally using tools/organize.py from the original
medical workspace. It is a maintainer utility and requires that workspace.
Do not run historical experiment outputs as new output destinations.
