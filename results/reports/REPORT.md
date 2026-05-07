# Behavioral Characterization Report — Frozen artifacts

Summary
- Dataset: frozen cached features and checkpoint: checkpoints/task_14_final/ranker_best.pt
- Validation set: human-validated clean subset (9 entries) and Phase-2 compatibility ablation (9 entries)
- Key high-level signals: mean entropy (unmasked=1.057, masked=0.793), clean semantic success unmasked=8/9, masked=7/9, top1 change count=1/9

Narrative Findings

Partial semantic grounding:
- The system performs task-conditioned affordance candidate ranking: when clear affordances are present the model ranks compatible candidates highly (clean_semantic_success_unmasked=8/9).
- These results indicate *partial semantic grounding* rather than full semantic understanding — the model often relies on visual proxies and dataset priors.

Semantic contamination (dataset priors / shortcut learning):
- Negative-control images (e.g., baseball bat, snowboard) produce extreme incompatible scores (incompatible_max ≈ 1.0), showing strong dataset-driven shortcuts.
- Task-4 (`water plant`) demonstrates contamination: noisy task-to-class mappings in training annotations create high scores for distracting classes.

Calibration-vs-Ranking tradeoff:
- Applying compatibility masking reveals a calibration-vs-ranking tradeoff: masking reduces scores for incompatible distractors and can increase margins for true compatibles, but it also increases conservative rejects in some cases.
- The policy-level choice is therefore a calibration vs ranking decision (`calibration-vs-ranking tradeoff`), not a simple performance win from masking.

Uncertainty under ambiguity:
- Ambiguous cases have high predictive entropy (mean ≈ 1.89) and show the largest sensitivity to masking; masking can change accept/reject outcomes in ambiguous scenarios.

Effect of clean semantic supervision:
- The human-validated clean manifest (n=9) shows that clean semantic supervision reveals failure modes and provides defensible evaluation: it raises confidence in positive examples and highlights contamination in negatives.

Masking false-accept exemplar (key visual)
- Example: `data/coco/train2014/COCO_train2014_000000131498.jpg` (task 1, dining-table context). Unmasked the model selects `person` (low compat rank for table) and rejects; masked, the model selects `dining table` with high confidence and accepts.
- This single exemplar communicates four points concisely: shortcut priors in the data, calibration distortion introduced by masking, ambiguity sensitivity, and semantic contamination. It is the most important figure in the package.

Methodology (short)
- Frozen artifacts: all inference code, thresholds, and checkpoint are unchanged.
- Evaluations: Phase-1 task-conditioning (task-targeted retrieval), Phase-2 compatibility ablation (mask/unmask runs), and a human-validated clean subset (9 entries). Metrics: top1/top3, entropy, compatible_avg_rank, incompatible_max_score, reject flags, semantic_success by human criteria.

Evaluation Summary
- Master table: reports/master_results_table.csv (per-image records)
- Behavioral summary: reports/package_manifest.json and logs/architecture_validation/behavioral_regime/behavioral_regime_summary.json
- Plots: copied to reports/assets/ (entropy_vs_case_type, incompatible_proxy_vs_semantic, masked_vs_unmasked_top1_delta, reject_behavior_matrix)

Failure Analysis (concise)
- Shortcut learning: strong incompatible peaks on negative-control images indicate label/noise-driven shortcuts in the training data.
- Ambiguity: high-entropy cases yield unstable rankings and inconsistent reject/accept behavior.
- Distractor-heavy scenes: compatible objects are present but ranked lower; masking reduces distractor scores but doesn't fully recover compatible rank in all cases.

Limitations
- Small clean-manifest (n=9) — signals are interpretable but not statistically precise.
- Clean manifest bias: human validation choices encode subjective thresholds (confidence_level, affordance_state).
- Frozen thresholds and mask logic may not generalize to broader data without recalibration.

Next steps (deferred until documentation reviewed)
- Produce slide-ready PDF/PNG from `reports/` assets and finalize wording.
- If user requests, prepare a short appendix with the raw JSON rows and per-image visualizations.
