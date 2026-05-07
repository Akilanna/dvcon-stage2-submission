================================================================================
AUDIT PACKAGE FOR SECONDARY AI ANALYSIS
================================================================================

This folder contains all essential files needed for a comprehensive code audit
of the DVCON object-affordance detection pipeline.

FOLDER STRUCTURE:
================================================================================

📋 [ROOT DIRECTORY]
├── generalization_tackle_plan.txt     (Problem statement & solution framework)
├── requirements.txt                   (Python dependencies)
├── README_UPLOAD_GUIDE.txt           (This file)
│
├── 📁 inference/                      (Main inference pipeline)
│   ├── run_pipeline.py               (TaskAwareObjectSelector - main entry point)
│   └── stage2a_interface.py          (Stage2a interface for FPGA integration)
│
├── 📁 models/                         (Neural network architectures)
│   ├── ranking_mlp.py                (2-layer MLP for object suitability scoring)
│   ├── feature_extractor.py          (ResNet18 backbone + projection)
│   ├── scene_descriptor.py           (Scene context generator - gated or mean-pool)
│   ├── task_embedding.py             (Task representation from text)
│   └── ranker_best.pt                (Trained checkpoint: best ranker weights)
│
├── 📁 training/                       (Training pipeline components)
│   ├── train_ranker.py               (Train RankingMLP on COCO-Tasks ground truth)
│   ├── train_simple.py               (Train RankingMLP+SceneDescriptor on cached features)
│   ├── loss_functions.py             (Focal loss implementation)
│   ├── cached_dataset.py             (Load precomputed features from disk)
│   └── precompute_features.py        (Extract & cache features for fast training)
│
├── 📁 evaluation/                     (Metrics & analysis)
│   └── metrics.py                    (Top-1, Top-3, MRR computation)
│
├── 📁 scripts/                        (Utility scripts)
│   ├── generate_task_embeddings.py   (Create task embeddings from descriptions)
│   └── export_hls_weights.py         (Quantize & export weights for FPGA)
│
├── 📁 configs/                        (Configuration files)
│   ├── model_config.yaml             (Model architecture parameters)
│   └── training_config.yaml          (Training hyperparameters)
│
├── 📁 data_and_reports/               (Validation data & results)
│   ├── targeted_task_cases_gold2.csv        (112 annotation-backed test cases)
│   └── targeted_task_cases_gold2_summary.txt (Validation results summary)
│
├── verify_model.py                    (Standalone verification script)
├── build_gold_task_cases.py           (Build gold validation set from annotations)
└── run_all_280_tests.py              (Full 280-image pipeline test)


KEY FILES FOR AUDIT FOCUS:
================================================================================

1. START HERE:
   └─ generalization_tackle_plan.txt
      (Human-readable problem statement + phase-by-phase solution framework)

2. CORE INFERENCE PROBLEMATIC CODE:
   └─ inference/run_pipeline.py
      Section: Compatibility filter logic (lines ~180-210)
      Issue: Hard filtering + soft boost combination causing unexpected rejections

3. RANKER ARCHITECTURE:
   └─ models/ranking_mlp.py
      Issue: Trained on limited COCO-Tasks distribution, poor generalization

4. FILTERING INCONSISTENCY:
   └─ verify_model.py
      Section: TASK_COMPATIBILITY_PENALTY logic (lines ~140-160)
      Issue: Uses hard 0.01× penalty while run_pipeline.py uses soft +0.12 boost

5. VALIDATION DATA:
   └─ data_and_reports/targeted_task_cases_gold2.csv
      112 annotation-backed test cases showing 6/112 pass rate (5.4%)

6. TRAINING CODE:
   └─ training/train_simple.py
      Issue: Limited training diversity on COCO-Tasks subset


CRITICAL PROBLEMS IDENTIFIED:
================================================================================

1. HARD FILTERING OVER-REJECTION
   File: inference/run_pipeline.py + verify_model.py
   Issue: Compatibility filter applies strict penalties to non-whitelisted classes
   Impact: Removes valid objects even when ranker prefers them

2. DETECTOR BOTTLENECK
   File: models/ranker/ranking_mlp.py
   Issue: YOLO misses or mis-identifies objects in non-COCO-style images
   Impact: No chance for ranker to select correct object if not detected

3. NARROW TASK-OBJECT MAPPING
   File: inference/run_pipeline.py (TASK_RELEVANT_CLASSES)
   Issue: Whitelist doesn't match real annotation distributions
   Impact: Rejects valid alternative objects (e.g., "sit comfortably" → only "chair")

4. TRAINING DISTRIBUTION MISMATCH
   File: training/train_simple.py
   Issue: Ranker trained on clean COCO scenes, fails on diverse real images
   Impact: Model overfits to training distribution instead of learning affordance

5. CODE INCONSISTENCY
   File: verify_model.py vs run_pipeline.py
   Issue: Different filtering strategies (hard penalty vs soft boost)
   Impact: Audit results don't match production inference


TEST RESULTS:
================================================================================

280-Image Full Pipeline (20 images × 14 tasks):
├─ Total combinations: 280
├─ Matches found: 22
├─ Pass rate: 7.9%
└─ Report: final_verification_results/

112-Case Annotation-Backed Validation (from official COCO-Tasks):
├─ Total cases: 112 (8 per task)
├─ Matches found: 6
├─ Pass rate: 5.4%
└─ Report: data_and_reports/targeted_task_cases_gold2_summary.txt

Failure Mode Analysis:
├─ Detector failures: YOLO misses object
├─ Ranker failures: Wrong object ranked highest
└─ Filter failures: Valid object rejected by compatibility filter


RECOMMENDATIONS FOR SECONDARY AI:
================================================================================

1. Validate problem diagnosis by tracing one failing case through all stages
2. Separate concerns: detector failures vs ranker failures vs filter failures
3. Measure impact of each filtering logic independently
4. Check task-object mapping against real annotation distributions
5. Propose solutions in priority order (easy wins first)
6. Suggest testing strategy (unit tests per component)


HOW TO USE THIS PACKAGE:
================================================================================

For ChatGPT/Claude:
1. Extract this entire folder to your file context
2. Start with generalization_tackle_plan.txt
3. Reference specific file paths when analyzing issues
4. Use data_and_reports/ for concrete failure examples

For Secondary Analysis:
1. Review README_UPLOAD_GUIDE.txt (you are here)
2. Read generalization_tackle_plan.txt for context
3. Examine critical files in priority order (see above)
4. Check test results in data_and_reports/
5. Trace one failing case: COCO_train2014_000000528892.jpg + task_id=1


QUESTIONS TO GUIDE AUDIT:
================================================================================

✓ Why is verify_model.py using different filtering than run_pipeline.py?
✓ What is the actual task-object compatibility based on annotations?
✓ How much does YOLO detector miss vs how much does ranker misrank?
✓ What is the training distribution vs test distribution mismatch?
✓ Can we replace hard thresholds with learned confidence calibration?
✓ Which tasks have highest failure rate and why?
✓ Should we use an open-vocabulary detector instead?
✓ What loss function would better suit the generalization task?


CONTACT / CONTEXT:
================================================================================

Original Issue: Obvious cases like "sit comfortably" are being rejected despite
clear positive objects in the images.

Current Hypothesis: System is too narrow (COCO-locked) + filtering too hard
+ training data too limited → model rejects valid objects.

Goal: Move from whitelist-based COCO-specific system to general object-affordance
ranking with soft guidance + annotation-backed evaluation.

Last Updated: April 27, 2026
Package Size: ~24 files across 7 categories

================================================================================
