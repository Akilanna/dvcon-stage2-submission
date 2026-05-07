# DVCON Stage-2 Submission Repository

This repository packages the final Stage-2 competition deliverables for the DVCON affordance-ranking system.

## What Is Included

- Reproducible code bundle in `src/` (core scripts from the required upload set)
- Competition engineering scripts in `competition/`
- Final competition outputs:
  - `competition/final_video_demo/` (14 curated per-task demo folders)
  - `competition/full_demo_verification.csv`
  - `competition/full_demo_summary.json`
  - `competition/final_demo_grid.png`
- Final report artifacts in `results/reports/`
- Final checkpoint(s) in `checkpoints/competition_14task_final/`
- Final training logs in `logs/competition_14task_final/`
- Submission/supporting docs in `docs/`
- Final manifests and cache summary in `data/`

## Repository Structure

```
DVCON_stage2_submission/
  src/                       # Required code bundle scripts
  configs/                   # Model/training YAML configs
  competition/               # Competition scripts + final demo outputs
  results/                   # Final CSV/JSON/grid and report artifacts
  checkpoints/               # Final competition model weights
  logs/                      # Final competition training logs
  data/                      # Final manifests and cache build summary
  docs/                      # Stage docs and upload guide
  requirements.txt
  README.md
```

## Final Result Entry Points

- Final curated video demo set: `competition/final_video_demo/`
- Final demo index: `competition/final_video_demo/index.csv`
- Full sweep CSV: `competition/full_demo_verification.csv`
- Full sweep summary: `competition/full_demo_summary.json`
- Final demo grid: `competition/final_demo_grid.png`

## Quick Start

1. Create and activate environment.
2. Install dependencies from `requirements.txt`.
3. Run the verification sweep or demo-grid generation scripts as needed.

Example commands:

```bash
python competition/run_full_demo_verification.py
python competition/generate_final_demo_grid.py
python competition/prepare_video_demo_sequence.py
```

## Notes

- This repository is organized for Stage-2 submission clarity and reproducibility.
- It intentionally contains final curated results and selected artifacts rather than the entire experiment history.
