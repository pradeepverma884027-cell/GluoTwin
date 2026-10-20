# train_*.py — one script per model, input = your original CSV files

```
train_scripts/
  data/                 <- put your 5 original files here: GlucoTwin.csv smart_wearable.csv t1dm.csv t2dm.csv ohhim.csv
  common.py             shared loading / metrics / training loop
  prep_pipeline.py      cleaning + feature engineering + split + scaling (runs automatically)
  train_linear.py       Ridge / Logistic + "same as now" baseline   (run first)
  train_rf.py           Random Forest
  train_lgbm.py         LightGBM
  train_xgb.py          XGBoost   (--gpu for GPU)
  train_gru.py          GRU, only cgm_t1dm_t2dm and ohhim (needs PyTorch)
  job_cpu.sh  job_gpu.sh   SLURM templates
```

## Quick start
```bash
python -m venv ~/envs/glu && source ~/envs/glu/bin/activate && pip install -r requirements.txt
python train_linear.py --raw_dir data          # 1st run builds processed/ from your CSV, then trains
python train_lgbm.py   --raw_dir data          # later runs reuse processed/
python train_rf.py     --raw_dir data
python train_xgb.py    --raw_dir data
python train_gru.py    --raw_dir data          # GPU recommended
```
Results: `results/<model>_metrics.csv` (val + test), trained models in `results/models/` (GRU: `results/gru__<dataset>.pt`).

## Useful options (all scripts)
- `--datasets glucotwin ohhim`  only these (names: smart_wearable, glucotwin, cgm_t1dm_t2dm, ohhim)
- `--targets target_hypo_60`    only this target column
- `--n_jobs 16`  CPU threads;  `--rebuild`  redo data preparation;  `--processed_dir / --results_dir`  change folders
- Model knobs: `--alpha` (linear), `--trees --min_leaf` (rf), `--lr --leaves` (lgbm), `--lr --depth --gpu` (xgb), `--epochs --hidden --batch --lr` (gru)

## What each dataset predicts
| Dataset | Targets | Split |
|---|---|---|
| smart_wearable | health_risk (0/1) | stratified rows |
| glucotwin | glucose t+30/60/90/120, peak, excursion, time-to-peak, iAUC, spike>180 | by participant |
| cgm_t1dm_t2dm | cgm t+30, t+60, hypo in 60 min | by patient |
| ohhim | glucose t+30, t+60, hypo in 60 min | by time, per patient |

## SLURM
Edit `YOUR_*_PARTITION` and `module load` in the job files, `mkdir logs`, then
`sbatch --export=ALL,MODEL=lgbm job_cpu.sh` or `sbatch job_gpu.sh`.

## Notes
- Tested here (pandas 3.0.2, scikit-learn 1.8): train_linear, train_rf, train_lgbm, train_xgb ran end-to-end from the raw CSVs.
  `train_gru.py` is only compile-checked (no PyTorch in my sandbox): run `--epochs 2` first.
- Do not change the split logic: val is used for early stopping, test only for final numbers.
- If it errors on an older pandas version, send me the error text.
