# Training package

Data is ready: `processed/<dataset>_model_ready.csv` (scaled with train-only stats, `split` column = train/val/test)
and `processed/schema_<dataset>.json` (feature list, targets, scaling params).

## Run order
```bash
python 01_baselines.py            # "same as now" + Ridge/Logistic  -> results/baseline_metrics.csv
python 02_train_lgbm.py           # LightGBM, all datasets/targets  -> results/lgbm_metrics.csv + models + importance
python 03_train_gru.py cgm_t1dm_t2dm    # needs PyTorch + GPU       -> results/gru_*_metrics.csv
python 03_train_gru.py ohhim
```
One dataset only: `python 02_train_lgbm.py --datasets ohhim --targets target_hypo_60`

## On the cluster (SLURM)
1. Upload folder, make env: `python -m venv ~/envs/glu && source ~/envs/glu/bin/activate && pip install -r requirements.txt` (+ torch for GRU).
2. Edit `YOUR_*_PARTITION` and `module load` lines in `job_lgbm.sh` / `job_gru.sh` (see `sinfo`, `module avail`).
3. `mkdir logs && sbatch job_lgbm.sh`, then `sbatch job_gru.sh`. Watch with `squeue -u $USER`.

## Test scores from my run (test split, LightGBM unless noted)
| Dataset | Target | Baseline | LightGBM |
|---|---|---|---|
| smart_wearable | health_risk AUC | 0.871 (Logistic) | 0.849 |
| glucotwin | glucose t+30 RMSE | 18.1 (same-as-now), 9.0 (Ridge) | 8.6 |
| glucotwin | glucose t+120 RMSE | 64.6 (same-as-now), 28.0 (Ridge) | 20.9 |
| glucotwin | spike over 180 AUC | 0.944 (Logistic) | 0.960 |
| cgm_t1dm_t2dm | cgm t+30 RMSE | 17.6 (same-as-now), 13.7 (Ridge) | 13.2 |
| cgm_t1dm_t2dm | cgm t+60 RMSE | 29.3 (same-as-now), 24.6 (Ridge) | 23.3 |
| cgm_t1dm_t2dm | hypo in 60 min AUC | 0.982 (Logistic) | 0.980 |
| ohhim | glucose t+30 RMSE | 24.5 (same-as-now) | 24.1 |
| ohhim | glucose t+60 RMSE | 35.6 (same-as-now) | 33.8 |
| ohhim | hypo in 60 min AUC | 0.974 (Logistic) | 0.985 |

Notes: gains over simple models are small for wearable, cgm hypo and ohhim; GRU is where to look next.
`03_train_gru.py` compiles but was NOT run here (no PyTorch in my sandbox), so test it with a short run first (`--epochs 2`).
ohhim has only 6 patients and ~5% hypo events, so its hypo precision is noisy.
