# Data

The input datasets are not redistributed in this repository. Download them and place the files in `data/raw/` before running `scripts/01_prepare_cfpb.py` onwards. The scripts write their intermediate files to `data/processed/`.

| Dataset | Use in the study | Source | Licence |
|---|---|---|---|
| CFPB Consumer Complaint Database | 112 complaint scenarios; distillation training scenarios | Consumer Financial Protection Bureau; we use the Hugging Face mirror `BEE-spoke-data/consumer-finance-complaints` (config `has-text`, file `train-00002-of-00003.parquet` in `data/raw/cfpb/has-text/`) | Public domain (CC0) |
| DarijaBanking (Skiredj et al., 2024) | 40 informational scenarios; 120 human-written Darija messages; distillation training scenarios | Hugging Face, `AbderrahmanSkiredj1/DarijaBanking` | Terms of the dataset card |
| EPITOME (Sharma et al., EMNLP 2020) | 240 seeker-responder pairs with human empathy labels (ER, IP, EX), used to validate the judges | https://github.com/behavioral-data/Empathy-Mental-Health | Terms of the original repository |
| DarijaBench (Atlas-Chat, 2025) | Held-out gold for the variety identifier: FLORES+, MADAR, DODa and DODa transliterations | Hugging Face, `MBZUAI-Paris/DarijaBench` | Terms of the dataset card |

Posts with explicit self-harm content are excluded from the EPITOME sample (`scripts/05_epitome_sample.py`).

The derived results that contain no raw dataset text are shipped under `results/`:
- judge scores;
- analysis tables;
- LoRA adapters.

The generated replies under `results/replies/` include the customer message each reply answers.
