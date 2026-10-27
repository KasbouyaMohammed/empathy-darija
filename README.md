# Can LLMs Be Empathetic in Moroccan Darija?

Code and results for the manuscript "Can LLMs Be Empathetic in Moroccan Darija? A Benchmark, a Validated Judge Jury, and a Distilled Model for Calibrated Empathy in Banking Customer Service" (M. Kasbouya, M. A. Lamhour, E. M. Lghaouch, and N. Sael, submitted to *IEEE Access*, 2026).

Banks deploy LLM chatbots as their first customer contact, and in Morocco customers write to their bank in Darija: in Arabic script, or in a Latin-script form (Arabizi) that encodes Arabic sounds with digits (3, 7, 9).

This repository contains the experiments behind a study of three questions:
- whether LLM empathy survives that change of language;
- whether empathy should be calibrated to the customer's emotion and the urgency of the request;
- whether LLM judges can be trusted to measure it.

**Benchmark.**
- 152 banking scenarios: 112 real CFPB complaints, stratified by category and emotion, and 40 DarijaBanking information requests.
- Each scenario is rendered in English, French, Modern Standard Arabic, Darija and Arabizi, for 760 messages.
- Automatic quality control: back-translation, LaBSE similarity of at least 0.80, and one regeneration round. It passes 99.7% of the 608 translated messages.
- 120 human-written Darija messages.
- A closed-set variety identifier for Darija and Arabizi. On 1,400 held-out human-written sentences its accuracy is 0.942, against 0.855 for a baseline version.

**Experiment.**
- Seven systems answer every message under three system prompts: P0 (neutral), P1 (always-on empathy) and P2 (calibrated to emotion × urgency).
- The systems are Claude Sonnet 5.5, Claude Haiku 5.5, Atlas-Chat-9B, Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-2-9B-it and Atlas-Chat-2B.
- That gives 21,539 judged replies; 461 truncated or empty replies are excluded by the pre-specified rule.
- A jury of five LLM judges from four model families scores every reply: Haiku and Qwen3, Llama and Gemma on all replies, Sonnet on a stratified 1,500.
- The judges are first validated against human EPITOME empathy labels translated into the five varieties.
- The calibrated policy is distilled into Atlas-Chat-2B with LoRA: 3 seeds, plus 25% and 50% data ablations. The distilled model answers with no system prompt.

**Results.**
- **The main gap is linguistic.** Share of replies to Darija messages written in Darija:

  | Sonnet | Atlas-Chat-9B | Haiku | Llama | Qwen3 | Gemma-2 |
  |---|---|---|---|---|---|
  | 96% | 96% | 73% | 40% | 6% | 2% |

  Most open models answer in MSA. Atlas-Chat-9B answers every variety in Darija, including 0% fidelity on Arabizi. Arabizi lowers empathy (−0.44 on 1–5 vs. English) and helpfulness (−0.57) for all systems.
- **Calibration.** It removes unsolicited sympathy from information requests: empathy judged appropriate in 99% of low-urgency replies under P2, against 92% under P1. It adds empathy to complaints: 81% under P2 against 45% under P0 at high urgency. The judges still prefer always-on empathy on urgent complaints (94% for P1).
- **Judges.** The median of the judges agrees with the human EPITOME labels at Spearman ρ = 0.61 to 0.71 in every variety. The Claude judges are stable across varieties. Qwen3-8B, the best judge in English (0.69), falls to 0.47 on Arabizi.
- **Distillation.** Without any prompt, the distilled model is non-inferior to its teacher, Sonnet under P2, according to the jury: empathy −0.02 (one-sided 95% lower bound −0.06), helpfulness −0.08 (−0.11), margin −0.3.
- **Robustness.** Refitted with the three non-Claude judges only, 93 of 98 RQ1/RQ2 contrasts keep their sign. The check also shows which judges can detect over-empathy: the open judges almost never label empathy excessive, whereas the Claude judges and the human annotator both flag the unsolicited sympathy of P1 on information requests. On the 120 human-written Darija messages, the policy effects on empathy and appropriateness replicate.
- **Blind human rating** of 199 Darija and Arabizi replies:
  - it agrees with the jury on empathy (ρ = 0.72);
  - it rates P2 appropriate on all 20 complaints, against 5% for P0;
  - it confirms the large gains of the distilled model over its base model.

## Repository structure

```
scripts/     pipeline: scenarios, messages, QC, replies, LoRA, judges, analysis, tables, figures (01-18)
results/     replies, judge scores, analysis tables, LoRA adapters, human ratings, logs
data/        input data, not shipped (provenance and download instructions in data/README.md)
requirements.txt
r-packages.txt
```

Support modules:
- `prompts.py`: the frozen experimental material: bank context, P0/P1/P2, judge prompts, required-action checklists, safety patterns. Every script imports it, so all systems and judges see the same text.
- `llm.py`: the Claude client, with Message Batches, a disk cache and a hard budget stop.
- `langid_darija.py`: the variety identifier.

## Installation

```
git clone https://github.com/KasbouyaMohammed/empathy-darija
cd empathy-darija
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The statistical models run in R 4.x with the packages in `r-packages.txt` (`lme4`, `lmerTest`, `ordinal`, `emmeans`).

The Claude calls need `ANTHROPIC_API_KEY` in a `.env` file at the repository root. Llama-3.1 and Gemma-2 are gated on Hugging Face and need `HF_TOKEN`.

## Data setup

The input datasets are not redistributed: CFPB complaints, EPITOME, DarijaBanking and DarijaBench. `data/README.md` lists, for each one:
- where to download it;
- which file to place in `data/raw/`;
- its licence.

Everything derived from them is under `results/`: replies, judge scores and analysis tables. Every manuscript table and figure can therefore be inspected without the raw data.

## Reproducing the experiments

Run from the repository root, in order:

```bash
python scripts/01_prepare_cfpb.py              # CFPB filtering, emotion classifier (Fig. 2)
python scripts/02_select_scenarios.py          # 112 complaint + 40 informational scenarios
python scripts/02b_natural_subset.py           # 120 human-written Darija messages
python scripts/03_build_messages.py            # messages in 5 varieties
python scripts/04_qc.py epitome                # back-translation + LaBSE + variety ID, EPITOME translations
python scripts/04_qc.py benchmark              # same, benchmark messages
python scripts/04a_langid_validation.py build  # held-out gold for the variety identifier
python scripts/04a_langid_validation.py eval   # accuracy of the identifier vs the baseline (Fig. 4)
python scripts/04b_finalize.py                 # regenerate failures once, final messages (Fig. 3)
python scripts/05_epitome_sample.py            # 240 EPITOME pairs, balanced by empathy level
python scripts/06_translate_epitome.py         # EPITOME pairs in the 5 varieties
python scripts/07_generate_claude.py           # Claude replies, P0/P1/P2, batch
python scripts/08_generate_open.py --model atlas9b   # open-model replies (also qwen3_8b, llama31_8b, gemma2_9b, atlas2b)
python scripts/09_build_training.py            # RQ4 training data: unseen scenarios, Sonnet P2 teacher
python scripts/10_train_lora.py --fraction 1.0 --seed 0   # LoRA (seeds 0-2; --fraction 0.5 / 0.25)
python scripts/11_judge.py --judge haiku --task replies   # judges (haiku, sonnet, qwen3_8b, llama31_8b, gemma2_9b; tasks replies, epitome)
python scripts/14_judge_precision_check.py --judge gemma2_9b --n 100   # fp16 vs fp32 judge scores
python scripts/12_analysis.py rq3              # judge validity against EPITOME labels (RQ3)
python scripts/12_analysis.py prepare          # long table reply x judge + objective measures
python scripts/12_analysis.py judges           # inter-judge alpha, judge emotion/urgency accuracy
python scripts/12_analysis.py natural          # human-written vs translated Darija
Rscript scripts/13_models.R                    # RQ1/RQ2 mixed models, Holm-corrected contrasts
Rscript scripts/15_rq4_selfpref.R              # RQ4 superiority/non-inferiority, judge self-preference
JUDGES=qwen3_8b,llama31_8b,gemma2_9b Rscript scripts/13_models.R   # robustness: non-Claude judges only (Table 14)
Rscript scripts/13b_natural_rq2.R             # RQ2 on the 120 human-written Darija messages (Table 14)
python scripts/18_human_eval.py score          # blind human rating (Tables 15-16)
python scripts/16_tables.py                    # LaTeX tables and number macros
python scripts/17_examples.py                  # per-system table and rule-selected examples
python scripts/make_tex_figures.py             # TikZ figures
```

The analysis can be rerun from the shipped results without any GPU or API call:
- the R models read `results/analysis/replies_long.csv`;
- `16_tables.py` reads `results/analysis/`, `results/lora/` and `results/human_eval/`;
- scripts that read generated replies expect them under `data/processed/`, so copy `results/replies/*.csv` there first.

Key protocol elements:

- **Policies:**
  - All three prompts share one simulated bank context. Placeholders such as `[HOTLINE]` and `[N]` stop models from inventing numbers.
  - The prompts differ only in the style instruction. P2 assesses emotion (anger, anxiety, distress, neutral) and urgency (high, medium, low) silently. Then it puts protective steps first at high urgency, validates in one sentence when the emotion is strongly negative, and uses no emotional language for neutral requests.
- **Decoding:**
  - Claude models: Message Batches API, low effort, default sampling. The API accepts no sampling parameters.
  - Open models: each model's own chat template and default generation configuration; for Qwen3, the model card's non-thinking settings. Maximum 512 new tokens.
  - Gemma-2-based models have no system role, so the policy is prepended to the user turn.
- **Precision:**
  - Gemma-2 and Atlas-Chat generate in fp32, because they are unstable in fp16 and the Turing GPUs have no bf16.
  - Qwen3 and Llama generate in fp16.
  - The local judges run in fp16. Gemma-2 in fp16 gave scores identical to fp32 on 100 EPITOME items (`results/judge/precision_check_gemma2_9b.csv`).
- **Exclusion rule:**
  - A reply truncated by the token limit is regenerated once with the next seed.
  - Greedy models reproduce the same output, so their truncated replies are excluded directly.
  - Empty replies are excluded.
- **Jury:**
  - Per reply and measure, the jury score is the median of the available judges; halves are rounded down for the EPITOME levels.
  - Judges never see the system or policy.
  - The rubric uses the EPITOME definitions verbatim, so the instrument applied to the replies is the one validated against human labels.
- **Distillation:**
  - The training scenarios never overlap the benchmark: CFPB complaints with TF-IDF cosine ≤ 0.6 to every seed, and unused DarijaBanking rows.
  - Training data: 3,413 training and 383 validation examples, after removing wrong-variety and unsafe teacher replies.
  - LoRA on Atlas-Chat-2B: r = 16, α = 32, all projection modules, 2 epochs, learning rate 2e-4, fp32.
  - The 25% and 50% subsets are nested.

## Runtime

| Step | Hardware | Time per run |
|---|---|---|
| LoRA training on 100% / 50% / 25% of the data | one 24 GB TITAN RTX | about 71 / 35 / 18 min |
| Open-model generation and local judging | four GPUs (`--shard i/4`) | most of the compute |
| Analysis scripts and R models | CPU | seconds to a few minutes |

The Claude calls go through the Message Batches API. `llm.py` caches every response, so reruns cost nothing.

## Reproducibility notes

The hypotheses, prompts, rubrics, checklists, safety patterns, model checkpoints, thresholds and analyses were fixed in a timestamped plan on 25 October 2026, before the main generation run. The manuscript reports every deviation from it.

| Item | Seed |
|---|---|
| Scenario selection and EPITOME sampling | 42 |
| RQ4 training data | 7 |
| Nested data ablation | 1234 |
| LoRA training | 0, 1, 2 |
| Bootstrap and human-evaluation sampling | 2026 |

Every seed is reported. No run is excluded from any aggregate.

## Manuscript table and figure mapping

| Manuscript | Repository source | Generator |
|---|---|---|
| Fig. 1 | pipeline schematic | `make_tex_figures.py` |
| Fig. 2 | `data/processed/cfpb_candidates.parquet` | `make_tex_figures.py` |
| Fig. 3 | `results/qc/qc_summary.csv` | `04b_finalize.py`, `make_tex_figures.py` |
| Fig. 4 | `results/qc/langid_test_predictions.csv` | `04a_langid_validation.py`, `make_tex_figures.py` |
| Fig. 5 | prompt design, summarised from `prompts.py` | `make_tex_figures.py` |
| Fig. 6, Tables 8-9 | `results/analysis/rq3_agreement.csv`, `rq3_drop_vs_en.csv` | `12_analysis.py rq3`, `16_tables.py` |
| Fig. 7 | `results/analysis/replies_objective.csv` | `12_analysis.py prepare`, `make_tex_figures.py` |
| Fig. 8 | `results/analysis/replies_long.csv` | `make_tex_figures.py` |
| Table 7 | `results/replies/` | `16_tables.py` |
| Table 10 | `results/analysis/rq3_interjudge_alpha.csv` | `12_analysis.py judges`, `16_tables.py` |
| Table 11 | `results/analysis/rq1_rq2_contrasts.csv` | `13_models.R`, `16_tables.py` |
| Table 12 | `results/analysis/emm_p_appropriate_policy_urgency.csv` | `13_models.R`, `16_tables.py` |
| Table 13 | `results/analysis/rq4_runs.csv`, `rq4_contrasts.csv` | `15_rq4_selfpref.R`, `16_tables.py` |
| Table 14 | `results/analysis/robust_qwen3_8b_llama31_8b_gemma2_9b/`, `natural_rq2_contrasts.csv` | `13_models.R` (with `JUDGES`), `13b_natural_rq2.R`, `16_tables.py` |
| Tables 15-16 | `results/human_eval/score_*.csv` | `18_human_eval.py score`, `16_tables.py` |
| Table 18 | `results/analysis/replies_long.csv` | `17_examples.py` |
| Table 19 | `results/replies/` | `17_examples.py` |

The other tables (1-6, 17) are written in the manuscript from the protocol and these outputs.

## Citation

M. Kasbouya, M. A. Lamhour, E. M. Lghaouch, and N. Sael, "Can LLMs Be Empathetic in Moroccan Darija? A Benchmark, a Validated Judge Jury, and a Distilled Model for Calibrated Empathy in Banking Customer Service," manuscript submitted to *IEEE Access*, 2026. Code and results: https://github.com/KasbouyaMohammed/empathy-darija

```bibtex
@misc{kasbouya2026empathy,
  title={Can LLMs Be Empathetic in Moroccan Darija? A Benchmark, a Validated Judge Jury,
         and a Distilled Model for Calibrated Empathy in Banking Customer Service},
  author={Kasbouya, Mohammed and
          Lamhour, Mohamed Akram and
          Lghaouch, El Mehdi and
          Sael, Nawal},
  year={2026},
  howpublished={\url{https://github.com/KasbouyaMohammed/empathy-darija}},
  note={Manuscript submitted to IEEE Access}
}
```

Once the paper is published, this entry will be replaced with the final citation and DOI.

## License

Code and results: MIT (`LICENSE`). The input datasets keep their own licences, documented in [data/README.md](data/README.md).
