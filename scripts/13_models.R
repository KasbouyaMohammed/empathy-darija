# Confirmatory mixed models for RQ1 and RQ2 (frozen plan), on results/analysis/replies_long.csv
# (one row per reply x judge, written by 12_analysis.py prepare).
#   score ~ variety * policy + system + urgency + emotion + (1 | scenario) + (1 | judge),
# plus policy:urgency and policy:emotion so that the planned by-urgency and by-emotion policy contrasts
# are estimable. Appropriateness: cumulative-link mixed model on too_little < appropriate < excessive,
# and a logistic mixed model on P(appropriate). Fidelity is a property of the reply, so it is modelled once
# per reply (no judge effect).
# Run: ~/micromamba/envs/rstats/bin/Rscript scripts/13_models.R
# Robustness: JUDGES=qwen3_8b,llama31_8b,gemma2_9b Rscript scripts/13_models.R refits the judged outcomes with
# those judges only and writes to results/analysis/robust_<judges>/ (fidelity is objective and is not refitted).
suppressMessages({library(lme4); library(lmerTest); library(ordinal); library(emmeans); library(dplyr); library(readr)})
emm_options(lmer.df = "asymptotic", pbkrtest.limit = 0, lmerTest.limit = 0)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out <- file.path(root, "results/analysis")
d <- read_csv(file.path(out, "replies_long.csv"), show_col_types = FALSE)
obj <- read_csv(file.path(out, "replies_objective.csv"), show_col_types = FALSE)

# Prompted benchmark replies (LoRA and the natural subset are analysed separately).
prep <- function(x) x %>% filter(set == "benchmark", policy %in% c("P0", "P1", "P2")) %>%
  mutate(variety = factor(variety, levels = c("en", "fr", "msa", "ary", "ary_latn")),
         policy = factor(policy, levels = c("P0", "P1", "P2")),
         system = factor(system), urgency = factor(urgency, levels = c("low", "medium", "high")),
         emotion = factor(emotion_target, levels = c("neutral", "anger", "anxiety", "distress")),
         scenario = factor(scenario_id))
judges <- strsplit(Sys.getenv("JUDGES"), ",")[[1]]
if (length(judges)) {
  d <- d %>% filter(judge %in% judges)
  out <- file.path(out, paste0("robust_", paste(judges, collapse = "_")))
  dir.create(out, showWarnings = FALSE)
}
b <- prep(d) %>% mutate(judge = factor(judge))
bo <- prep(obj)
# (1 | judge) needs at least two judges; with one (preliminary runs) it is dropped.
re_j <- if (nlevels(b$judge) > 1) " + (1 | judge)" else ""
rhs <- paste("variety * policy + policy:urgency + policy:emotion + system + urgency + emotion + (1 | scenario)", re_j)

rq1_pairs <- list("ary - en" = c(-1, 0, 0, 1, 0), "ary - fr" = c(0, -1, 0, 1, 0), "ary - msa" = c(0, 0, -1, 1, 0),
                  "ary_latn - en" = c(-1, 0, 0, 0, 1), "ary_latn - fr" = c(0, -1, 0, 0, 1),
                  "ary_latn - msa" = c(0, 0, -1, 0, 1))
rq2_pairs <- list("P2 - P0" = c(-1, 0, 1), "P2 - P1" = c(0, -1, 1))
res <- list()
add <- function(rq, outcome, ct, scale = "response") {
  s <- as.data.frame(summary(ct, infer = TRUE, adjust = "none"))
  s$rq <- rq; s$outcome <- outcome; s$scale <- scale
  res[[length(res) + 1]] <<- s
}
by_groups <- function(model, outcome, scale) {
  for (g in c("urgency", "emotion"))
    add(paste0("RQ2_by_", g), outcome, contrast(emmeans(model, as.formula(paste("~ policy |", g))), rq2_pairs), scale)
}

for (y in c("empathy", "helpfulness", "action_cov")) {
  m <- lmer(as.formula(paste(y, "~", rhs)), data = b, REML = TRUE)
  add("RQ1", y, contrast(emmeans(m, ~ variety), rq1_pairs))
  add("RQ2", y, contrast(emmeans(m, ~ policy), rq2_pairs))
  by_groups(m, y, "response")
  write_csv(as.data.frame(emmeans(m, ~ variety | policy)), file.path(out, paste0("emm_", y, "_variety_policy.csv")))
}

# Appropriateness (ordinal) and P(appropriate).
b$approp_f <- factor(b$approp, levels = c(-1, 0, 1), labels = c("too_little", "appropriate", "excessive"), ordered = TRUE)
mc <- clmm(as.formula(paste("approp_f ~", rhs)), data = b, Hess = TRUE)
add("RQ2", "appropriateness_latent", contrast(emmeans(mc, ~ policy), rq2_pairs), "latent")
by_groups(mc, "appropriateness_latent", "latent")
b$appropriate <- as.integer(b$approp == 0)
ma <- glmer(as.formula(paste("appropriate ~", rhs)), data = b, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
add("RQ2", "p_appropriate", contrast(emmeans(ma, ~ policy), rq2_pairs), "logit")
by_groups(ma, "p_appropriate", "logit")
write_csv(as.data.frame(emmeans(ma, ~ policy | urgency, type = "response")), file.path(out, "emm_p_appropriate_policy_urgency.csv"))
write_csv(as.data.frame(emmeans(ma, ~ policy | emotion, type = "response")), file.path(out, "emm_p_appropriate_policy_emotion.csv"))
write_csv(b %>% group_by(policy, urgency, emotion, appropriateness) %>% summarise(n = n(), .groups = "drop"),
          file.path(out, "appropriateness_counts.csv"))

if (!length(judges)) {
# Language fidelity, once per reply. Quasi-complete separation (some systems answer every French message
# in French) inflates the logit SEs, so a linear probability mixed model gives the robust percentage-point
# contrasts (deviation 4); the logistic model is kept for reference.
mf <- glmer(fidelity ~ variety * policy + system + urgency + (1 | scenario), data = bo, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
add("RQ1", "fidelity_logit", contrast(emmeans(mf, ~ variety), rq1_pairs), "logit")
ml <- lmer(fidelity ~ variety * policy + system + urgency + (1 | scenario), data = bo)
add("RQ1", "fidelity", contrast(emmeans(ml, ~ variety), rq1_pairs), "probability")
# H1: the Darija gap is smaller for Atlas-Chat-9B than for the other open models (LPM, interaction term).
bo$darija <- as.integer(bo$variety %in% c("ary", "ary_latn"))
open_sys <- c("atlas9b", "qwen3_8b", "llama31_8b", "gemma2_9b")
bo_open <- bo %>% filter(system %in% open_sys) %>% mutate(atlas = as.integer(system == "atlas9b"))
if (length(unique(bo_open$system)) > 1) {
  for (y in c("fidelity")) {
    mg <- lmer(as.formula(paste(y, "~ darija * atlas + policy + (1 | scenario)")), data = bo_open)
    co <- summary(mg)$coefficients
    write_csv(data.frame(term = rownames(co), co, check.names = FALSE), file.path(out, paste0("rq1_atlas_gap_", y, ".csv")))
  }
}
write_csv(bo %>% group_by(system, variety) %>% summarise(fidelity = mean(fidelity), n = n(), .groups = "drop"),
          file.path(out, "fidelity_system_variety.csv"))
}

r <- bind_rows(res)
r$p_holm <- ave(r$p.value, r$rq, FUN = function(p) p.adjust(p, "holm"))
write_csv(r, file.path(out, "rq1_rq2_contrasts.csv"))
show <- r %>% select(any_of(c("rq", "outcome", "urgency", "emotion", "contrast", "estimate", "asymp.LCL", "asymp.UCL", "p_holm")))
print(as.data.frame(show %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)
