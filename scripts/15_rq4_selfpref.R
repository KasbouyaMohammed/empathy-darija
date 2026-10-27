# RQ4 (distilled model) and RQ3 self-preference, on results/analysis/replies_long.csv and replies_objective.csv.
# RQ4, Darija benchmark messages (ary, ary_latn):
#   LoRA-2B (3 seeds, 100% data, no prompt) vs prompted Atlas-Chat-2B P2: superiority, one-sided;
#   LoRA-2B vs Claude Sonnet 5.5 P2: non-inferiority, margin 0.3 on 1-5 scales, one-sided 95%;
#   data-size curve (25/50/100%, seed 0); mean and SD over seeds.
# Self-preference: score ~ judge_family * reply_family + variety + policy + (1 | item) + (1 | judge).
# Run: ~/micromamba/envs/rstats/bin/Rscript scripts/15_rq4_selfpref.R
suppressMessages({library(lme4); library(lmerTest); library(emmeans); library(dplyr); library(readr)})
emm_options(lmer.df = "asymptotic", pbkrtest.limit = 0, lmerTest.limit = 0)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out <- file.path(root, "results/analysis")
d <- read_csv(file.path(out, "replies_long.csv"), show_col_types = FALSE)
obj <- read_csv(file.path(out, "replies_objective.csv"), show_col_types = FALSE)
MARGIN <- 0.3

# ---- RQ4 -------------------------------------------------------------------------------------------------
arm <- function(x) case_when(
  grepl("^lora2b_frac100", x$system) ~ "lora",
  x$system == "atlas2b" & x$policy == "P2" ~ "base_p2",
  x$system == "claude-sonnet-5-5" & x$policy == "P2" ~ "sonnet_p2",
  TRUE ~ NA_character_)
dd <- d %>% filter(set == "benchmark", variety %in% c("ary", "ary_latn")) %>% mutate(arm = arm(.)) %>%
  filter(!is.na(arm)) %>% mutate(arm = factor(arm, levels = c("lora", "base_p2", "sonnet_p2")),
                                 scenario = factor(scenario_id), judge = factor(judge),
                                 appropriate = as.integer(approp == 0))
oo <- obj %>% filter(set == "benchmark", variety %in% c("ary", "ary_latn")) %>% mutate(arm = arm(.)) %>%
  filter(!is.na(arm)) %>% mutate(arm = factor(arm, levels = c("lora", "base_p2", "sonnet_p2")),
                                 scenario = factor(scenario_id))
res <- list()
if (nlevels(droplevels(dd$arm)) == 3) {
  re_j <- if (nlevels(dd$judge) > 1) " + (1 | judge)" else ""
  for (y in c("empathy", "helpfulness", "appropriate", "action_cov")) {
    m <- lmer(as.formula(paste(y, "~ arm + variety + (1 | scenario)", re_j)), data = dd)
    ct <- summary(contrast(emmeans(m, ~ arm), list("lora - base_p2" = c(1, -1, 0), "lora - sonnet_p2" = c(1, 0, -1))),
                  infer = TRUE, level = 0.90, adjust = "none")   # 90% two-sided = 95% one-sided bounds
    ct$outcome <- y
    res[[length(res) + 1]] <- as.data.frame(ct)
  }
  mf <- lmer(fidelity ~ arm + variety + (1 | scenario), data = oo)
  ct <- summary(contrast(emmeans(mf, ~ arm), list("lora - base_p2" = c(1, -1, 0), "lora - sonnet_p2" = c(1, 0, -1))),
                infer = TRUE, level = 0.90, adjust = "none")
  ct$outcome <- "fidelity"
  res[[length(res) + 1]] <- as.data.frame(ct)
  r <- bind_rows(res) %>% mutate(
    p_superiority = ifelse(contrast == "lora - base_p2", pnorm(-z.ratio), NA),          # H: lora > base
    noninferior = ifelse(contrast == "lora - sonnet_p2" & outcome %in% c("empathy", "helpfulness"),
                         asymp.LCL > -MARGIN, NA))
  write_csv(r, file.path(out, "rq4_contrasts.csv"))
  print(as.data.frame(r %>% select(outcome, contrast, estimate, asymp.LCL, asymp.UCL, p_superiority, noninferior)), row.names = FALSE)
}
# Per-run means (seeds and data fractions), Darija messages.
runs <- d %>% filter(set == "benchmark", variety %in% c("ary", "ary_latn"), grepl("^lora2b_", system)) %>%
  group_by(system, item_key) %>% summarise(across(c(empathy, helpfulness, action_cov), ~ median(.x, na.rm = TRUE)),
                                           appropriate = mean(approp == 0, na.rm = TRUE), .groups = "drop") %>%
  left_join(obj %>% select(item_key, fidelity), by = "item_key") %>%
  group_by(system) %>% summarise(across(c(empathy, helpfulness, action_cov, appropriate, fidelity), mean), n = n())
write_csv(runs, file.path(out, "rq4_runs.csv"))
print(as.data.frame(runs), row.names = FALSE)

# ---- RQ3 self-preference -------------------------------------------------------------------------------
sp <- d %>% filter(set == "benchmark", policy %in% c("P0", "P1", "P2")) %>%
  mutate(item = factor(item_key), judge = factor(judge), judge_family = factor(judge_family),
         reply_family = factor(family), variety = factor(variety), policy = factor(policy))
if (nlevels(sp$judge_family) == 2 && nlevels(sp$judge) > 2) {
  out_sp <- list()
  for (y in c("empathy", "helpfulness")) {
    m <- lmer(as.formula(paste(y, "~ judge_family * reply_family + variety + policy + (1 | item) + (1 | judge)")), data = sp)
    co <- summary(m)$coefficients
    out_sp[[y]] <- data.frame(outcome = y, term = rownames(co), co, check.names = FALSE)
    em <- as.data.frame(emmeans(m, ~ reply_family | judge_family))
    write_csv(em, file.path(out, paste0("selfpref_emm_", y, ".csv")))
  }
  r <- bind_rows(out_sp)
  write_csv(r, file.path(out, "rq3_selfpreference.csv"))
  print(as.data.frame(r %>% filter(grepl(":", term))), row.names = FALSE)
}
