# RQ2 replicated on the 120 human-written Darija messages (natural subset, not pre-registered).
# These messages are all complaints or service requests (high or medium urgency), so the test is whether the
# policy effects found on the benchmark complaints hold on real Darija text:
#   score ~ policy * urgency + system + (1 | scenario) + (1 | judge), prompted systems only.
# Run: ~/micromamba/envs/rstats/bin/Rscript scripts/13b_natural_rq2.R
suppressMessages({library(lme4); library(lmerTest); library(emmeans); library(dplyr); library(readr)})
emm_options(lmer.df = "asymptotic", pbkrtest.limit = 0, lmerTest.limit = 0)
root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out <- file.path(root, "results/analysis")
d <- read_csv(file.path(out, "replies_long.csv"), show_col_types = FALSE) %>%
  filter(set == "natural", policy %in% c("P0", "P1", "P2"),
         system %in% c("claude-sonnet-5-5", "claude-haiku-5-5", "atlas9b", "qwen3_8b", "llama31_8b", "gemma2_9b")) %>%
  mutate(policy = factor(policy, levels = c("P0", "P1", "P2")), urgency = factor(urgency, levels = c("medium", "high")),
         system = factor(system), scenario = factor(scenario_id), judge = factor(judge),
         appropriate = as.integer(approp == 0))
rhs <- "policy * urgency + system + (1 | scenario) + (1 | judge)"
pairs <- list("P2 - P0" = c(-1, 0, 1), "P2 - P1" = c(0, -1, 1))
res <- list()
add <- function(outcome, by, ct) {
  s <- as.data.frame(summary(ct, infer = TRUE, adjust = "none"))
  s$outcome <- outcome; s$by <- by
  res[[length(res) + 1]] <<- s
}
for (y in c("empathy", "helpfulness")) {
  m <- lmer(as.formula(paste(y, "~", rhs)), data = d, REML = TRUE)
  add(y, "all", contrast(emmeans(m, ~ policy), pairs))
  add(y, "urgency", contrast(emmeans(m, ~ policy | urgency), pairs))
}
ma <- glmer(as.formula(paste("appropriate ~", rhs)), data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
add("p_appropriate", "all", contrast(emmeans(ma, ~ policy), pairs))
add("p_appropriate", "urgency", contrast(emmeans(ma, ~ policy | urgency), pairs))
write_csv(as.data.frame(emmeans(ma, ~ policy | urgency, type = "response")), file.path(out, "natural_emm_p_appropriate.csv"))
r <- bind_rows(res)
r$p_holm <- p.adjust(r$p.value, "holm")
write_csv(r, file.path(out, "natural_rq2_contrasts.csv"))
print(as.data.frame(r %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)
