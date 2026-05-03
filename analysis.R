#!/usr/bin/env Rscript
# analysis.R — Run experiment + causal analysis for Adversarial MRTT
#
# With no arguments: runs python main.py --experiment-matrix, then analyses
# the output it produces.
#
# With an experiment matrix dir or timestamp: skips the Python run and analyses
# the existing outputs directly.
#
# Usage:
#   Rscript analysis.R                                  # run Python + analyse
#   Rscript analysis.R outputs/experiment_matrix_XYZ    # analyse existing dir
#   Rscript analysis.R 20260503_142301                  # timestamp shorthand

required_pkgs <- c("jsonlite", "ggplot2", "dplyr", "purrr", "scales")
missing_pkgs  <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(purrr)
  library(scales)
})

CONDITIONS <- c("DPG1_W0", "DPG1_W1", "DPG2_W0", "DPG2_W1")

# ---------------------------------------------------------------------------
# Resolve experiment matrix directory — run Python if none given
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  # Accept full path or bare timestamp (e.g. "20260503_142301")
  candidate <- args[1]
  if (!dir.exists(candidate)) {
    candidate <- file.path("outputs", paste0("experiment_matrix_", candidate))
  }
  if (!dir.exists(candidate)) stop("Directory not found: ", args[1])
  matrix_dir <- candidate
  message("Skipping Python run — using existing: ", matrix_dir)

} else {
  # Run the full experiment matrix via Python
  message("=== Running python main.py --experiment-matrix ===")
  py_cmd  <- Sys.which("python3")
  if (py_cmd == "") py_cmd <- Sys.which("python")
  if (py_cmd == "") stop("Cannot find python3 or python on PATH")

  py_out  <- system2(py_cmd, args = c("main.py", "--experiment-matrix"),
                     stdout = TRUE, stderr = TRUE)
  cat(py_out, sep = "\n")

  # Extract the output directory from the last "All outputs in: ..." line
  out_lines  <- grep("All outputs in:", py_out, value = TRUE)
  if (length(out_lines) == 0) stop("Could not find output directory in Python output")
  matrix_dir <- trimws(sub(".*All outputs in:\\s*", "", tail(out_lines, 1)))
  matrix_dir <- sub("/$", "", matrix_dir)   # strip trailing slash
  if (!dir.exists(matrix_dir)) stop("Python output dir not found: ", matrix_dir)
  message("Python run complete. Output: ", matrix_dir)
}

message("Analysing: ", matrix_dir)

# ---------------------------------------------------------------------------
# Load per-condition data
# ---------------------------------------------------------------------------
load_log <- function(cond) {
  log_path <- file.path(matrix_dir, cond, "game_log.csv")
  sum_path <- file.path(matrix_dir, cond, "summary.json")
  if (!file.exists(log_path)) { warning("Missing: ", log_path); return(NULL) }
  df        <- read.csv(log_path, stringsAsFactors = FALSE)
  meta      <- fromJSON(sum_path)
  df$condition <- cond
  df$ii_edge   <- meta$ii_edge
  df$aa_edge   <- meta$aa_edge
  df
}

load_summary <- function(cond) {
  p <- file.path(matrix_dir, cond, "summary.json")
  if (!file.exists(p)) { warning("Missing: ", p); return(NULL) }
  s            <- fromJSON(p)
  s$condition  <- cond
  s
}

all_dfs   <- set_names(map(CONDITIONS, load_log),     CONDITIONS)
summaries <- set_names(map(CONDITIONS, load_summary), CONDITIONS) |> compact()
combined  <- bind_rows(compact(all_dfs))

message(sprintf("Loaded %d/%d conditions, %d rows total",
                sum(!sapply(all_dfs, is.null)), length(CONDITIONS), nrow(combined)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

bootstrap_diff_ci <- function(a, b, n_boot = 5000, seed = 42) {
  set.seed(seed)
  diffs <- vapply(seq_len(n_boot), function(i) {
    mean(sample(b, length(b), replace = TRUE)) -
      mean(sample(a, length(a), replace = TRUE))
  }, numeric(1))
  list(
    point_est = mean(b) - mean(a),
    ci_lo     = as.numeric(quantile(diffs, 0.025)),
    ci_hi     = as.numeric(quantile(diffs, 0.975))
  )
}

# investor_cumulative is the running total; take the value at the last
# timestep of each episode (same across all agent rows in that timestep).
get_inv_cumul <- function(cond) {
  df <- all_dfs[[cond]]
  if (is.null(df)) return(NULL)
  df |>
    group_by(episode) |>
    filter(timestep == max(timestep)) |>
    slice(1) |>
    ungroup() |>
    pull(investor_cumulative)
}

n_ep <- if (length(summaries) > 0) summaries[[1]]$n_eval_episodes else "?"

# ---------------------------------------------------------------------------
# 1. 2x2 grid: mean repayment % by round
# ---------------------------------------------------------------------------
repay_by_round <- combined |>
  group_by(condition, ii_edge, aa_edge, timestep, agent_name) |>
  summarise(mean_repay = mean(repayment_pct, na.rm = TRUE), .groups = "drop") |>
  mutate(
    ii_label = factor(ii_edge, levels = c(0, 1),
                      labels = c("No Investor Link (ii=0)", "Investor Link (ii=1)")),
    aa_label = factor(aa_edge, levels = c(0, 1),
                      labels = c("No Agent Link (aa=0)", "Agent Link (aa=1)"))
  )

p_2x2 <- ggplot(repay_by_round,
                 aes(x = timestep, y = mean_repay,
                     colour = agent_name, group = agent_name)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  facet_grid(aa_label ~ ii_label) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(-0.05, 1.05)) +
  scale_colour_brewer(palette = "Set1", name = "Agent") +
  labs(
    title    = "2x2 Comparison: ii_edge x aa_edge",
    subtitle = sprintf("Mean Repayment %% by Round (%s eval episodes)", n_ep),
    x = "Round", y = "Mean Repayment %"
  ) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(file.path(matrix_dir, "comparison_2x2_grid.png"), p_2x2,
       width = 12, height = 8, dpi = 150)
message("Saved: comparison_2x2_grid.png")

# ---------------------------------------------------------------------------
# 2. Main effects table
# ---------------------------------------------------------------------------
main_eff_rows <- map_dfr(summaries, function(s) {
  map_dfr(names(s$agents), function(aname) {
    a <- s$agents[[aname]]
    fg <- if (!is.null(a$fair_gap_mean) && !is.na(a$fair_gap_mean))
            round(a$fair_gap_mean, 2) else NA_real_
    data.frame(
      Condition         = s$condition,
      ii_edge           = s$ii_edge,
      aa_edge           = s$aa_edge,
      Type              = toupper(a$policy),
      Agent             = aname,
      Investor_earnings = round(s$investor_earnings_mean, 2),
      Agent_earnings    = round(a$mean_reward, 2),
      FAIR_gap          = fg,
      stringsAsFactors  = FALSE
    )
  })
})

write.csv(main_eff_rows, file.path(matrix_dir, "main_effects_table.csv"), row.names = FALSE)
message("Saved: main_effects_table.csv")
cat("\nMain Effects Table:\n")
print(main_eff_rows, row.names = FALSE)

# ---------------------------------------------------------------------------
# 3. Effect decomposition with bootstrap 95% CIs
# ---------------------------------------------------------------------------
dpg1_w0 <- get_inv_cumul("DPG1_W0")
dpg1_w1 <- get_inv_cumul("DPG1_W1")
dpg2_w0 <- get_inv_cumul("DPG2_W0")
dpg2_w1 <- get_inv_cumul("DPG2_W1")

make_eff_row <- function(label, comp, a, b) {
  if (is.null(a) || is.null(b)) return(NULL)
  r <- bootstrap_diff_ci(a, b)
  data.frame(
    Effect     = label,
    Comparison = comp,
    Point_est  = round(r$point_est, 3),
    CI_lo_95   = round(r$ci_lo, 3),
    CI_hi_95   = round(r$ci_hi, 3),
    stringsAsFactors = FALSE
  )
}

effect_rows <- list(
  make_eff_row("ii effect (aa=0)", "DPG1_W1 - DPG1_W0", dpg1_w0, dpg1_w1),
  make_eff_row("ii effect (aa=1)", "DPG2_W1 - DPG2_W0", dpg2_w0, dpg2_w1),
  make_eff_row("aa effect (ii=0)", "DPG2_W0 - DPG1_W0", dpg1_w0, dpg2_w0),
  make_eff_row("aa effect (ii=1)", "DPG2_W1 - DPG1_W1", dpg1_w1, dpg2_w1)
)

# Interaction: (DPG2_W1 - DPG2_W0) - (DPG1_W1 - DPG1_W0)
if (!any(sapply(list(dpg1_w0, dpg1_w1, dpg2_w0, dpg2_w1), is.null))) {
  set.seed(43)
  n_boot <- 5000
  interactions <- vapply(seq_len(n_boot), function(i) {
    (mean(sample(dpg2_w1, length(dpg2_w1), replace = TRUE)) -
       mean(sample(dpg2_w0, length(dpg2_w0), replace = TRUE))) -
      (mean(sample(dpg1_w1, length(dpg1_w1), replace = TRUE)) -
         mean(sample(dpg1_w0, length(dpg1_w0), replace = TRUE)))
  }, numeric(1))
  obs_int <- (mean(dpg2_w1) - mean(dpg2_w0)) - (mean(dpg1_w1) - mean(dpg1_w0))
  effect_rows[[5]] <- data.frame(
    Effect     = "ii x aa interaction",
    Comparison = "(DPG2_W1-DPG2_W0) - (DPG1_W1-DPG1_W0)",
    Point_est  = round(obs_int, 3),
    CI_lo_95   = round(as.numeric(quantile(interactions, 0.025)), 3),
    CI_hi_95   = round(as.numeric(quantile(interactions, 0.975)), 3),
    stringsAsFactors = FALSE
  )
}

effects_df <- bind_rows(compact(effect_rows))
write.csv(effects_df, file.path(matrix_dir, "effect_decomposition.csv"), row.names = FALSE)
message("Saved: effect_decomposition.csv")
cat("\nEffect Decomposition:\n")
print(effects_df, row.names = FALSE)

# ---------------------------------------------------------------------------
# 4. Forest plot of effects
# ---------------------------------------------------------------------------
if (nrow(effects_df) > 0) {
  p_forest <- ggplot(effects_df,
                     aes(x = Point_est, y = reorder(Effect, Point_est))) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
    geom_errorbarh(aes(xmin = CI_lo_95, xmax = CI_hi_95),
                   height = 0.25, linewidth = 0.9, colour = "steelblue") +
    geom_point(size = 3.5, colour = "steelblue4") +
    labs(
      title    = "Effect Decomposition (95% Bootstrap CI)",
      subtitle = "Investor cumulative earnings: treatment minus control",
      x = "Difference in Investor Cumulative Earnings",
      y = NULL
    ) +
    theme_bw(base_size = 13)

  ggsave(file.path(matrix_dir, "effect_decomposition_forest.png"), p_forest,
         width = 10, height = 5, dpi = 150)
  message("Saved: effect_decomposition_forest.png")
}

# ---------------------------------------------------------------------------
# 5. EDA plots
# ---------------------------------------------------------------------------

# Investment distribution by condition
p_invest <- ggplot(combined, aes(x = investment, fill = condition)) +
  geom_histogram(binwidth = 1, alpha = 0.7, colour = "white") +
  facet_wrap(~condition, ncol = 2) +
  labs(title = "Investment Distribution by Condition",
       x = "Investment", y = "Count") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(matrix_dir, "eda_investment_dist.png"), p_invest,
       width = 10, height = 6, dpi = 150)

# Repayment % boxplot by condition
p_repay <- ggplot(combined, aes(x = condition, y = repayment_pct, fill = condition)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.6) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Repayment % by Condition",
       x = "Condition", y = "Repayment %") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(matrix_dir, "eda_repayment_boxplot.png"), p_repay,
       width = 8, height = 5, dpi = 150)

# Investor cumulative earnings violin by condition
inv_by_ep <- combined |>
  group_by(condition, ii_edge, aa_edge, episode) |>
  filter(timestep == max(timestep)) |>
  slice(1) |>
  ungroup()

p_violin <- ggplot(inv_by_ep,
                   aes(x = condition, y = investor_cumulative, fill = condition)) +
  geom_violin(alpha = 0.6, trim = FALSE) +
  geom_boxplot(width = 0.12, outlier.size = 0.5, fill = "white", alpha = 0.8) +
  labs(title = "Investor Cumulative Earnings by Condition",
       x = "Condition", y = "Cumulative Earnings") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(file.path(matrix_dir, "eda_investor_earnings_violin.png"), p_violin,
       width = 8, height = 5, dpi = 150)

# Mean repayment trend over rounds, all conditions overlaid with SE ribbon
repay_trend <- combined |>
  group_by(condition, timestep) |>
  summarise(
    mean_repay = mean(repayment_pct, na.rm = TRUE),
    se         = sd(repayment_pct, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

p_trend <- ggplot(repay_trend,
                  aes(x = timestep, y = mean_repay, colour = condition,
                      ymin = mean_repay - 1.96 * se,
                      ymax = mean_repay + 1.96 * se)) +
  geom_ribbon(aes(fill = condition), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_colour_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title    = "Mean Repayment % by Round and Condition",
    subtitle = "Shaded: +/- 1.96 SE",
    x = "Round", y = "Mean Repayment %",
    colour = "Condition", fill = "Condition"
  ) +
  theme_bw(base_size = 13)

ggsave(file.path(matrix_dir, "eda_repayment_trend.png"), p_trend,
       width = 10, height = 5, dpi = 150)

message("Saved: EDA plots (4 files)")

# ---------------------------------------------------------------------------
# 6. Markdown report
# ---------------------------------------------------------------------------
sig_str <- function(lo, hi) {
  if (lo > 0 || hi < 0) "CI excludes 0" else "CI includes 0 (not significant at 95%)"
}

get_eff <- function(eff_name) {
  r <- effects_df[effects_df$Effect == eff_name, ]
  if (nrow(r) == 0) return(NULL)
  r[1, ]
}

report <- c(
  "# Experiment Matrix Report",
  "",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "## Condition Summaries",
  ""
)

for (cond in names(summaries)) {
  s <- summaries[[cond]]
  report <- c(report,
    sprintf("### %s", cond),
    sprintf("- ii_edge=%d, aa_edge=%d", s$ii_edge, s$aa_edge),
    sprintf("- Investor earnings: %.2f +/- %.2f  (n=%d)",
            s$investor_earnings_mean, s$investor_earnings_std, s$n_eval_episodes)
  )
  for (aname in names(s$agents)) {
    a    <- s$agents[[aname]]
    fg_s <- if (!is.null(a$fair_gap_mean) && !is.na(a$fair_gap_mean))
               sprintf(", FAIR gap=%.2f", a$fair_gap_mean) else ""
    report <- c(report,
      sprintf("- %s (%s): reward=%.2f +/- %.2f%s",
              aname, a$policy, a$mean_reward, a$std_reward, fg_s)
    )
  }
  report <- c(report, "")
}

report <- c(report,
  "## Main Effects Table", "",
  "```",
  capture.output(print(main_eff_rows, row.names = FALSE)),
  "```",
  "",
  "## Effect Decomposition (95% Bootstrap CI)", ""
)

if (nrow(effects_df) > 0) {
  report <- c(report,
    "```",
    capture.output(print(effects_df, row.names = FALSE)),
    "```", ""
  )
}

report <- c(report, "## Observations", "")

r <- get_eff("ii effect (aa=0)")
report <- c(report, "### 1. Does ii_edge help at aa=0?")
if (!is.null(r)) {
  dir_w <- if (r$Point_est > 0) "increases" else "decreases"
  report <- c(report, sprintf(
    "ii_edge %s investor earnings by %+.2f [%.2f, %.2f]. %s.",
    dir_w, r$Point_est, r$CI_lo_95, r$CI_hi_95,
    sig_str(r$CI_lo_95, r$CI_hi_95)
  ))
}
report <- c(report, "")

r <- get_eff("aa effect (ii=0)")
report <- c(report, "### 2. Does aa_edge help at ii=0?")
if (!is.null(r)) {
  dir_w <- if (r$Point_est > 0) "increases" else "decreases"
  report <- c(report, sprintf(
    "aa_edge %s investor earnings by %+.2f [%.2f, %.2f]. %s.",
    dir_w, r$Point_est, r$CI_lo_95, r$CI_hi_95,
    sig_str(r$CI_lo_95, r$CI_hi_95)
  ))
}
report <- c(report, "")

r <- get_eff("ii x aa interaction")
report <- c(report, "### 3. Is there an ii x aa interaction?")
if (!is.null(r)) {
  amp <- if (r$Point_est > 0) "amplifies" else "dampens"
  report <- c(report, sprintf(
    "ii x aa interaction = %+.2f [%.2f, %.2f]. aa_edge %s the ii_edge effect. %s.",
    r$Point_est, r$CI_lo_95, r$CI_hi_95, amp,
    sig_str(r$CI_lo_95, r$CI_hi_95)
  ))
}
report <- c(report, "")

n_ep_val <- if (length(summaries) > 0) summaries[[1]]$n_eval_episodes else "?"
report <- c(report,
  "## Limitations", "",
  "- Single training run per condition; no error bars on DQN convergence.",
  sprintf("- %s eval episodes; CIs reflect sampling variance only.", n_ep_val),
  "- Fixed pairing i_k <-> a_k; no cross-pair allocation.",
  ""
)

writeLines(report, file.path(matrix_dir, "experiment_report.md"))
message("Saved: experiment_report.md")

message(sprintf("\nAll analysis outputs written to: %s/", matrix_dir))
