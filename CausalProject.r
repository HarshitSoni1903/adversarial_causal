############################################################
# Simulation Results Script
# Project: Adversarial Communication and Investor Return
#
# This script:
# 1. Reads the four simulation CSV files
# 2. Converts round-level logs into episode-level investor returns
# 3. Computes cell means for the 2x2 design
# 4. Estimates treatment effects within each DGP
# 5. Uses bootstrap resampling to approximate repeated simulation iterations
# 6. Computes bias, RMSE, and 95% confidence interval coverage
# 7. Saves clean CSV tables and figures to the Desktop
############################################################

library(tidyverse)
library(knitr)

set.seed(123)

############################################################
# 1. User settings
############################################################

data_dir <- "~/Desktop"

output_dir <- file.path(data_dir, "simulation_results_outputs")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

B <- 1000

############################################################
# 2. Define the four simulation cells
#
# File naming convention:
# 1_0.csv = DGP 1, Control
# 1_1.csv = DGP 1, Treatment
# 2_0.csv = DGP 2, Control
# 2_1.csv = DGP 2, Treatment
#
# e_ii = investor-investor communication
# e_aa = adversary-adversary communication
#
# DGP 1: e_ii = 0, no investor communication
# DGP 2: e_ii = 1, investor communication
#
# Control:   e_aa = 0, no adversary communication
# Treatment: e_aa = 1, adversary communication
############################################################

files <- tibble(
  file = c("1_0.csv", "1_1.csv", "2_0.csv", "2_1.csv"),
  e_ii = c(0, 0, 1, 1),
  e_aa = c(0, 1, 0, 1),
  condition = c(
    "DGP 1 + Control",
    "DGP 1 + Treatment",
    "DGP 2 + Control",
    "DGP 2 + Treatment"
  )
)

############################################################
# 3. Function to read one CSV and collapse it to episode level
#
# Expected columns in each CSV:
# - episode
# - dyad_idx
# - timestep
# - dyad_investor_cumulative
#
# For each episode:
# - take the final timestep for each dyad
# - average investor cumulative return across the two dyads
#
# This gives one outcome Y per episode:
# Y = (Y_1 + Y_2) / 2
############################################################

read_cell <- function(file, e_ii, e_aa, condition) {
  path <- file.path(data_dir, file)
  
  if (!file.exists(path)) {
    stop(paste("Could not find file:", path))
  }
  
  df <- read_csv(path, show_col_types = FALSE)
  
  required_cols <- c(
    "episode",
    "dyad_idx",
    "timestep",
    "dyad_investor_cumulative"
  )
  
  missing_cols <- setdiff(required_cols, names(df))
  
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "File",
        file,
        "is missing required columns:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }
  
  episode_outcomes <- df %>%
    group_by(episode, dyad_idx) %>%
    filter(timestep == max(timestep, na.rm = TRUE)) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    group_by(episode) %>%
    summarise(
      Y = mean(dyad_investor_cumulative, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      e_ii = e_ii,
      e_aa = e_aa,
      condition = condition
    )
  
  return(episode_outcomes)
}

############################################################
# 4. Read all four files
############################################################

dat <- pmap_dfr(files, read_cell)

write_csv(dat, file.path(output_dir, "episode_level_outcomes.csv"))

############################################################
# 5. Compute cell means
############################################################

cell_means <- dat %>%
  group_by(condition, e_ii, e_aa) %>%
  summarise(
    n = n(),
    mean_Y = mean(Y, na.rm = TRUE),
    sd_Y = sd(Y, na.rm = TRUE),
    se_Y = sd_Y / sqrt(n),
    .groups = "drop"
  ) %>%
  arrange(e_ii, e_aa)

print(cell_means)

write_csv(cell_means, file.path(output_dir, "cell_means.csv"))

############################################################
# 6. Estimate treatment effects within each DGP
#
# tau_0 = E[Y(0,1) - Y(0,0)]
# tau_1 = E[Y(1,1) - Y(1,0)]
#
# First argument = investor communication e_ii
# Second argument = adversary communication e_aa
############################################################

tau_hat <- cell_means %>%
  select(e_ii, e_aa, mean_Y) %>%
  pivot_wider(
    names_from = e_aa,
    values_from = mean_Y,
    names_prefix = "aa_"
  ) %>%
  mutate(
    tau = aa_1 - aa_0,
    dgp = if_else(
      e_ii == 0,
      "DGP 1: No investor communication",
      "DGP 2: Investor communication"
    ),
    sutva_status = if_else(
      e_ii == 0,
      "Fulfilled",
      "Violated"
    )
  ) %>%
  select(dgp, e_ii, sutva_status, tau)

print(tau_hat)

write_csv(tau_hat, file.path(output_dir, "tau_estimates.csv"))

delta_hat <- tau_hat$tau[tau_hat$e_ii == 1] -
  tau_hat$tau[tau_hat$e_ii == 0]

delta_table <- tibble(
  quantity = "Delta = tau_1 - tau_0",
  estimate = delta_hat
)

print(delta_table)

write_csv(delta_table, file.path(output_dir, "delta_estimate.csv"))

############################################################
# 7. Define simulation truth
#
# Since we have one large batch of 1000 episodes per cell,
# we treat the full-sample tau estimates as the approximate
# simulation estimands.
#
# Then we use bootstrap resampling to approximate repeated
# simulation iterations.
############################################################

truth <- tau_hat %>%
  rename(tau_truth = tau) %>%
  select(dgp, e_ii, sutva_status, tau_truth)

print(truth)

############################################################
# 8. Bootstrap function
#
# For each bootstrap iteration:
# - resample episodes within each of the four cells
# - recompute cell means
# - estimate tau within each DGP
# - compute a normal-approximation 95% CI
############################################################

one_boot <- function(b, dat) {
  boot_dat <- dat %>%
    group_by(e_ii, e_aa) %>%
    slice_sample(prop = 1, replace = TRUE) %>%
    ungroup()
  
  cell_stats <- boot_dat %>%
    group_by(e_ii, e_aa) %>%
    summarise(
      mean_Y = mean(Y, na.rm = TRUE),
      var_Y = var(Y, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )
  
  out <- cell_stats %>%
    select(e_ii, e_aa, mean_Y, var_Y, n) %>%
    pivot_wider(
      names_from = e_aa,
      values_from = c(mean_Y, var_Y, n),
      names_sep = "_aa"
    ) %>%
    mutate(
      iter = b,
      tau_hat = mean_Y_aa1 - mean_Y_aa0,
      se_hat = sqrt(var_Y_aa1 / n_aa1 + var_Y_aa0 / n_aa0),
      ci_low = tau_hat - 1.96 * se_hat,
      ci_high = tau_hat + 1.96 * se_hat,
      dgp = if_else(
        e_ii == 0,
        "DGP 1: No investor communication",
        "DGP 2: Investor communication"
      ),
      sutva_status = if_else(
        e_ii == 0,
        "Fulfilled",
        "Violated"
      )
    ) %>%
    select(
      iter,
      dgp,
      e_ii,
      sutva_status,
      tau_hat,
      se_hat,
      ci_low,
      ci_high
    )
  
  return(out)
}

############################################################
# 9. Run bootstrap iterations
############################################################

boot_results <- map_dfr(1:B, one_boot, dat = dat)

write_csv(boot_results, file.path(output_dir, "bootstrap_estimates.csv"))

############################################################
# 10. Compute bias, RMSE, and CI coverage
#
# Bias:
# mean(tau_hat - tau_truth)
#
# RMSE:
# sqrt(mean((tau_hat - tau_truth)^2))
#
# Coverage:
# percent of bootstrap CIs containing tau_truth
############################################################

boot_eval <- boot_results %>%
  left_join(truth, by = c("dgp", "e_ii", "sutva_status")) %>%
  mutate(
    error = tau_hat - tau_truth,
    covered = ci_low <= tau_truth & ci_high >= tau_truth
  )

write_csv(boot_eval, file.path(output_dir, "bootstrap_iterations_with_truth.csv"))

performance_table <- boot_eval %>%
  group_by(dgp, e_ii, sutva_status) %>%
  summarise(
    estimand = first(tau_truth),
    mean_estimate = mean(tau_hat, na.rm = TRUE),
    bias = mean(error, na.rm = TRUE),
    rmse = sqrt(mean(error^2, na.rm = TRUE)),
    coverage_95 = mean(covered, na.rm = TRUE),
    mean_se = mean(se_hat, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(e_ii)

print(performance_table)

write_csv(performance_table, file.path(output_dir, "simulation_performance.csv"))

############################################################
# 11. Create LaTeX table for performance results
############################################################

performance_latex <- performance_table %>%
  mutate(
    estimand = round(estimand, 3),
    mean_estimate = round(mean_estimate, 3),
    bias = round(bias, 3),
    rmse = round(rmse, 3),
    coverage_95 = paste0(round(100 * coverage_95, 1), "\\%"),
    mean_se = round(mean_se, 3)
  ) %>%
  select(
    dgp,
    sutva_status,
    estimand,
    mean_estimate,
    bias,
    rmse,
    coverage_95,
    mean_se
  ) %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    caption = "Simulation performance of the difference-in-means estimator.",
    col.names = c(
      "DGP",
      "SUTVA status",
      "Estimand",
      "Mean estimate",
      "Bias",
      "RMSE",
      "95\\% CI coverage",
      "Mean SE"
    )
  )

cat(performance_latex)

writeLines(
  performance_latex,
  con = file.path(output_dir, "simulation_performance_table.tex")
)

############################################################
# 12. Create LaTeX table for cell means
############################################################

cell_means_latex <- cell_means %>%
  mutate(
    mean_Y = round(mean_Y, 3),
    sd_Y = round(sd_Y, 3),
    se_Y = round(se_Y, 3)
  ) %>%
  select(condition, e_ii, e_aa, n, mean_Y, sd_Y, se_Y) %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    caption = "Mean investor return by communication condition.",
    col.names = c(
      "Condition",
      "$e_{ii}$",
      "$e_{aa}$",
      "N",
      "Mean investor return",
      "SD",
      "SE"
    )
  )

cat(cell_means_latex)

writeLines(
  cell_means_latex,
  con = file.path(output_dir, "cell_means_table.tex")
)

############################################################
# 13. Figure 1: Investor return by communication condition
############################################################

p1 <- ggplot(dat, aes(x = factor(e_aa), y = Y)) +
  geom_boxplot(outlier.alpha = 0.25) +
  facet_wrap(
    ~ e_ii,
    labeller = labeller(
      e_ii = c(
        "0" = "DGP 1: No investor communication",
        "1" = "DGP 2: Investor communication"
      )
    )
  ) +
  labs(
    x = "Adversary communication condition",
    y = "Episode-level investor return",
    title = "Investor return by adversary communication condition",
    subtitle = "Outcome is average investor return across the two dyads"
  ) +
  scale_x_discrete(
    labels = c(
      "0" = "Control\ne_aa = 0",
      "1" = "Treatment\ne_aa = 1"
    )
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(output_dir, "fig_1_cell_distributions.png"),
  plot = p1,
  width = 8,
  height = 5,
  dpi = 300
)

############################################################
# 14. Figure 2: Bootstrap distribution of treatment effects
############################################################

p2 <- boot_eval %>%
  ggplot(aes(x = tau_hat)) +
  geom_histogram(bins = 40, alpha = 0.75) +
  geom_vline(
    aes(xintercept = tau_truth),
    linetype = "dashed",
    linewidth = 1
  ) +
  facet_wrap(~ dgp, scales = "free") +
  labs(
    x = "Estimated treatment effect",
    y = "Bootstrap iterations",
    title = "Distribution of estimated treatment effects",
    subtitle = "Dashed line shows the approximate simulation estimand"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(output_dir, "fig_2_tau_bootstrap_distributions.png"),
  plot = p2,
  width = 8,
  height = 5,
  dpi = 300
)

############################################################
# 15. Figure 3: Bias, RMSE, and coverage
############################################################

performance_long <- performance_table %>%
  transmute(
    dgp,
    sutva_status,
    Bias = bias,
    RMSE = rmse,
    `95% CI Coverage` = coverage_95
  ) %>%
  pivot_longer(
    cols = c(Bias, RMSE, `95% CI Coverage`),
    names_to = "metric",
    values_to = "value"
  )

p3 <- ggplot(performance_long, aes(x = dgp, y = value)) +
  geom_col() +
  facet_wrap(~ metric, scales = "free_y") +
  labs(
    x = NULL,
    y = NULL,
    title = "Estimator performance across data-generating processes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave(
  filename = file.path(output_dir, "fig_3_estimator_performance.png"),
  plot = p3,
  width = 9,
  height = 5,
  dpi = 300
)

############################################################
# 16. Figure 4: Cell means with standard errors
############################################################

p4 <- ggplot(
  cell_means,
  aes(
    x = factor(e_aa),
    y = mean_Y,
    ymin = mean_Y - 1.96 * se_Y,
    ymax = mean_Y + 1.96 * se_Y
  )
) +
  geom_col(alpha = 0.75) +
  geom_errorbar(width = 0.15) +
  facet_wrap(
    ~ e_ii,
    labeller = labeller(
      e_ii = c(
        "0" = "DGP 1: No investor communication",
        "1" = "DGP 2: Investor communication"
      )
    )
  ) +
  labs(
    x = "Adversary communication condition",
    y = "Mean investor return",
    title = "Mean investor return by cell",
    subtitle = "Error bars show approximate 95% confidence intervals for cell means"
  ) +
  scale_x_discrete(
    labels = c(
      "0" = "Control\ne_aa = 0",
      "1" = "Treatment\ne_aa = 1"
    )
  ) +
  theme_minimal(base_size = 12)

ggsave(
  filename = file.path(output_dir, "fig_4_cell_means.png"),
  plot = p4,
  width = 8,
  height = 5,
  dpi = 300
)

############################################################
# 17. Print final summary
############################################################

cat("\n\n============================================================\n")
cat("Simulation results complete.\n")
cat("Outputs saved to:\n")
cat(output_dir, "\n")
cat("============================================================\n\n")

cat("Estimated treatment effects:\n")
print(tau_hat)

cat("\nDifference in treatment effects:\n")
print(delta_table)

cat("\nPerformance table:\n")
print(performance_table)

cat("\nFiles created:\n")
cat("- episode_level_outcomes.csv\n")
cat("- cell_means.csv\n")
cat("- tau_estimates.csv\n")
cat("- delta_estimate.csv\n")
cat("- bootstrap_estimates.csv\n")
cat("- bootstrap_iterations_with_truth.csv\n")
cat("- simulation_performance.csv\n")
cat("- simulation_performance_table.tex\n")
cat("- cell_means_table.tex\n")
cat("- fig_1_cell_distributions.png\n")
cat("- fig_2_tau_bootstrap_distributions.png\n")
cat("- fig_3_estimator_performance.png\n")
cat("- fig_4_cell_means.png\n")
cat("============================================================\n")




ci_bounds <- boot_eval %>%
  group_by(dgp, e_ii, tau_truth) %>%
  summarise(
    q025 = quantile(tau_hat, 0.025),
    q975 = quantile(tau_hat, 0.975),
    .groups = "drop"
  ) %>%
  mutate(
    dgp_label = if_else(
      e_ii == 0,
      "DGP 1: No investor communication\nSUTVA fulfilled",
      "DGP 2: Investor communication\nSUTVA violated"
    )
  )

p_ci_hist_shaded <- boot_eval %>%
  mutate(
    dgp_label = if_else(
      e_ii == 0,
      "DGP 1: No investor communication\nSUTVA fulfilled",
      "DGP 2: Investor communication\nSUTVA violated"
    )
  ) %>%
  ggplot(aes(x = tau_hat)) +
  geom_rect(
    data = ci_bounds,
    aes(xmin = q025, xmax = q975, ymin = -Inf, ymax = Inf),
    inherit.aes = FALSE,
    alpha = 0.15
  ) +
  geom_histogram(
    bins = 40,
    alpha = 0.75,
    color = "white"
  ) +
  geom_vline(
    data = ci_bounds,
    aes(xintercept = tau_truth),
    linetype = "dashed",
    linewidth = 1
  ) +
  geom_vline(
    data = ci_bounds,
    aes(xintercept = q025),
    linetype = "dotted",
    linewidth = 0.8
  ) +
  geom_vline(
    data = ci_bounds,
    aes(xintercept = q975),
    linetype = "dotted",
    linewidth = 0.8
  ) +
  facet_grid(dgp_label ~ ., scales = "free_y") +
  labs(
    title = "Bootstrap confidence interval distribution",
    subtitle = "Shaded region = central 95% bootstrap interval; dashed line = approximate estimand",
    x = "Estimated treatment effect",
    y = "Bootstrap count"
  ) +
  theme_minimal(base_size = 12)

print(p_ci_hist_shaded)

ggsave(
  filename = file.path(output_dir, "fig_ci_histograms_stacked_shaded.png"),
  plot = p_ci_hist_shaded,
  width = 8,
  height = 7,
  dpi = 300
)