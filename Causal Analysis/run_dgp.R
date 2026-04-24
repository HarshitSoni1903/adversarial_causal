#!/usr/bin/env Rscript
# ==============================================================================
# run_dgp.R — MRTT Data Generation Driver
# ==============================================================================
#
# Place this file inside the MRTT Python project root. Run with:
#   Rscript run_dgp.R --n-seeds 7
#
# For each seed s in 1..n_seeds, runs W0 then W1 with seed s (paired design).
# Mutates config.yaml in place. Backup is YOUR responsibility.
#
# Collects training_returns.csv from each run into mrtt_dgp_output.csv.
# ==============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(dplyr)
  library(readr)
})

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

option_list <- list(
  make_option("--n-seeds", type = "integer", default = 7,
              help = "Number of paired seeds [default: %default]"),
  make_option("--seed-start", type = "integer", default = 1,
              help = "First seed value [default: %default]"),
  make_option("--config-name", type = "character", default = "config.yaml",
              help = "Config filename [default: %default]"),
  make_option("--output-csv", type = "character", default = "mrtt_dgp_output.csv",
              help = "Tidy output CSV path [default: %default]"),
  make_option("--python", type = "character", default = "python",
              help = "Python executable [default: %default]"),
  make_option("--dry-run", action = "store_true", default = FALSE,
              help = "Print what would run without executing"),
  make_option("--skip-existing", action = "store_true", default = FALSE,
              help = "Skip (seed, world) if a matching run folder already exists"),
  make_option("--time-budget-hours", type = "double", default = Inf,
              help = "Stop starting new runs after this many hours [default: Inf]")
)

opt <- parse_args(OptionParser(option_list = option_list))

# ------------------------------------------------------------------
# Paths (assumes script is IN the project root)
# ------------------------------------------------------------------

project_dir <- normalizePath(".", mustWork = TRUE)
cfg_pth     <- file.path(project_dir, opt$`config-name`)
outputs_dir <- file.path(project_dir, "outputs")

stopifnot(
  "config.yaml not found — is this the project root?" = file.exists(cfg_pth),
  "main.py not found — is this the project root?" = file.exists(file.path(project_dir, "main.py"))
)

base_cfg <- yaml::read_yaml(cfg_pth)

cat("========================================\n")
cat("MRTT DGP Driver\n")
cat("========================================\n")
cat(sprintf("  Seeds:      %d..%d (n=%d)\n",
            opt$`seed-start`,
            opt$`seed-start` + opt$`n-seeds` - 1,
            opt$`n-seeds`))
cat(sprintf("  Total runs: %d (paired W0/W1)\n", 2 * opt$`n-seeds`))
cat(sprintf("  Output:     %s\n", opt$`output-csv`))
if (is.finite(opt$`time-budget-hours`)) {
  cat(sprintf("  Budget:     %.2f hours (stops between seeds, not mid-run)\n",
              opt$`time-budget-hours`))
}
if (opt$`dry-run`) cat("  *** DRY RUN ***\n")
cat("========================================\n\n")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

write_cfg_for_run <- function(seed, communication) {
  cfg <- base_cfg
  cfg$seed <- seed
  cfg$world$communication <- communication
  yaml::write_yaml(cfg, cfg_pth)
}

find_run_folder <- function(world_tag, start_time) {
  if (!dir.exists(outputs_dir)) return(NA_character_)
  folders <- list.dirs(outputs_dir, recursive = FALSE)
  folders <- folders[grepl(sprintf("/%s_", world_tag), folders)]
  if (length(folders) == 0) return(NA_character_)
  info <- file.info(folders)
  info <- info[!is.na(info$ctime) & info$ctime >= start_time, , drop = FALSE]
  if (nrow(info) == 0) return(NA_character_)
  rownames(info)[which.max(info$ctime)]
}

find_existing_run <- function(seed, communication) {
  world_tag <- if (communication) "w1" else "w0"
  if (!dir.exists(outputs_dir)) return(NA_character_)
  candidates <- list.dirs(outputs_dir, recursive = FALSE)
  candidates <- candidates[grepl(sprintf("/%s_", world_tag), candidates)]
  matches <- Filter(function(f) {
    snap <- file.path(f, "config.yaml")
    if (!file.exists(snap)) return(FALSE)
    tryCatch({
      snap_cfg <- yaml::read_yaml(snap)
      isTRUE(snap_cfg$seed == seed) &&
        isTRUE(snap_cfg$world$communication == communication)
    }, error = function(e) FALSE)
  }, candidates)
  if (length(matches) == 0) return(NA_character_)
  matches[which.max(file.info(matches)$ctime)]
}

run_one <- function(seed, communication) {
  world_tag <- if (communication) "w1" else "w0"

  if (opt$`skip-existing`) {
    existing <- find_existing_run(seed, communication)
    if (!is.na(existing)) {
      cat(sprintf("[seed=%d %s] SKIP (exists): %s\n",
                  seed, toupper(world_tag), basename(existing)))
      return(list(ok = TRUE, folder = existing))
    }
  }

  cat(sprintf("[seed=%d %s] running python main.py\n", seed, toupper(world_tag)))

  if (opt$`dry-run`) return(list(ok = TRUE, folder = NA_character_))

  write_cfg_for_run(seed, communication)

  start_time <- Sys.time()
  status <- system2(opt$python, "main.py")

  if (status != 0) {
    cat(sprintf("  FAILED (exit=%d)\n", status))
    return(list(ok = FALSE, folder = NA_character_))
  }

  folder <- find_run_folder(world_tag, start_time - 1)  # -1s for fs clock skew
  if (is.na(folder)) {
    cat("  WARNING: run succeeded but output folder not found\n")
    return(list(ok = FALSE, folder = NA_character_))
  }
  cat(sprintf("  OK -> %s\n", basename(folder)))
  list(ok = TRUE, folder = folder)
}

load_training_returns <- function(folder, seed, world_tag) {
  csv_pth <- file.path(folder, "training_returns.csv")
  if (!file.exists(csv_pth)) {
    warning(sprintf("No training_returns.csv in %s", folder))
    return(NULL)
  }
  df <- readr::read_csv(csv_pth, show_col_types = FALSE)
  df$seed   <- seed
  df$world  <- ifelse(world_tag == "w1", 1L, 0L)
  df$run_id <- basename(folder)
  df
}

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

out_pth <- file.path(project_dir, opt$`output-csv`)

# Helper: write current accumulated rows to the output CSV.
# Called after each seed pair so a crash/interrupt doesn't lose progress.
write_tidy <- function(rows_list) {
  if (length(rows_list) == 0) return(invisible(NULL))
  tidy <- dplyr::bind_rows(rows_list)
  meta_cols  <- c("seed", "world", "run_id", "episode")
  other_cols <- setdiff(names(tidy), meta_cols)
  tidy <- tidy[, c(meta_cols, other_cols)]
  readr::write_csv(tidy, out_pth)
  invisible(tidy)
}

all_rows <- list()
seeds    <- seq(opt$`seed-start`, length.out = opt$`n-seeds`)
n_failed <- 0L
t_start  <- Sys.time()
budget_secs <- opt$`time-budget-hours` * 3600
budget_hit  <- FALSE

for (seed in seeds) {
  # Check time budget at the start of each seed pair (not mid-pair, to keep
  # W0/W1 pairing intact). A partially-finished pair would be unusable.
  elapsed_secs <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  if (elapsed_secs >= budget_secs) {
    cat(sprintf("\n*** Time budget reached (%.2f hrs elapsed). Stopping before seed %d. ***\n",
                elapsed_secs / 3600, seed))
    budget_hit <- TRUE
    break
  }
  if (is.finite(budget_secs)) {
    cat(sprintf("[budget: %.2f / %.2f hrs elapsed]\n",
                elapsed_secs / 3600, budget_secs / 3600))
  }

  for (comm in c(FALSE, TRUE)) {
    world_tag <- if (comm) "w1" else "w0"
    res <- run_one(seed, comm)
    if (!res$ok) { n_failed <- n_failed + 1L; next }
    if (opt$`dry-run`) next
    rows <- load_training_returns(res$folder, seed, world_tag)
    if (!is.null(rows)) all_rows[[length(all_rows) + 1L]] <- rows
  }

  # Incremental save after each seed pair (both W0 and W1 done)
  if (!opt$`dry-run`) write_tidy(all_rows)
}

if (opt$`dry-run`) {
  cat("\n[dry-run] No simulations executed, no output written.\n")
  quit(status = 0)
}

if (length(all_rows) == 0) {
  stop("No runs produced training_returns.csv; nothing to write.")
}

# Final write (may be redundant with incremental save, but ensures consistency)
tidy <- write_tidy(all_rows)

cat(sprintf("\n========================================\n"))
if (budget_hit) {
  cat("Stopped early due to time budget.\n")
}
cat(sprintf("Done. %d runs ok, %d failed.\n",
            length(all_rows), n_failed))
cat(sprintf("Wall clock: %.2f hours\n",
            as.numeric(difftime(Sys.time(), t_start, units = "hours"))))
cat(sprintf("Tidy output: %s (%d rows)\n", out_pth, nrow(tidy)))
cat("========================================\n")