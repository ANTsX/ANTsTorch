#!/usr/bin/env Rscript

# Sanity check: marginal Gaussianity + decorrelation
# --------------------------------------------------
# Inputs:
#   --raw path/to/raw.csv
#   --z   path/to/z1.csv --z path/to/z2.csv ...   # repeatable
# Options:
#   --labels-z "Label1,Label2,..."  # optional labels matching --z order
#   --by-meta                       # aggregate z's by PKG and K (averaging seeds)
#   --outdir sanity_outputs         # output directory
#
# Output files:
#   sanity_summary.csv
#   sanity_bars_gaussianity.png
#   sanity_bars_decorrelation.png
#
# Notes:
#   * Non-numeric columns in CSVs are dropped automatically.
#   * KS p-values are very sensitive with large N; use them for *relative* comparisons.

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(moments)
  library(ggplot2)
  library(purrr)
  library(patchwork)  # for combining plots
})

# ---------- helpers ----------
read_numeric_matrix <- function(path) {
  df <- suppressMessages(readr::read_csv(path, guess_max = 10000, show_col_types = FALSE))
  num <- dplyr::select(df, where(is.numeric))
  as.matrix(num)
}

gaussianity_metrics <- function(M) {
  if (!is.matrix(M) || ncol(M) == 0L) {
    return(list(n_features = 0L, mean_abs_skew = NA_real_, mean_abs_exkurt = NA_real_, frac_ks_p_gt_0.05 = NA_real_))
  }
  # Standardize per column
  Mz <- scale(M)
  # Drop zero-variance or all-NA columns
  sds <- apply(M, 2, function(x) sd(x, na.rm = TRUE))
  keep <- which(is.finite(sds) & sds > 0)
  if (length(keep) == 0) {
    return(list(n_features = 0L, mean_abs_skew = NA_real_, mean_abs_exkurt = NA_real_, frac_ks_p_gt_0.05 = NA_real_))
  }
  Z <- Mz[, keep, drop = FALSE]

  sk <- apply(Z, 2, moments::skewness, na.rm = TRUE)
  ek <- apply(Z, 2, moments::kurtosis, na.rm = TRUE) - 3

  ks_fun <- function(v) {
    v <- v[is.finite(v)]
    if (length(v) < 100) return(NA_real_)
    suppressWarnings(ks.test(v, "pnorm")$p.value)
  }
  ks_p <- apply(Z, 2, ks_fun)

  list(
    n_features = ncol(Z),
    mean_abs_skew = mean(abs(sk), na.rm = TRUE),
    mean_abs_exkurt = mean(abs(ek), na.rm = TRUE),
    frac_ks_p_gt_0.05 = mean(ks_p > 0.05, na.rm = TRUE)
  )
}

decorrelation_metrics <- function(M) {
  if (!is.matrix(M) || ncol(M) < 2L) {
    return(list(avg_abs_corr = NA_real_, cond_number = NA_real_, pc1_var_frac = NA_real_))
  }
  C <- suppressWarnings(cor(M, use = "pairwise.complete.obs"))
  if (!is.matrix(C) || any(!is.finite(C))) {
    return(list(avg_abs_corr = NA_real_, cond_number = NA_real_, pc1_var_frac = NA_real_))
  }
  avg_abs_corr <- mean(abs(C[upper.tri(C)]))

  S <- suppressWarnings(cov(M, use = "pairwise.complete.obs"))
  p <- ncol(M)
  eps <- 1e-6
  S_r <- S + diag(eps, p)
  ev <- suppressWarnings(eigen(S_r, symmetric = TRUE, only.values = TRUE)$values)
  ev <- ev[is.finite(ev) & ev > 0]
  if (length(ev) < 2) {
    return(list(avg_abs_corr = avg_abs_corr, cond_number = NA_real_, pc1_var_frac = NA_real_))
  }
  list(
    avg_abs_corr = avg_abs_corr,
    cond_number = max(ev) / min(ev),
    pc1_var_frac = max(ev) / sum(ev)
  )
}

summarize_matrix <- function(path, label) {
  M <- read_numeric_matrix(path)
  message(sprintf("Loaded '%s' as numeric matrix: %d rows x %d cols", path, nrow(M), ncol(M)))
  g <- gaussianity_metrics(M)
  d <- decorrelation_metrics(M)
  tibble::tibble(
    label = label,
    path = path,
    n_features = g$n_features,
    mean_abs_skew = g$mean_abs_skew,
    mean_abs_exkurt = g$mean_abs_exkurt,
    frac_ks_p_gt_0.05 = g$frac_ks_p_gt_0.05,
    avg_abs_corr = d$avg_abs_corr,
    cond_number = d$cond_number,
    pc1_var_frac = d$pc1_var_frac
  )
}


parse_meta <- function(paths) {
  tibble::tibble(path = paths) |>
    mutate(
      PKG  = str_match(path, "PKG=([^/]+)")[,2],
      K    = suppressWarnings(as.integer(str_match(path, "K=([0-9]+)")[,2])),
      seed = suppressWarnings(as.integer(str_match(path, "seed=([0-9]+)")[,2]))
    )
}

# ---------- CLI ----------
option_list <- list(
  make_option(c("--raw"), type="character", help="Path to RAW features CSV", metavar="file"),
  make_option(c("--z"), type="character", action="append",
              help="Path to a transformed Z CSV (repeat --z for multiple).", metavar="file"),
  make_option(c("--outdir"), type="character", default="sanity_outputs",
              help="Output directory for CSV summaries and plots [default %default]"),
  make_option(c("--labelraw"), type="character", default="raw",
              help="Label to use for raw matrix [default %default]"),
  make_option(c("--labelsz"), type="character", default="z",
              help="Comma-separated labels matching the order of --z entries (optional)"),
  make_option(c("--by-meta"), action="store_true", default=FALSE,
              help="Aggregate z metrics by parsed PKG and K (averaging over seeds).")
)

args <- parse_args(OptionParser(option_list=option_list))

z_paths <- args$z

# sanity checks on files
if (!file.exists(args$raw)) {
  stop("RAW file does not exist: ", args$raw)
}
missing_z <- z_paths[!file.exists(z_paths)]
if (length(missing_z) > 0) {
  stop("These Z files do not exist:\n  - ", paste(missing_z, collapse = "\n  - "))
}

dir.create(args$outdir, showWarnings = FALSE, recursive = TRUE)

# --- Safe labels (fallback to file basenames if not provided) ---
safe_label <- function(arg_label, path) {
  if (is.null(arg_label) || is.na(arg_label) || identical(arg_label, "")) {
    basename(path)
  } else {
    arg_label
  }
}

label_raw <- safe_label(args$labelraw, args$raw)
if( length( args$z ) == 1 ) {
  label_z <- safe_label(args$labelsz, args$z)
  z_labels <- label_z
} else { 
  # Labels for z's
  if (!is.null(args$labels_z)) {
    z_labels <- strsplit(args$labels_z, ",")[[1]]
    if (length(z_labels) != length(z_paths)) {
      stop("--labels-z length must match number of --z entries.")
    }
  } else {
    z_labels <- basename(z_paths)
  }
}

# Summaries
raw_sum <- summarize_matrix(args$raw, label_raw)
z_summaries <- map2_dfr(z_paths, z_labels, summarize_matrix)

# Optional: aggregate by PKG,K across seeds
if (isTRUE(args$by_meta)) {
  meta <- parse_meta(z_paths)
  z_summaries <- z_summaries |>
    left_join(meta, by = "path") |>
    group_by(PKG, K) |>
    summarise(
      n_features = mean(n_features, na.rm = TRUE),
      mean_abs_skew = mean(mean_abs_skew, na.rm = TRUE),
      mean_abs_exkurt = mean(mean_abs_exkurt, na.rm = TRUE),
      frac_ks_p_gt_0.05 = mean(frac_ks_p_gt_0.05, na.rm = TRUE),
      avg_abs_corr = mean(avg_abs_corr, na.rm = TRUE),
      cond_number = mean(cond_number, na.rm = TRUE),
      pc1_var_frac = mean(pc1_var_frac, na.rm = TRUE),
      n_runs = dplyr::n(),
      .groups = "drop"
    ) |>
    mutate(
      path = paste0("META:PKG=", PKG, ";K=", K),
      label = paste0("flow(PKG=", PKG, ",K=", K, ",n=", n_runs, ")")
    )
}

# Combine and write
combined <- bind_rows(raw_sum, z_summaries)
readr::write_csv(combined, file.path(args$outdir, "sanity_summary.csv"))

if (nrow(combined) == 0) {
  stop("No rows to plot. Check that the CSVs have numeric columns and are read correctly.")
}
if (!"label" %in% names(combined)) {
  stop("Internal error: 'combined' has no 'label' column.")
}

plot_df <- combined
# default everything to "flow"
plot_df$type <- rep("flow", nrow(plot_df))

# mark the raw row(s) explicitly if present
raw_idx <- which(!is.na(plot_df$label) & plot_df$label == args$label_raw)
if (length(raw_idx) > 0) {
  plot_df$type[raw_idx] <- "raw"
}


numfmt <- scales::label_number(accuracy = 0.001)

p1 <- ggplot(plot_df, aes(x = label, y = mean_abs_skew, fill = type)) +
  geom_col() + labs(title="Mean |skew| (lower better)", x=NULL, y=NULL) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

p2 <- ggplot(plot_df, aes(x = label, y = mean_abs_exkurt, fill = type)) +
  geom_col() + labs(title="Mean |excess kurtosis| (lower better)", x=NULL, y=NULL) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

p3 <- ggplot(plot_df, aes(x = label, y = frac_ks_p_gt_0.05, fill = type)) +
  geom_col() + labs(title="Frac(KS p>0.05) (higher better)", x=NULL, y=NULL) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

p4 <- ggplot(plot_df, aes(x = label, y = avg_abs_corr, fill = type)) +
  geom_col() + labs(title="Mean |corr| offdiag (lower better)", x=NULL, y=NULL) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

p5 <- ggplot(plot_df, aes(x = label, y = cond_number, fill = type)) +
  geom_col() + labs(title="Cov condition number (lower better)", x=NULL, y=NULL) +
  scale_y_log10(labels = numfmt) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

p6 <- ggplot(plot_df, aes(x = label, y = pc1_var_frac, fill = type)) +
  geom_col() + labs(title="PC1 variance fraction (lower better)", x=NULL, y=NULL) +
  theme_bw(base_size = 12) + theme(legend.position = "none", axis.text.x = element_text(angle=20, hjust=1))

ggsave(file.path(args$outdir, "sanity_bars_gaussianity.png"),
       plot = (p1 | p2 | p3), width = 12, height = 4, dpi = 150, limitsize = FALSE)
ggsave(file.path(args$outdir, "sanity_bars_decorrelation.png"),
       plot = (p4 | p5 | p6), width = 12, height = 4, dpi = 150, limitsize = FALSE)

message("Wrote: ", file.path(args$outdir, "sanity_summary.csv"))
message("Wrote: ",
        file.path(args$outdir, "sanity_bars_gaussianity.png"), " and ",
        file.path(args$outdir, "sanity_bars_decorrelation.png"))
