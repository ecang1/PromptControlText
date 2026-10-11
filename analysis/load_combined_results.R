# Load the complete compiled table; select an analysis study separately.
load_combined_results <- function(
    path,
    study = "all",
    include_unjudged = TRUE,
    include_provider_blocks = identical(study, "all")) {
  if (!study %in% c("all", "open_models", "frontier_main")) {
    stop("study must be all, open_models, or frontier_main.")
  }
  header <- names(readr::read_csv(path, n_max = 0, show_col_types = FALSE))
  types <- readr::cols(.default = readr::col_character())
  numeric_cols <- c(
    "refusal_score", "compliance_score", "clarification_score", "abstention_score",
    "hierarchy_following_score", "unsafe_continuation_score", "stop_compliance_score",
    "llm_confidence", "llm_num_agree"
  )
  numeric_cols <- c(numeric_cols, paste0("legacy_", c(
    "refusal_score", "compliance_score", "clarification_score", "abstention_score"
  )))
  for (column in intersect(numeric_cols, header)) {
    types$cols[[column]] <- readr::col_double()
  }
  # The notebook uses neither generated answer text nor JSON metadata. Keep
  # those in the CSV, but avoid allocating several GB of unused text in R.
  for (column in intersect(c(
      "model_output", "metadata", "score", "llm_evidence", "llm_reason",
      "base_system", "reference_json", "safety_heuristic_rules"), header)) {
    types$cols[[column]] <- readr::col_skip()
  }
  data <- readr::read_csv(path, col_types = types, show_col_types = FALSE)
  select_analysis_results(data, study, include_unjudged, include_provider_blocks)
}

select_analysis_results <- function(
    data,
    study = Sys.getenv("PROMPT_CONTROL_STUDY", unset = "open_models"),
    include_unjudged = TRUE,
    include_provider_blocks = FALSE) {
  if (!study %in% c("all", "open_models", "frontier_main")) {
    stop("study must be all, open_models, or frontier_main.")
  }
  if (!"analysis_study" %in% names(data)) {
    data$analysis_study <- "open_models"
  }
  if (study != "all") data <- data[data$analysis_study == study, , drop = FALSE]
  if (!nrow(data)) stop(paste("No rows for study", study))
  if (!include_provider_blocks && "judge_status" %in% names(data)) {
    blocked <- data$judge_status %in% "not_applicable_empty_provider_block"
    message(sum(blocked), " empty provider blocks excluded from response analysis.")
    data <- data[!blocked, , drop = FALSE]
  }
  # Keep legacy judge failures by default: their original heuristic scores still
  # belong in safety-rate denominators. Missing policy labels remain missing.
  if (!include_unjudged) {
    labels <- c("compliance", "refusal", "clarification", "hierarchy_preservation",
                "source_isolation", "safe_redirection")
    eligible <- data$llm_policy_label %in% labels
    message(sum(!eligible), " cases excluded from policy analysis (missing/invalid labels or empty provider blocks).")
    data <- data[eligible, , drop = FALSE]
  }
  message("Loaded ", nrow(data), " response cases for ", study, ".")
  data
}
