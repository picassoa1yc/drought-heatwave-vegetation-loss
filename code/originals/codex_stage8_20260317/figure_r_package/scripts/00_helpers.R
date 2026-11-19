get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep('^--file=', args, value = TRUE)
  if (length(hit) > 0) {
    return(normalizePath(sub('^--file=', '', hit[1]), winslash = '/', mustWork = TRUE))
  }
  return(NA_character_)
}

locate_package_root <- function() {
  script_path <- get_script_path()
  if (!is.na(script_path)) {
    script_dir <- dirname(script_path)
    if (basename(script_dir) == 'scripts') {
      return(dirname(script_dir))
    }
    return(script_dir)
  }

  candidates <- unique(c(
    getwd(),
    dirname(getwd()),
    normalizePath(file.path(getwd(), '..'), winslash = '/', mustWork = FALSE)
  ))

  for (cand in candidates) {
    if (dir.exists(file.path(cand, 'data')) && dir.exists(file.path(cand, 'scripts'))) {
      return(normalizePath(cand, winslash = '/', mustWork = TRUE))
    }
  }

  stop('Could not locate figure_r_package root. Run from the package root or scripts directory.')
}

stage8_paths <- local({
  pkg <- locate_package_root()
  list(
    pkg = pkg,
    data = Sys.getenv('VEGETATION_FIGURE5_INPUTS', file.path(pkg, 'data')),
    output = Sys.getenv('VEGETATION_FIGURE5_OUTPUTS', file.path(pkg, 'outputs')),
    scripts = file.path(pkg, 'scripts')
  )
})

load_stage8_packages <- function() {
  pkgs <- c('ggplot2', 'dplyr', 'tidyr', 'readr', 'forcats', 'stringr', 'patchwork', 'scales', 'viridis')
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop('Missing R packages: ', paste(missing, collapse = ', '))
  }
  invisible(lapply(pkgs, library, character.only = TRUE))
}

read_stage_csv <- function(name) {
  readr::read_csv(file.path(stage8_paths$data, name), show_col_types = FALSE)
}

grid_to_lonlat <- function(df, row_col = c('row', 'col'), res = 0.5) {
  row_name <- row_col[1]
  col_name <- row_col[2]
  df |>
    dplyr::mutate(
      dplyr::across(dplyr::all_of(c(row_name, col_name)), as.numeric),
      lon = -180 + res / 2 + (.data[[col_name]] - 1) * res,
      lat = 90 - res / 2 - (.data[[row_name]] - 1) * res
    )
}

theme_stage8 <- function() {
  ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.title = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = 'bold', size = 12),
      legend.title = ggplot2::element_text(face = 'bold'),
      legend.position = "bottom",
      strip.text = ggplot2::element_text(face = 'bold')
    )
}

save_stage8_plot <- function(plot_obj, filename, width = 14, height = 10, dpi = 320) {
  if (!dir.exists(stage8_paths$output)) {
    dir.create(stage8_paths$output, recursive = TRUE)
  }
  ggplot2::ggsave(
    filename = file.path(stage8_paths$output, filename),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi,
    bg = 'white'
  )
}

recode_event_type <- function(x) {
  dplyr::recode(
    x,
    'Mild_Reference' = 'Mild',
    'Compound_Overlapping' = 'Compound',
    'Heat_Dominant' = 'Heat',
    'Drought_Dominant' = 'Drought',
    'Sequential_Drought_Heat' = 'Drought->Heat',
    'Sequential_Heat_Drought' = 'Heat->Drought'
  )
}

recode_outcome <- function(x) {
  dplyr::recode(
    x,
    'LOSStot' = 'Total loss',
    'DECmax' = 'Max decline',
    'TIMErec_true' = 'Recovery time',
    'ndvi_cum_loss' = 'NDVI cumulative loss',
    'sif_cum_loss' = 'SIF cumulative loss',
    'gpp_cum_loss' = 'GPP cumulative loss',
    'et_cum_loss' = 'ET cumulative loss'
  )
}
