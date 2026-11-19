helper_candidates <- c(
  file.path(getwd(), 'scripts', '00_helpers.R'),
  file.path(getwd(), '00_helpers.R'),
  if (any(grepl('^--file=', commandArgs(FALSE)))) file.path(dirname(normalizePath(sub('^--file=', '', commandArgs(FALSE)[grep('^--file=', commandArgs(FALSE))][1]))), '00_helpers.R') else NA_character_
)
helper_path <- helper_candidates[file.exists(helper_candidates)][1]
if (is.na(helper_path) || length(helper_path) == 0) {
  stop('Could not locate 00_helpers.R')
}
source(helper_path)
load_stage8_packages()

stage_fill <- c('Pre-event' = '#e9f0f7', 'Event' = '#fce8dc', 'Post-event' = '#eef4e3')
event_palette <- c(
  'Compound' = '#542788',
  'Drought' = '#b35806',
  'Heat' = '#d7301f',
  'Drought->Heat' = '#1b7837'
)

build_relative_month <- function(df, event_max = 6, pre_max = 3, post_max = 3) {
  df |>
    dplyr::filter((stage == 'pre' & stage_month <= pre_max) | (stage == 'event' & stage_month <= event_max) | (stage == 'post' & stage_month <= post_max)) |>
    dplyr::mutate(
      rel_month = dplyr::case_when(
        stage == 'pre' ~ stage_month,
        stage == 'event' ~ stage_month - 1,
        stage == 'post' ~ event_max + stage_month - 1
      )
    )
}

stage_blocks <- data.frame(
  xmin = c(-3.5, -0.5, 5.5),
  xmax = c(-0.5, 5.5, 8.5),
  stage = factor(c('Pre-event', 'Event', 'Post-event'), levels = c('Pre-event', 'Event', 'Post-event'))
)

main_traj <- read_stage_csv('event_trajectory_summary_robust.csv') |>
  dplyr::mutate(stage_month = as.numeric(stage_month), mean_value = as.numeric(mean_value), n_obs = as.numeric(n_obs)) |>
  dplyr::filter(event_type %in% c('Compound_Overlapping', 'Drought_Dominant', 'Heat_Dominant')) |>
  dplyr::filter(variable == 'ndvi') |>
  build_relative_month(event_max = 6, pre_max = 3, post_max = 3) |>
  dplyr::mutate(event_type_label = recode_event_type(event_type))

panel_a <- ggplot2::ggplot(main_traj, ggplot2::aes(rel_month, mean_value, color = event_type_label)) +
  ggplot2::geom_rect(
    data = stage_blocks,
    ggplot2::aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = stage),
    inherit.aes = FALSE,
    alpha = 0.18,
    colour = NA
  ) +
  ggplot2::geom_vline(xintercept = 0, colour = '#8f8f8f', linewidth = 0.4, linetype = 'dashed') +
  ggplot2::geom_line(linewidth = 0.9) +
  ggplot2::geom_point(size = 1.6) +
  ggplot2::scale_fill_manual(values = stage_fill, guide = 'none') +
  ggplot2::scale_color_manual(values = event_palette, name = NULL) +
  ggplot2::scale_x_continuous(limits = c(-3.5, 8.5), breaks = c(-3, -2, -1, 0, 1, 3, 5, 7, 8)) +
  ggplot2::labs(
    title = 'A. NDVI event trajectories',
    x = 'Relative event month (0 = event onset)',
    y = 'NDVI anomaly'
  ) +
  theme_stage8() +
  ggplot2::theme(
    axis.text = ggplot2::element_text(colour = '#2f2f2f'),
    axis.ticks = ggplot2::element_line(colour = '#a0a0a0'),
    plot.title = ggplot2::element_text(face = 'bold'),
    panel.grid.major.x = ggplot2::element_line(colour = '#e8e8e8', linewidth = 0.3),
    panel.grid.major.y = ggplot2::element_blank()
  )
panel_a <- ggplot2::ggplot(main_traj, ggplot2::aes(x = rel_month, y = mean_value, colour = event_type_label)) +
  
  # === 1. 背景色块层 (放在最底层) ===
  ggplot2::geom_rect(
    data = stage_blocks,
    ggplot2::aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = stage),
    inherit.aes = FALSE,
    alpha = 0.25,      # 稍微调低透明度，让色块通透、不闷
    colour = NA
  ) +
  
  # === 2. 参考线层 ===
  # 【极其重要】：异常值的 0 基准线
  ggplot2::geom_hline(yintercept = 0, colour = 'grey85', linewidth = 0.6, linetype = 'solid') +
  # 事件爆发起点 (x = 0)
  ggplot2::geom_vline(xintercept = 0, colour = '#7a7a7a', linewidth = 0.5, linetype = 'dashed') +
  
  # === 3. 数据层：圆角线条 + 白边质感点 ===
  ggplot2::geom_line(
    linewidth = 1.0, 
    alpha = 0.9, 
    linejoin = "round", # 消除线条转折处的尖角
    lineend = "round"
  ) +
  
  # === 4. 颜色与坐标轴比例 ===
  ggplot2::scale_fill_manual(values = stage_fill, guide = 'none') +
  ggplot2::scale_colour_manual(values = event_palette, name = NULL) +
  ggplot2::scale_x_continuous(
    limits = c(-3.5, 8.5), 
    breaks = c(-3, -2, -1, 0, 1, 3, 5, 7, 8),
    # 限制左右两端的无用留白，让色块更好地铺满
    expand = ggplot2::expansion(mult = c(0.02, 0.02)) 
  ) +
  
  # === 5. 标签 ===
  ggplot2::labs(
    x = 'Relative event month (0 = event onset)',
    y = 'NDVI anomaly'
  ) +
  
  # === 6. 顶刊专属纯净主题 ===
  ggplot2::theme_classic(base_size = 14) + # 替代你原来的 theme_stage8()
  ggplot2::theme(
    
    # 网格线优化：保留极淡的水平线辅助读数，去掉垂直线
    panel.grid.major.y = ggplot2::element_line(colour = '#f0f0f0', linewidth = 0.5),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    
    # 坐标轴与刻度文字加粗
    axis.text = ggplot2::element_text(colour = 'black', face = 'bold'),
    axis.line = ggplot2::element_line(colour = 'black', linewidth = 0.5),
    axis.ticks = ggplot2::element_line(colour = 'black'),
    
    # 图例位置优化：置于右上角，极其节省空间
    legend.position = 'top',
    legend.justification = 'right',
    legend.margin = margin(b = -10),
    legend.text = ggplot2::element_text(size = 12)
  )
panel_a
functional_traj <- read_stage_csv('functional_trajectory_summary_stage2_robust.csv') |>
  dplyr::mutate(stage_month = as.numeric(stage_month), mean_value = as.numeric(mean_value), n_obs = as.numeric(n_obs)) |>
  dplyr::filter(event_type %in% c('Compound_Overlapping', 'Drought_Dominant', 'Heat_Dominant', 'Sequential_Drought_Heat')) |>
  dplyr::filter(variable %in% c('sif', 'gpp', 'et')) |>
  build_relative_month(event_max = 6, pre_max = 3, post_max = 3) |>
  dplyr::mutate(
    event_type_label = recode_event_type(event_type),
    variable_label = dplyr::recode(variable, sif = 'SIF anomaly', gpp = 'GPP anomaly', et = 'ET anomaly')
  )


panel_b <- ggplot2::ggplot(functional_traj, ggplot2::aes(x = rel_month, y = mean_value, colour = event_type_label)) +
  
  # === 1. 背景色块层 ===
  ggplot2::geom_rect(
    data = stage_blocks,
    ggplot2::aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = stage),
    inherit.aes = FALSE,
    alpha = 0.25,      # 与 Panel A 保持一致的通透度
    colour = NA
  ) +
  
  # === 2. 参考线层 ===
  # 强烈建议补上 0 值基准线，方便读者衡量异常程度
  ggplot2::geom_hline(yintercept = 0, colour = 'grey85', linewidth = 0.6, linetype = 'solid') +
  ggplot2::geom_vline(xintercept = 0, colour = '#7a7a7a', linewidth = 0.5, linetype = 'dashed') +
  
  # === 3. 数据层：家族式圆角线条 + 白边质感点 ===
  ggplot2::geom_line(
    linewidth = 1.0, 
    alpha = 0.9, 
    linejoin = "round", 
    lineend = "round"
  ) +
  
  # === 4. 分面系统 ===
  ggplot2::facet_wrap(~ variable_label, scales = 'free_y', nrow = 1) +
  
  # === 5. 颜色与坐标轴比例 ===
  ggplot2::scale_fill_manual(values = stage_fill, guide = 'none') +
  ggplot2::scale_color_manual(values = event_palette, name = NULL) +
  ggplot2::scale_x_continuous(
    limits = c(-3.5, 8.5), 
    breaks = c(-3, -2, -1, 0, 1, 3, 5, 7, 8),
    expand = ggplot2::expansion(mult = c(0.02, 0.02)) 
  ) +
  
  # === 6. 标签 ===
  ggplot2::labs(
    x = 'Relative event month (0 = event onset)',
    y = NULL
  ) +
  
  # === 7. 顶刊专属纯净主题 ===
  ggplot2::theme_classic(base_size = 14) + 
  ggplot2::theme(
    
    # 【核心美化】：彻底去掉分面的灰色底框，标题左对齐加粗，产生高级悬浮感
    strip.background = ggplot2::element_blank(),
    strip.text = ggplot2::element_text(face = 'bold',hjust = 0, margin = margin(b = 6)),
    
    # 网格线优化：反转网格，保留水平线，去掉垂直线
    panel.grid.major.y = ggplot2::element_line(colour = '#f0f0f0', linewidth = 0.5),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    
    # 坐标轴与刻度文字
    axis.text = ggplot2::element_text(colour = 'black', face = 'bold'),
    axis.line = ggplot2::element_line(colour = 'black', linewidth = 0.5),
    axis.ticks = ggplot2::element_line(colour = 'black'),
    
    # 图例位置优化：移至右上角
    legend.position = 'top',
    legend.justification = 'right',
    legend.margin = margin(b = -10),
    legend.text = ggplot2::element_text(size = 12),
    
    # 【关键】：稍微拉开三个子图之间的水平间距，防止坐标轴文字拥挤打架
    panel.spacing = unit(1.2, "lines")
  )
panel_b

local_coef <- read_stage_csv('global_local_coefficients_90p_ndvi_min60.csv') |>
  grid_to_lonlat() |>
  dplyr::mutate(
    beta_interaction = as.numeric(beta_interaction),
    p_interaction = as.numeric(p_interaction),
    heat_amplifies_drought_sig = tolower(as.character(heat_amplifies_drought_sig)) == 'true'
  )
local_coef <- local_coef |>
  dplyr::mutate(
    interaction_class = dplyr::case_when(
      beta_interaction < 0 & p_interaction < 0.05 ~ 'Significant negative',
      beta_interaction < 0 ~ 'Negative, non-significant',
      beta_interaction >= 0 ~ 'Non-negative',
      TRUE ~ NA_character_
    )
  )

interaction_pct <- local_coef |>
  dplyr::filter(!is.na(beta_interaction), !is.na(interaction_class)) |>
  dplyr::count(interaction_class, name = 'n') |>
  dplyr::mutate(
    pct = 100 * n / sum(n),
    interaction_class = factor(
      interaction_class,
      levels = c('Significant negative', 'Negative, non-significant', 'Non-negative')
    )
  ) |>
  dplyr::arrange(interaction_class)

pct_negative <- interaction_pct |>
  dplyr::filter(interaction_class %in% c('Significant negative', 'Negative, non-significant')) |>
  dplyr::summarise(pct = sum(pct)) |>
  dplyr::pull(pct)

pct_sig_negative <- interaction_pct |>
  dplyr::filter(interaction_class == 'Significant negative') |>
  dplyr::pull(pct)

inset_x0 <- -176
inset_y0 <- -54
inset_width <- 64
bar_height <- 4.2

inset_segments <- interaction_pct |>
  dplyr::mutate(
    frac = pct / 100,
    xmax_frac = cumsum(frac),
    xmin_frac = dplyr::lag(xmax_frac, default = 0),
    xmin = inset_x0 + inset_width * xmin_frac,
    xmax = inset_x0 + inset_width * xmax_frac,
    ymin = inset_y0 + 10,
    ymax = inset_y0 + 10 + bar_height,
    fill_col = dplyr::case_when(
      interaction_class == 'Significant negative' ~ '#2166ac',
      interaction_class == 'Negative, non-significant' ~ '#67a9cf',
      interaction_class == 'Non-negative' ~ '#b2182b'
    )
  )

inset_keys <- data.frame(
  x = c(inset_x0 + 2, inset_x0 + 2, inset_x0 + 2),
  y = c(inset_y0 + 5.8, inset_y0 + 1.8, inset_y0 - 2.2),
  label = c(
    paste0('Significant negative: ', sprintf('%.1f', pct_sig_negative), '%'),
    paste0('Negative, n.s.: ', sprintf('%.1f', pct_negative - pct_sig_negative), '%'),
    paste0('Non-negative: ', sprintf('%.1f', 100 - pct_negative), '%')
  ),
  fill_col = c('#2166ac', '#67a9cf', '#b2182b')
)


library(rnaturalearth)
# 定义全球地图底图
world <- ne_coastline(scale = "medium", returnclass = "sf")
summary(local_coef$beta_interaction)
panel_cd_combined <- ggplot2::ggplot() +
  # === 底层: 世界地图背景 (保持 C 图设置，fill="white" 作为基底) ===
  geom_sf(data = world, fill = "white", color = "black", linewidth = 0.2) +
  
  # === 中层: 连续的局部交互强度 Raster (保持 C 图设置) ===
  ggplot2::geom_raster(data = local_coef, ggplot2::aes(lon, lat, fill = beta_interaction)) +
  
  # === 【核心新增】: 叠加显著格点 (Stippling) ===
  # 只提取显著性为 TRUE 的数据行进行叠加
  ggplot2::geom_point(
    data = local_coef[local_coef$heat_amplifies_drought_sig, ],
    ggplot2::aes(lon, lat),
    # [物理质感设置]
    shape = 1,         # 使用最小的点形 "." (pixel size)，防止格点密集时变成黑块
    color = "grey",   # 使用深灰色而不是纯黑，显得更高级、柔和
    size = 0.55,          # 如果 "." 还太大，可以微调这个值
    alpha = 0.6,          # 稍微透一点，让底部颜色隐约可见
    stroke = 0.5
  ) +
  ggplot2::annotate(
    'rect',
    xmin = inset_x0 - 3,
    xmax = inset_x0 + inset_width + 8,
    ymin = inset_y0 - 5,
    ymax = inset_y0 + 16.5,
    fill = 'white',
    colour = 'grey55',
    linewidth = 0.35,
    alpha = 0.90
  ) +
  
  
  ggplot2::geom_rect(
    data = inset_segments,
    ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = inset_segments$fill_col,
    colour = 'white',
    linewidth = 0.35
  ) +
  
  ggplot2::geom_rect(
    data = inset_keys,
    ggplot2::aes(
      xmin = x,
      xmax = x + 2.2,
      ymin = y - 0.9,
      ymax = y + 0.9
    ),
    inherit.aes = FALSE,
    fill = inset_keys$fill_col,
    colour = 'grey35',
    linewidth = 0.15
  ) +
  
  ggplot2::geom_text(
    data = inset_keys,
    ggplot2::aes(x = x + 3.5, y = y, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    size = 4,
    colour = '#202020'
  ) +
  
  
  # === 顶层: Coord 限制 (保持 C 图设置，使用 coord_sf) ===
  ggplot2::coord_sf(xlim = c(-180, 180), ylim = c(-60, 85), expand = FALSE) +
  
  # === 颜色标尺与图例美化 (同步 A/B/C 的标准) ===
  ggplot2::scale_fill_gradient2(
    low = '#2166ac',     
    mid = 'white',       
    high = '#b2182b',    
    midpoint = 0,
    
    # 【核心修改 1】：收紧色带范围。根据你的 summary，-1.5 到 1.5 是个绝佳的对比度区间
    # (如果觉得颜色还是太淡，可以进一步收紧到 c(-1, 1))
    limits = c(-1.5, 1.5), 
    
    # 【核心修改 2】：处理越界值 (Out-of-bounds)。
    # squish 会把 >1.5 的值全涂成最深的红色，把 <-1.5 的全涂成最深的蓝色，防止地图发白
    oob = scales::squish,  
    
    na.value = 'grey92', 
    name = 'Drought x heat\ninteraction beta',
    
    # 工业级水平图例排版 (让色带变得细长高级)
    guide = ggplot2::guide_colorbar(
      direction = "horizontal",
      barwidth = ggplot2::unit(15, "lines"), 
      barheight = ggplot2::unit(0.6, "lines"),
      title.position = "top",                  
      title.hjust = 0.5,
      ticks.colour = "black",                  
      frame.colour = "black",                  
      frame.linewidth = 0.5
    )
  ) +
  
  # === 标签与标题 ===
  ggplot2::labs(
    x = NULL, y = NULL
  ) +
  
  # === 顶刊专属悬浮主题 (去刻度线，加粗文字) ===
  ggplot2::theme_classic(base_size = 14) + # 替代 theme_stage8
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = 'bold',hjust = 0),
    
    # 彻底去掉地图的刻度线、网格和坐标轴线，产生高级悬浮感
    axis.line = ggplot2::element_blank(),
    axis.ticks = ggplot2::element_blank(),
    axis.text = ggplot2::element_blank(), # 地图通常不需要刻度数字，除非有特殊要求
    panel.grid = ggplot2::element_blank(),
    panel.border = ggplot2::element_blank(),
    
    # 图例排版优化：置于顶部右侧
    legend.position = "bottom",
    #legend.justification = "right",
    legend.margin = margin(b = -10),
    legend.title = ggplot2::element_text(size = 14, face = "bold")
  )

panel_cd_combined
fig4 <- (panel_a + panel_b) / panel_cd_combined

save_stage8_plot(fig4, 'Figure4_dynamics_selectivity_R.png', width = 16, height = 11)
ggplot2::ggsave(
  filename = file.path(stage8_paths$output, 'Figure4_dynamics_selectivity_R.pdf'),
  plot = fig4, 
  width = 16,      # 【大幅拉宽】：给 A 和 B 图充足的水平呼吸空间
  height = 11,     # 【增加高度】：确保地图不会被压成一条缝
  units = "in", 
  device = cairo_pdf
)
ggsave(file.path(stage8_paths$output, 'Figure4_a_R.pdf'),
       panel_a, width = 5, height = 5,device = cairo_pdf)
ggsave(file.path(stage8_paths$output, 'Figure4_b_R1.pdf'),
       panel_b, width = 12, height = 5,device = cairo_pdf)
ggsave(file.path(stage8_paths$output, 'Figure4_c_R1.pdf'),
       panel_cd_combined, width = 14, height = 7,device = cairo_pdf)
# 1. 宽度补偿：给 B 图分配 3.5 甚至 3.8 的权重，补偿它内部元素吃掉的空间
# 这样 B 图里的 3 个小图，视觉宽度就会和 A 图几乎完全一致了
top_row <- panel_a + panel_b + 
  patchwork::plot_layout(widths = c(1, 3.5))

# 2. 高度压制：大幅压扁上半部分的轨迹图，把纵向空间让给下方的地图
# 将 1:1.2 改为 1:1.8 或 1:2
fig4 <- top_row / panel_cd_combined + 
  patchwork::plot_layout(heights = c(1, 1.8))

fig4
