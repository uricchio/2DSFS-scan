library(ggplot2)
library(tidyverse)
library(dplyr)
library(stringr)
library(qqman)
library(readr)
library(ggrepel)
library(RColorBrewer)
library(fastman)

setwd("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan")

#### Fst and CLR significance comparisons ####

# change chromosome IDs format and extract coordinates
process_pixy_data = function(data){
  data.tmp = read.csv(data)
  data.tmp$chromosome = sub("^(.*?_.*?)_(.*)$", "\\1.\\2", data.tmp$chromosome)
  colnames(data.tmp)[4] <- "window_start"
  colnames(data.tmp)[5] <- "window_end"
  data.tmp <- data.tmp %>% filter(!str_detect(chromosome, "^NW"))
  
  mapping <- read.table("chromosomes.txt", header = TRUE, stringsAsFactors = FALSE)
  data.tmp <- data.tmp %>%
    left_join(mapping, by = c("chromosome" = "chr_id")) %>%
    mutate(chromosome = chr_num) %>%
    select(-chr_num)
}

fst_500kb = process_pixy_data("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/pixy_data/fst_500kb.csv")

# read ECB Stats file and merge with FST data
clr_500kb = read.csv("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/ECBstats_500kb.csv")
clr_500kb <- clr_500kb %>% filter(!str_detect(chromosome, "^NW"))

fst_clr_500kb <- merge(
  clr_500kb, 
  fst_500kb[, c("chromosome", "window_start", "window_end", "avg_wc_fst")], 
  by = c("chromosome", "window_start", "window_end"), 
  all.x = TRUE
)

# Manhattan plots
# to filter windows with few SNPs, estimate quantile of snp counts per chromosome
filter_windows_numSnps = function(data, probs){
  data$chromosome = as.numeric(data$chromosome)
  data %>% 
    group_by(chromosome) %>%
    summarise(quantile_filter = quantile(snp_count, probs = probs, na.rm = TRUE))
}

# gg.manhattan colors the outlier windows depending on the threshold 'probs'
# also plots windows that have SNP counts that are above 'qtile.counts' for each chromosome
gg.manhattan <- function(data, y_variable, ylab, title, limits, qtile.counts = NULL, probs = NULL){
  
  # Ensure chromosome in `qtile.counts` matches the type in `data`
  if (!is.null(qtile.counts)) {
    qtile.counts <- qtile.counts %>%
      mutate(chromosome = as.character(chromosome))  # Convert to character to match `data`
    
    data <- data %>%
      left_join(qtile.counts, by = "chromosome") %>%
      filter(snp_count > quantile_filter)
  }
  
  # Calculate the significance threshold if `probs` is provided
  threshold <- if (!is.null(probs)) {
    quantile(data[[y_variable]], probs = probs, na.rm = TRUE)
  } else {
    NULL
  }
  
  # Add a column to identify points above the threshold
  data <- data %>%
    mutate(above_threshold = if (!is.null(threshold)) !!sym(y_variable) > threshold else FALSE)
  
  # Ensure chromosome is numeric
  data$chromosome <- as.numeric(data$chromosome)
  
  # Preprocess data for Manhattan plot
  data.tmp <- data %>% 
    group_by(chromosome) %>% 
    summarise(chr_len = max(window_end)) %>% 
    mutate(tot = cumsum(chr_len) - chr_len) %>%
    select(-chr_len) %>%
    left_join(data, ., by = c("chromosome" = "chromosome")) %>%
    arrange(chromosome, window_end) %>%
    mutate(BPcum = window_end + tot)
  
  # Compute axis labels
  axisdf <- data.tmp %>%
    group_by(chromosome) %>%
    summarize(center = (max(BPcum) + min(BPcum)) / 2)
  
  # Convert chromosome numbers to character and rename 31 and 32
  axisdf$chromosome <- as.character(axisdf$chromosome)
  axisdf$chromosome[axisdf$chromosome == "31"] <- "W"
  axisdf$chromosome[axisdf$chromosome == "32"] <- "Z"
  
  # Ensure labels appear for every other chromosome but always show "W" and "Z"
  labels <- ifelse(
    (seq_along(axisdf$chromosome) %% 2 == 1) | axisdf$chromosome %in% c("W", "Z"),
    axisdf$chromosome, 
    ""
  )
  
  # Assign colors for chromosomes
  chromosome_colors <- rep(c("lightblue2", "midnightblue"), length.out = length(unique(data$chromosome)))
  names(chromosome_colors) <- as.character(sort(unique(data$chromosome)))
  
  # Add a custom color for points above the threshold
  chromosome_colors <- c(chromosome_colors, "Above Threshold" = "red")
  
  # Generate the plot
  p <- ggplot(data.tmp, aes(x = BPcum, y = !!sym(y_variable))) +
    geom_point(aes(color = ifelse(above_threshold, "Above Threshold", as.character(chromosome))),
               alpha = 0.8, size = 1.3) +
    scale_color_manual(
      values = chromosome_colors
    ) +
    scale_x_continuous(
      breaks = axisdf$center,  # Keep all breaks for tick lines
      labels = labels  # Show labels only for every other chromosome
    ) +
    scale_y_continuous(limits = limits) +
    labs(y = ylab, x = "Chromosome", title = title) +
    theme_bw() +
    theme(
      legend.position = "none",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    theme(axis.text.x = element_text(color = "black", size = 10)) +
    theme(axis.text.y = element_text(color = "black", size = 10))
  
  # Add threshold line if it exists
  if (!is.null(threshold)) {
    p <- p + geom_hline(yintercept = threshold, color = "red", linetype = "dashed")
  }
  
  return(p)
}

# compare statistics, probs is the threshold of choice (e.g., probs = 0.95 for the 95% threshold)
# qtile.counts is a data frame with the minimum number of SNPs per chromosome to filter

plot.stats.comparison = function(data, x_variable, y_variable, 
                                 title = NULL, probs = NULL, qtile.counts = NULL){
  
  # list for labels
  var_labels <- list(
    "avg_wc_fst" = bquote(F['ST']),
    "T2D" = bquote(T['2D']),
    "T1D_p1" = bquote(uv ~ T['1D']),
    "T1D_p2" = bquote(bv ~ T['1D']),
    "new_term_p1" = bquote(uv ~ T['2D'] - T['1D']),
    "new_term_p2" = bquote(bv ~ T['2D'] - T['1D']),
    "T2D_diff" = "T2D - (T1Dp1 + T1Dp2)/2"
  )
  
  x_label <- if (!is.null(var_labels[[x_variable]])) var_labels[[x_variable]] else x_variable
  y_label <- if (!is.null(var_labels[[y_variable]])) var_labels[[y_variable]] else y_variable
  
  
  # filter out windows with snp counts under the quantiles for each chromosome
  if (!is.null(qtile.counts)) {
    qtile.counts <- qtile.counts %>%
      mutate(chromosome = as.character(chromosome))  # Convert to character to match `data`
    
    data <- data %>%
      left_join(qtile.counts, by = "chromosome") %>%
      filter(snp_count > quantile_filter)
  }
  
  # get thresholds
  y_threshold <- if (!is.null(probs) && nrow(data) > 0) {
    quantile(data[[y_variable]], probs = probs, na.rm = TRUE)
  } else {
    NULL
  }
  
  x_threshold <- if (!is.null(probs) && nrow(data) > 0) {
    quantile(data[[x_variable]], probs = probs, na.rm = TRUE)
  } else {
    NULL
  }
  
  # determine significance
  data$significance <- with(data, ifelse(
    data[[x_variable]] > x_threshold & data[[y_variable]] > y_threshold, "both significant",
    ifelse(data[[x_variable]] > x_threshold, "x significant only",
           ifelse(data[[y_variable]] > y_threshold, "y significant only", "not significant")
    )
  ))
  
  plot <- ggplot(data, aes(x = !!sym(x_variable), y = !!sym(y_variable))) +
    geom_point(aes(color = significance), size = 2, alpha = 0.8) +
    geom_hline(yintercept = y_threshold, color = "gray", linetype = "dashed", size = 0.5) +  
    geom_vline(xintercept = x_threshold, color = "gray", linetype = "dashed", size = 0.5) + 
    labs(x = x_label, y = y_label, color = "Significance", shape = "Significance") +
    scale_color_manual(values = c("both significant" = "seagreen", 
                                  "x significant only" = "navy", 
                                  "y significant only" = "lightcoral",
                                  "not significant" = "grey"))+
    theme_minimal() +
    ggtitle(title) +
    theme(
      axis.text.x = element_text(color = "black", size = 10),
      axis.text.y = element_text(color = "black", size = 10),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA),
      axis.ticks = element_line(color = "black"),
      axis.ticks.length = unit(0.2, "cm"),
      legend.position = 'none'
    )
  
  return(plot)
}

# 500 kb windows
fst_clr_500kb <- fst_clr
numSnps_qtile_clr500kb = filter_windows_numSnps(fst_clr_500kb, 0.15)

gg.manhattan(fst_clr_500kb, y_variable = 'avg_wc_fst', 
             ylab = bquote(F['ST']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)

gg.manhattan(fst_clr_500kb, y_variable = 'T2D', 
             ylab = bquote(T['2D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)

gg.manhattan(fst_clr_500kb, y_variable = 'T1D_p1', 
             ylab = bquote(uv ~ T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)

gg.manhattan(fst_clr_500kb, y_variable = 'T1D_p2', 
             ylab = bquote(bv ~ T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)

gg.manhattan(fst_clr_500kb, y_variable = 'new_term_p1', 
             ylab = bquote(uv ~ T['2D'] - T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)

gg.manhattan(fst_clr_500kb, y_variable = 'new_term_p2', 
             ylab = bquote(bv ~ T['2D'] - T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr500kb, probs = 0.95)



plot.stats.comparison(fst_clr_500kb, 'T1D_p1', 'T2D', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)
plot.stats.comparison(fst_clr_500kb, 'T1D_p2', 'T2D', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)

plot.stats.comparison(fst_clr_500kb, 'avg_wc_fst', 'T2D', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)

plot.stats.comparison(fst_clr_500kb, 'T1D_p1', 'new_term_p1', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)
plot.stats.comparison(fst_clr_500kb, 'T1D_p2', 'new_term_p2', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)

plot.stats.comparison(fst_clr_500kb, 'T1D_p1', 'T2D_diff', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)
plot.stats.comparison(fst_clr_500kb, 'T1D_p2', 'T2D_diff', probs = 0.95, qtile.counts = numSnps_qtile_clr500kb)

# 20 kb windows 
fst_20kb = process_pixy_data("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/pixy_data/fst_20kb.csv")

clr_20kb = read.csv("/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/data/ECBstats_20kb.csv")
clr_20kb <- clr_20kb %>% filter(!str_detect(chromosome, "^NW"))

fst_clr_20kb <- merge(
  clr_20kb, 
  fst_20kb[, c("chromosome", "window_start", "window_end", "avg_wc_fst")], 
  by = c("chromosome", "window_start", "window_end"), 
  all.x = TRUE
)

# quantile filter is set at 85% of snp count
numSnps_qtile_clr20kb = filter_windows_numSnps(fst_clr_20kb, 0.15)

gg.manhattan(fst_clr_20kb, y_variable = 'avg_wc_fst', 
             ylab = bquote(F['ST']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

gg.manhattan(fst_clr_20kb, y_variable = 'T2D', 
             ylab = bquote(T['2D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

gg.manhattan(fst_clr_20kb, y_variable = 'T1D_p1', 
             ylab = bquote(uv ~ T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

gg.manhattan(fst_clr_20kb, y_variable = 'T1D_p2', 
             ylab = bquote(bv ~ T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

gg.manhattan(fst_clr_20kb, y_variable = 'new_term_p1', 
             ylab = bquote(uv ~ T['2D'] - T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

gg.manhattan(fst_clr_20kb, y_variable = 'new_term_p2', 
             ylab = bquote(bv ~ T['2D'] - T['1D']), limits = NULL,
             title = NULL, qtile.counts = numSnps_qtile_clr20kb, probs = 0.99)

plot.stats.comparison(fst_clr_20kb, 'T1D_p1', 'T2D', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)
plot.stats.comparison(fst_clr_20kb, 'T1D_p2', 'T2D', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)

plot.stats.comparison(fst_clr_20kb, 'avg_wc_fst', 'T2D', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)

plot.stats.comparison(fst_clr_20kb, 'T1D_p1', 'new_term_p1', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)
plot.stats.comparison(fst_clr_20kb, 'T1D_p2', 'new_term_p2', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)

plot.stats.comparison(fst_clr_20kb, 'T1D_p1', 'T2D_diff', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)
plot.stats.comparison(fst_clr_20kb, 'T1D_p2', 'T2D_diff', probs = 0.99, qtile.counts = numSnps_qtile_clr20kb)


# correlation comparisons
# 'get_correlation_ecb' gets correlations based on selected generation and region
get_correlation_ecb <- function(data, plot_title) {
  
  # select only relevant numeric variables
  vars_to_correlate <- c("avg_wc_fst", "snp_count", "T2D", "T1D_p1", "T1D_p2", "new_term_p1", "new_term_p2")
  
  df_selected <- data[vars_to_correlate]
  
  # compute correlation matrix
  cor_matrix <- cor(df_selected, method = "spearman", use = "pairwise.complete.obs")
  
  # compute p-values
  p_values <- matrix(NA, ncol = length(vars_to_correlate), nrow = length(vars_to_correlate), 
                     dimnames = list(vars_to_correlate, vars_to_correlate))
  
  for (i in 1:length(vars_to_correlate)) {
    for (j in i:length(vars_to_correlate)) {
      test <- cor.test(df_selected[[i]], df_selected[[j]], method = "spearman")
      p_values[i, j] <- test$p.value
      p_values[j, i] <- test$p.value
    }
  }
  
  # print p-values
  print("P-value matrix:")
  print(p_values)
  
  
  # generate correlation plot
  corrplot(cor_matrix, method = "circle", type = "lower", tl.col = "black",
           addCoef.col = "black", diag=FALSE, insig='blank', p.mat = p_values)
  
  title(main = plot_title, col.main = "black", font.main = 2)
  
}

get_correlation_ecb(fst_clr_20kb, plot_title = '20kb scan')
get_correlation_ecb(fst_clr_500kb, plot_title = '500kb scan')

#save data frames
fst_clr_500kb <- fst_clr_500kb %>% rename(FST=avg_wc_fst)
fst_clr_20kb <- fst_clr_20kb %>% rename(FST=avg_wc_fst)

write.csv(fst_clr_500kb,file='/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/data/ECBstats_FST_500kb.csv', row.names=FALSE)
write.csv(fst_clr_20kb,file='/Users/marlonalejandrocalderonbalcazar/Desktop/ECB/2DSFS_scan/data/ECBstats_FST_20kb.csv', row.names=FALSE)
