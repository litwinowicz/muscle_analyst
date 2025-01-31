library(tidyverse)
library(stats)

# Read the data
areas_data <- read.csv(snakemake@input[["areas"]])
volumes_data <- read.csv(snakemake@input[["volumes"]])

# Function to calculate correlation for each vertebral level and slice
calculate_correlations <- function(areas_data, volumes_data) {
    # Get unique combinations of vertebral level and slice
    level_slice_combos <- areas_data %>%
        group_by(vertebral_level, slice) %>%
        summarise(n = n(), .groups = "drop")

    # Calculate correlation for each combination
    correlations <- lapply(1:nrow(level_slice_combos), function(i) {
        level <- level_slice_combos$vertebral_level[i]
        slice_num <- level_slice_combos$slice[i]

        # Get areas for this level and slice
        areas <- areas_data %>%
            filter(vertebral_level == level, slice == slice_num) %>%
            select(subject_id, area)

        # Join with volumes
        combined <- areas %>%
            inner_join(volumes_data, by = "subject_id")

        # Calculate correlation if we have enough data points
        if (nrow(combined) > 2) {
            cor_result <- cor.test(combined$area, combined$original_volume)
            data.frame(
                vertebral_level = level,
                slice = slice_num,
                correlation = cor_result$estimate,
                p_value = cor_result$p.value,
                significant = cor_result$p.value < 0.05
            )
        }
    })

    # Combine results
    correlations_df <- do.call(rbind, correlations)

    # Create factor for vertebral levels in correct order
    all_levels <- c(paste0("C", 1:7), paste0("T", 1:12), paste0("L", 1:5))
    correlations_df$vertebral_level <- factor(correlations_df$vertebral_level, levels = all_levels)

    return(correlations_df)
}

# Calculate correlations
correlations <- calculate_correlations(areas_data, volumes_data)

# Create x-axis position for continuous plotting
correlations <- correlations %>%
    arrange(vertebral_level, slice) %>%
    mutate(x_position = as.numeric(vertebral_level) * 5 + slice - 5)

# Create labels for x-axis
all_levels <- c(paste0("C", 1:7), paste0("T", 1:12), paste0("L", 1:5))
x_labels <- expand.grid(
    level = all_levels,
    slice = 1:5
) %>%
    arrange(level, slice) %>%
    mutate(x_position = as.numeric(factor(level, levels = all_levels)) * 5 + slice - 5)

# Create the plot
ggplot(correlations, aes(x = x_position, y = correlation)) +
    # Add points and lines
    geom_line(linewidth = 0.3, alpha = 0.8) +
    geom_point(aes(shape = significant), size = 2, alpha = 0.8) +
    # Customize scales
    scale_x_continuous(
        breaks = x_labels$x_position,
        labels = paste(x_labels$level, x_labels$slice, sep = "-"),
        expand = c(0.02, 0.02)
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_shape_manual(
        values = c("TRUE" = 16, "FALSE" = 1),
        labels = c("TRUE" = "p < 0.05", "FALSE" = "p ≥ 0.05"),
        name = "Statistical\nSignificance"
    ) +
    # Labels
    labs(
        title = "Correlation between Muscle Area and Fat Volume",
        subtitle = "Correlations calculated across subjects for each vertebral level and slice",
        x = "Vertebral Level - Slice",
        y = "Correlation coefficient (r)"
    ) +
    theme_minimal() +
    theme(
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        panel.grid.minor = element_blank(),
        legend.position = "right"
    )

# Save the plot
ggsave(snakemake@output[[1]], width = 15, height = 8)
