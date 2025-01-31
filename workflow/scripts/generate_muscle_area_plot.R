library(tidyverse)

# Read the data
data <- read.csv(snakemake@input[[1]])

# Create a factor for vertebral levels in the correct order
all_levels <- c(paste0("C", 1:7), paste0("T", 1:12), paste0("L", 1:5))
data$vertebral_level <- factor(data$vertebral_level, levels = all_levels)

# Create a numerical x-axis value that combines level and slice
data <- data %>%
  arrange(subject_id, vertebral_level, slice) %>%
  mutate(x_position = as.numeric(factor(vertebral_level)) * 5 + slice - 5)

# Create labels for x-axis
x_labels <- expand.grid(
  level = all_levels,
  slice = 1:5
) %>%
  arrange(level, slice) %>%
  mutate(x_position = as.numeric(factor(level, levels = all_levels)) * 5 + slice - 5)

# Create the plot
ggplot(data, aes(x = x_position, y = area, group = subject_id, color = subject_id)) +
  # Original data with thin lines and points
  geom_line(linewidth = 0.3, alpha = 0.3) +
  geom_point(size = 0.8, alpha = 0.3) +
  # Add smoothed lines
  geom_smooth(se = FALSE, linewidth = 1, span = 0.2, alpha = 0.8) +
  scale_x_continuous(
    breaks = x_labels$x_position,
    labels = paste(x_labels$level, x_labels$slice, sep = "-"),
    expand = c(0.02, 0.02)
  ) +
  labs(
    title = "Muscle Area by Subject, Vertebral Level and Slice",
    x = "Vertebral Level - Slice",
    y = "Area",
    color = "Subject ID"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

# Save the plot
ggsave(snakemake@output[[1]], width = 15, height = 8)
