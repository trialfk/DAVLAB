# Aim : Perform exploratory data analysis (EDA) using R by importing, cleaning, and visualizing data to extract 
# insights and understand data distributions.(na,summary,plot,hist,boxplot).
install.packages("ggplot2") 
install.packages("dplyr")
# Load libraries
library(ggplot2)
library(dplyr)

# Load dataset 
# data <- mtcars  # Use if importing not working
data <- read.csv("mtcars.csv")
  
# Introduce missing values
set.seed(123)
data[sample(1:nrow(data), 5), "mpg"] <- NA
data[sample(1:nrow(data), 3), "wt"] <- NA

# Check missing values
cat("Missing values after introducing NAs:\n")
print(colSums(is.na(data)))

# Handle missing values (remove rows)
rows_before <- nrow(data)
data <- na.omit(data)
rows_after <- nrow(data)

cat("\nRows removed:", rows_before - rows_after, "\n")

# Check again
cat("\nMissing values after cleaning:\n")
print(colSums(is.na(data)))

# Dataset info
cat("\nHead of dataset:\n")
print(head(data))

cat("\nStructure:\n")
str(data)

cat("\nSummary:\n")
print(summary(data))

# Scatter Plot
ggplot(data, aes(x = wt, y = mpg)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "MPG vs Weight", x = "Weight", y = "MPG")

# Histogram
ggplot(data, aes(x = mpg)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 10) +
  labs(title = "MPG Distribution", x = "MPG", y = "Count")

# Box Plot
ggplot(data, aes(x = factor(cyl), y = mpg, fill = factor(cyl))) +
  geom_boxplot() +
  labs(title = "MPG by Cylinders", x = "Cylinders", y = "MPG") +
  theme(legend.position = "none")

# Summary of numeric columns
numeric_cols <- select(data, where(is.numeric))
cat("\nSummary of Numeric Columns:\n")
print(summary(numeric_cols))
