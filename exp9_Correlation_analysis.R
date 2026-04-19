# Aim : Analyze relationships between multiple variables using visualizations and correlation analysis in R for better data interpretation. 
install.packages("ggplot2")
install.packages("GGally")
install.packages("corrplot")

# Load libraries
library(ggplot2)
library(GGally)
library(corrplot)

# Load dataset
# data <- mtcars   # Use if importing csv not working
data <- read.csv("mtcars.csv")

# Correlation matrix
cor_matrix <- cor(data)
print(cor_matrix)

# Correlation plot
corrplot(cor_matrix, method = "circle", type = "upper")

# Pairwise scatter plots
ggpairs(data, columns = 1:5)

# Scatter plot with regression line
ggplot(data, aes(x = wt, y = mpg)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "MPG vs Weight", x = "Weight", y = "MPG")

# Scatter plot by cylinder category
ggplot(data, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "MPG vs Weight by Cylinders",
       x = "Weight",
       y = "MPG",
       color = "Cylinders")# Load libraries
library(ggplot2)
library(GGally)
library(corrplot)

# Load dataset
data <- mtcars

# Correlation matrix
cor_matrix <- cor(data)
print(cor_matrix)

# Correlation plot
corrplot(cor_matrix, method = "circle", type = "upper")

# Pairwise scatter plots
ggpairs(data, columns = 1:5)

# Scatter plot with regression line
ggplot(data, aes(x = wt, y = mpg)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "MPG vs Weight", x = "Weight", y = "MPG")

# Scatter plot by cylinder category
ggplot(data, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "MPG vs Weight by Cylinders",
       x = "Weight",
       y = "MPG",
       color = "Cylinders")