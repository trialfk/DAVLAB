install.packages("dplyr")   
install.packages("ggplot2")
install.packages("tidyr")
install.packages("data.table")
install.packages("caret")

library(dplyr)
# Filter cars with mpg greater than 20
filtered_data <- mtcars %>%
  filter(mpg > 20)
print(filtered_data)

# Arrange (sort)
arranged_data <- mtcars %>%
  arrange(desc(mpg))

print(arranged_data)

# Group and summarize
grouped_data <- mtcars %>%
  group_by(cyl) %>%
  summarize(avg_mpg = mean(mpg))

print(grouped_data)


library(ggplot2)
# Scatter plot of mpg vs. hp
ggplot(mtcars, aes(x = hp, y = mpg)) +
  geom_point() +
  labs(title = "Horsepower vs. MPG",
       x = "Horsepower",
       y = "Miles per Gallon")



library(tidyr)
# Sample messy dataset
data <- data.frame(
  id = 1:3,
  year_2023 = c(10, 15, 20),
  year_2024 = c(25, 30, 35)
)
# Transform data to a tidy format
tidy_data <- data %>%
  pivot_longer(cols = starts_with("year"),
               names_to = "year",
               values_to = "value")

print(tidy_data)



library(data.table)
# Create a data.table
data <- data.table(mtcars)
# Calculate mean mpg by number of cylinders
result <- data[, .(mean_mpg = mean(mpg)), by = cyl]
print(result)



library(caret)
# Splitting data into training and testing sets
data <- mtcars
set.seed(123)
trainIndex <- createDataPartition(data$mpg, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data  <- data[-trainIndex, ]
# Train a linear regression model
model <- train(mpg ~ hp + wt, data = train_data, method = "lm")
print(summary(model))
