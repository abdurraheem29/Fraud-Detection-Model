# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(doParallel)

# Load datasets
train_data <- read.csv("/Users/YourUsername/Downloads/fraudTrain.csv")
test_data <- read.csv("/Users/YourUsername/Downloads/fraudTest.csv")

# Explore dataset structure
str(train_data)

# Data Preprocessing
## Convert target variable to factor
train_data$is_fraud <- as.factor(train_data$is_fraud)
test_data$is_fraud <- as.factor(test_data$is_fraud)

## Check for missing values
print(paste("Missing values in train data:", sum(is.na(train_data))))
print(paste("Missing values in test data:", sum(is.na(test_data))))

# Feature Selection
## Remove non-informative columns
non_informative_columns <- c("X", "first", "last", "street")
train_data <- train_data %>% select(-all_of(non_informative_columns))
test_data <- test_data %>% select(-all_of(non_informative_columns))

## Select important columns to reduce memory usage
important_columns <- c("is_fraud", "amt", "category", "gender", "city_pop", "state", "zip")
if (all(important_columns %in% colnames(train_data))) {
  train_data <- train_data[, important_columns]
  test_data <- test_data[, important_columns]
} else {                                                                                                                                                     
  stop("One or more important columns are missing in the dataset")
}

# Split train dataset for model validation
set.seed(123)
train_index <- createDataPartition(train_data$is_fraud, p = 0.8, list = FALSE)
train_split <- train_data[train_index, ]
validation_split <- train_data[-train_index, ]

# Sample a subset of the training data to reduce memory usage
set.seed(123)
train_split_sample <- train_split[sample(1:nrow(train_split), size = min(100000, nrow(train_split))), ]

# Model Building
## Logistic Regression
logistic_model <- glm(is_fraud ~ ., data = train_split_sample, family = "binomial")
logistic_preds <- predict(logistic_model, validation_split, type = "response")
logistic_class <- ifelse(logistic_preds > 0.5, 1, 0)
logistic_accuracy <- mean(logistic_class == validation_split$is_fraud)
print(paste("Logistic Regression Accuracy:", logistic_accuracy))

## Random Forest with optimized parameters
cl <- makeCluster(detectCores() - 1)  # Use parallel processing
registerDoParallel(cl)

rf_model <- randomForest(is_fraud ~ ., data = train_split_sample, ntree = 50, mtry = 3)
stopCluster(cl)  # Stop the cluster

rf_preds <- predict(rf_model, validation_split)
rf_accuracy <- mean(rf_preds == validation_split$is_fraud)
print(paste("Random Forest Accuracy:", rf_accuracy))

# Test Model on Test Data
rf_test_preds <- predict(rf_model, test_data)
rf_test_accuracy <- mean(rf_test_preds == test_data$is_fraud)
print(paste("Random Forest Test Accuracy:", rf_test_accuracy))

# Save the Model
saveRDS(rf_model, "fraud_detection_rf_model.rds")

# Conclusion
## Random Forest is likely to perform better than Logistic Regression for this dataset based on its accuracy.
