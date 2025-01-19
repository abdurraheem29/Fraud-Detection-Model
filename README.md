# Fraud-Detection-Model
# Fraud Detection Model

This project focuses on building machine learning models to detect fraudulent credit card transactions. The models classify transactions as either **fraudulent** or **legitimate**, based on patterns in the dataset.

## Dataset
The dataset contains information about credit card transactions with features such as:
- **`is_fraud`**: Target variable (1 = Fraudulent, 0 = Legitimate)
- **Transaction details**: `amt` (amount), `category` (merchant category), etc.
- **User details**: `gender`, `state`, `zip`, `city_pop` (city population), etc.

### Files
You can get the dataset from here https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download
- `fraudTrain.csv`: Training data for model building.
- `fraudTest.csv`: Test data for evaluation.

## Project Structure
1. **Data Preprocessing**:
   - Conversion of categorical columns to factors.
   - Selection of important features for model building.
   - Handling of missing values (if any).

2. **Model Building**:
   - **Logistic Regression**: A simple linear model to classify transactions.
   - **Random Forest**: An ensemble model for better accuracy and handling non-linear relationships.

3. **Model Evaluation**:
   - Models were evaluated using accuracy on both validation and test datasets.

## Results
- **Logistic Regression Accuracy**:
  - Validation: 99.34%
  - Test: Not applicable (validation results used for comparison).

- **Random Forest Accuracy**:
  - Validation: 99.69%
  - Test: 99.76%

## Conclusion
- The **Random Forest model** outperformed Logistic Regression, making it the recommended model for deployment.
- Both models showed exceptional accuracy in classifying transactions as fraudulent or legitimate.

## How to Run
1. **Install Required Libraries**:
   Ensure you have the following R packages installed:
   ```R
   install.packages(c("tidyverse", "caret", "randomForest", "data.table", "doParallel"))
   ```

2. **Prepare the Dataset**:
   - Place `fraudTrain.csv` and `fraudTest.csv` in your working directory.

3. **Run the Script**:
   Execute the R script provided in `fraud_detection_model.R`.

4. **Output**:
   - The model will output the accuracy metrics for both Logistic Regression and Random Forest.
   - The trained Random Forest model is saved as `fraud_detection_rf_model.rds`.

## Recommendations
- Deploy the Random Forest model for production use.
- Periodically retrain the model with updated data to maintain its effectiveness.

## Future Improvements
- Evaluate additional metrics like Precision, Recall, and F1-Score to assess performance on imbalanced data.
- Explore advanced models such as Gradient Boosting (e.g., XGBoost) or Neural Networks for further accuracy improvements.
- Incorporate real-time data for dynamic fraud detection.

---

### Author
This project was developed using R on macOS. If you have any questions or suggestions, feel free to reach out!



