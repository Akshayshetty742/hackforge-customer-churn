# Developer Guide: Customer Churn Prediction

This guide explains the codebase structure, the rationale behind technical decisions, and how to extend or maintain the project.

## 1. Project Architecture

The project is a standalone Python script (`01_churn_prediction_xgboost.py`) that performs an end-to-end machine learning workflow:
1.  **Data Ingestion:** Reads CSV data.
2.  **Preprocessing:** Cleans and encodes data for machine learning.
3.  **Model Training:** Uses XGBoost with Cross-Validation.
4.  **Evaluation:** Generates detailed performance metrics.
5.  **Serialization:** Saves the model and encoders for future use.

## 2. Key Decisions & Rationale

### Why XGBoost?
We migrated from Random Forest to **XGBoost** because:
-   **Better Generalization:** XGBoost uses boosting (correcting previous errors) rather than bagging (averaging independent trees), often leading to higher accuracy.
-   **Regularization:** Built-in parameters (`lambda`, `alpha`) help prevents overfitting, which was a major issue in previous iterations.

### Handling Imbalanced Data
The dataset contains roughly **73% Non-Churners** and **27% Churners**.
-   **Old Approach (SMOTE):** Created synthetic (fake) data. Resulted in overfitting (97% Train / 78% Test).
-   **Intermediate Approach (Downsampling):** Deleted majority class data. Resulted in data loss and lower accuracy (74%).
-   **Current Approach (Class Weights):** We use `scale_pos_weight` calculated as `Majority Count / Minority Count` (~2.77).
    -   *Why:* This forces the model to treat every missed churner as **2.77x more important** than a missed non-churner. It improves Recall without discarding real data.

### Preventing Overfitting
We explicitly limit the model's complexity to ensure it learns patterns, not noise:
-   **`max_depth=5`**: Restricts trees to 5 levels deep.
-   **`subsample=0.8`**: Trains each tree on only 80% of data (adds randomness).
-   **`colsample_bytree=0.8`**: Uses only 80% of features per tree.

## 3. Code Walkthrough

### Dependencies
The project relies on a virtual environment (`churn_env`) to manage:
-   `pandas`, `numpy`: Data manipulation.
-   `sklearn`: Splitting, encoding, and metrics.
-   `xgboost`: The core modeling library.

### The Pipeline (`01_churn_prediction_xgboost.py`)
-   **Lines 1-30 (Preprocessing):** Standard cleaning. Note that `TotalCharges` is converted to numeric, and empty strings are handled.
-   **Lines 33-44 (Encoding):** `LabelEncoder` transforms text to numbers.
    -   *Crucial:* The encoders are saved to `encoders.pkl`. You **MUST** use these same encoders when predicting on new data to ensure mapping consistency (e.g., "Yes" is always `1`).
-   **Lines 50-55 (Splitting):** `train_test_split` with `stratify=y` ensures the test set represents the real world.
-   **Lines 84-100 (Tuning):** `GridSearchCV` searches for the best hyperparameters. It uses 3-fold cross-validation to validate results accurately.

## 4. Deployment & Usage

### Files Required for Production
To use this model in an application (e.g., a web service), you need two artifacts:
1.  `customer_churn_model_xgboost.pkl`: The trained XGBoost model.
2.  `encoders.pkl`: The dictionary of LabelEncoders.

### Example Prediction Code
```python
import pickle
import pandas as pd

# 1. Load Artifacts
with open('customer_churn_model_xgboost.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# 2. Prepare New Data (Example)
new_customer = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    # ... include all other columns ...
})

# 3. Apply Encoders
for col, encoder in encoders.items():
    if col in new_customer.columns:
        new_customer[col] = encoder.transform(new_customer[col])

# 4. Predict
prediction = model.predict(new_customer)
print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")
```

## 5. Future Improvements
To further increase accuracy (target: >80%):
1.  **Feature Engineering:** Create new features like `TenureYears` (binning months) or `HasStreamingService` (combining TV/Movies).
2.  **Threshold Tuning:** Adjust the classification threshold. By default, probability > 0.5 is "Churn". Lowering to 0.4 might catch more churners (higher Recall) at the cost of Precision.
