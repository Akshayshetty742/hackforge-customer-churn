# Customer Churn Prediction Model

## Project Overview
This project implements a machine learning solution to predict customer churn (customers likely to leave a service). It analyzes customer data (demographics, services used, billing info) to identify at-risk users, enabling proactive retention strategies.

## Key Achievements
- **High Recall for Churners:** Successfully identifies **~76%** of customers who are actually leaving.
- **Addressed Overfitting:** Reduced the gap between training and testing accuracy from **~19%** to **~7%**, creating a robust, generalizable model.
- **Handled Imbalanced Data:** Implemented advanced techniques to ensure the model doesn't just bias towards the majority "Stay" class.

## Techniques Implemented
This project successfully implements all the advanced strategies required for a robust machine learning pipeline:

### 1. Model Selection
- **Implementation:** Switched from `RandomForestClassifier` to `XGBoostClassifier`.
- **Reason:** XGBoost provided better handling of tabular data and built-in regularization, which was crucial for improving model generalization.

### 2. Addressed Overfitting
- **Implementation:**
  - Limited tree depth (`max_depth=5`) to prevent memorization.
  - Used `subsample=0.8` to train on random subsets of data.
  - Switched to XGBoost for its superior regularization parameters.
- **Result:** Drastically reduced the Train-Test accuracy gap from **~19%** (High Overfitting) to **~7%** (Good Generalization).

### 3. Hyperparameter Tuning
- **Implementation:** Used `GridSearchCV` to exhaustively test 36 different combinations of 6 hyperparameters.
- **Outcome:** Automatically identified the optimal configuration:
  - `learning_rate`: 0.1
  - `n_estimators`: 100
  - `max_depth`: 5

### 4. Stratified K-Fold CV
- **Implementation:** Used `StratifiedKFold(n_splits=3)` during the Grid Search.
- **Reason:** Ensuring that every training fold had the exact same percentage of churners (27%) as the full dataset prevented biased training runs.

### 5. Handling Class Imbalance (Alternative to Downsampling)
- **Implementation:** Instead of "Downsampling" (which deletes data), we implemented **Class Weighting** (`scale_pos_weight`).
- **Reason:** Downsampling reduced accuracy (74%) by discarding 2,600 valid samples. Weighting (giving Churners a 2.77x importance score) allowed us to use **100% of the data** while still achieving high Recall (75%) and higher Accuracy (76%).

## Files in Repository
- `01_churn_prediction_xgboost.py`: The main script. Loads data, trains model, tunes hyperparameters, evaluates performance, and saves the model.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset.
- `customer_churn_model_xgboost.pkl`: The saved, trained model (ready for deployment).
- `encoders.pkl`: Saved encoding rules (required to process new data).
- `churn_env/`: (Directory) Local Python virtual environment containing dependencies.

## How to Run

1. **Activate Environment:**
   ```bash
   source churn_env/bin/activate
   ```

2. **Run Script:**
   ```bash
   python 01_churn_prediction_xgboost.py
   ```

3. **Output:**
   The script will print accuracy metrics, the confusion matrix, and the classification report to the terminal.

## Metrics Explained
- **Precision (0.53):** When predicting "Churn", correct 53% of the time. (Higher false alarms, but necessary for safety).
- **Recall (0.75):** Correctly identifies 75% of ALL actual churners. (This is the critical metric for retention).
- **Accuracy (0.76):** Overall correct predictions across both classes.
