import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. Data Loading
print("Loading data...")
try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found in current directory or project folder.")
    exit(1)

# 2. Preprocessing
print("Preprocessing data...")
# Drop customerID
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Handle TotalCharges - replace empty strings with 0 and convert to float
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Label Encoding for categorical columns
object_columns = df.select_dtypes(include="object").columns
encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save encoders (Important: These must be consistent with the model)
output_path_encoders = "encoders.pkl"
with open(output_path_encoders, "wb") as f:
    pickle.dump(encoders, f)
print(f"Encoders saved to '{output_path_encoders}'.")

# 3. Train-Test Split
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split data FIRST (before downsampling) to keep test data pure
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Original training shape: {X_train.shape}")
print(f"Original target distribution:\n{y_train.value_counts()}")

# 4. XGBoost with Class Weights (Handling Imbalance without deleting data)
print("\nCalculating class weights...")
# scale_pos_weight = total_negative_examples / total_positive_examples
# This tells the model to pay more attention to the minority class
count_minority = y_train.sum()
count_majority = len(y_train) - count_minority
ratio = count_majority / count_minority
print(f"Majority/Minority Ratio: {ratio:.2f}")

# 5. Hyperparameter Tuning
print("\nStarting Hyperparameter Tuning with XGBoost & GridSearchCV...")

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],           # Kept low to prevent overfitting on the larger dataset
    'learning_rate': [0.05, 0.1],     # Standard rates
    'subsample': [0.8],               
    'colsample_bytree': [0.8],
    'scale_pos_weight': [ratio]       # Use the calculated ratio
}

# specific to XGBoost: scale_pos_weight helps with imbalance
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Use StratifiedKFold for robust validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                           cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')

# Train on the FULL X_train (no downsampling)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\nBest Parameters found: {grid_search.best_params_}")

# 6. Evaluation
print("\nEvaluating model...")
y_train_pred = best_model.predict(X_train) 
y_test_pred = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Check for overfitting
diff = train_accuracy - test_accuracy
if diff > 0.1:
    print("Warning: Potential overfitting detected.")
else:
    print("Model generalization looks reasonable (Gap is small).")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# 7. Save Model
output_path_model = "customer_churn_model_xgboost.pkl"
model_data = {"model": best_model, "features_names": X.columns.tolist()}
with open(output_path_model, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nXGBoost model saved to '{output_path_model}'")
