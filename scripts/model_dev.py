import requests
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import joblib

warnings.filterwarnings("ignore")

# Load the trimmed dataset from Google Drive
file_id = '1_cB6mECcscH2gH-Q4VGXr8lFjbfdh7qt'
url = f'https://drive.google.com/uc?id={file_id}'
response = requests.get(url)

if response.status_code == 200:
    trimmed_data = pd.read_csv(StringIO(response.text))
else:
    raise ValueError(f"Failed to fetch the file. Status code: {response.status_code}")

# Feature Selection
selected_features = [
    "Severity", "Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)", "Humidity(%)", 
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Weather_Condition"
]
data = trimmed_data[selected_features]

# Handle categorical data
## One-hot encode categorical features
categorical_features = ["Weather_Condition"]
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Define features and target
X = data.drop(columns=["Severity"]).reset_index(drop=True)
y = data["Severity"].reset_index(drop=True)

# Adjust target classes for XGBoost compatibility
y_xgb = y - 1

# Train-test split with stratification
def stratified_split(X, y, train_size=0.75, val_size=0.15, test_size=0.10, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=random_state)
    val_fraction = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_fraction, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train_xgb, X_val_xgb, X_test_xgb, y_train_xgb, y_val_xgb, y_test_xgb = stratified_split(X, y_xgb)

# Handle class imbalance using SMOTE with memory efficiency
batch_size = 50000  # Reduced batch size for SMOTE to handle memory constraints
smote = SMOTE(random_state=42)

if len(X_train_xgb) > batch_size:
    sampled_indices = np.random.choice(len(X_train_xgb), batch_size, replace=False)
    X_train_sampled = X_train_xgb.iloc[sampled_indices].reset_index(drop=True)
    y_train_sampled = y_train_xgb.iloc[sampled_indices].reset_index(drop=True)
    X_train_xgb_resampled, y_train_xgb_resampled = smote.fit_resample(X_train_sampled, y_train_sampled)
else:
    X_train_xgb_resampled, y_train_xgb_resampled = smote.fit_resample(X_train_xgb, y_train_xgb)

# Function to limit batch size for model training
def train_model_in_batches(model, X_train, y_train, param_grid, cv=3):
    batch_size = 100000
    if len(X_train) > batch_size:
        sampled_indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_train_sampled = X_train.iloc[sampled_indices].reset_index(drop=True)
        y_train_sampled = y_train.iloc[sampled_indices].reset_index(drop=True)
    else:
        X_train_sampled, y_train_sampled = X_train, y_train

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_sampled, y_train_sampled)
    return grid_search

# Model Training and Hyperparameter Tuning
## Initialize XGBoost model
xgb_model = XGBClassifier(eval_metric="mlogloss", random_state=42)

## Define parameter grid for XGBoost tuning
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Train XGBoost
print("Training XGBoost...")
xgb_grid_search = train_model_in_batches(xgb_model, X_train_xgb_resampled, y_train_xgb_resampled, xgb_param_grid)
print("Best XGBoost Parameters:", xgb_grid_search.best_params_)
joblib.dump(xgb_grid_search.best_estimator_, 'best_xgb_model.pkl')  # Save the trained model

# Model Evaluation
## Evaluate on validation set
xgb_best_model = xgb_grid_search.best_estimator_

xgb_val_predictions = xgb_best_model.predict(X_val_xgb)

print("\nXGBoost Validation Report:")
print(classification_report(y_val_xgb, xgb_val_predictions))

# Evaluate on test set
xgb_test_predictions = xgb_best_model.predict(X_test_xgb)

print("\nXGBoost Test Report:")
print(classification_report(y_test_xgb, xgb_test_predictions))
