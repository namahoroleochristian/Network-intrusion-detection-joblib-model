import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import ADASYN # New: Import ADASYN
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.colab import drive
from collections import Counter # New: for class distribution check
import matplotlib.pyplot as plt # New: for visualization

# --- 0. INITIAL SETUP & DATA LOADING (Same as previous script) ---
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"Could not mount Google Drive: {e}")

# --- CONFIGURATION ---
DATA_FOLDER = '/content/drive/MyDrive/NetworkFiles/' # <--- VERIFY THIS PATH
RESAMPLED_FILE = os.path.join(DATA_FOLDER, 'X_y_resampled_adasyn.parquet') # NEW: Resampled data path
MODEL_FILENAME = 'final_stacked_ensemble_model_resampled.joblib'

TRAIN_FILES = [f"traffic{i}.csv" for i in range(1, 8)]
TEST_FILE = "traffic8.csv"

# Column definitions (kept for data loading integrity)
FULL_COLUMN_NAMES = [
    'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto',
    'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state',
    'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts',
    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'tunnel_parents',
    'label', 'detailed-label'
]

COLUMNS_TO_DROP = [
    'uid', 'ts', 'id.orig_h', 'id.resp_h', 'detailed-label',
    'local_orig', 'local_resp', 'tunnel_parents'
]

CATEGORICAL_COLS = ["proto", "service", "conn_state", "history"]

NUMERIC_COLS = [
    'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
    'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]

# Load and clean function (kept for data loading integrity)
def load_and_clean_data(file_path):
    full_path = os.path.join(DATA_FOLDER, file_path)
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(full_path, sep='|', low_memory=False)
        if df.shape[1] < len(FULL_COLUMN_NAMES): return pd.DataFrame()
        df.columns = FULL_COLUMN_NAMES
        df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# --- 1. DATA PREPARATION FOR ADASYN (ONCE) ---
if not os.path.exists(RESAMPLED_FILE):
    print(f"\n--- ADASYN is required. {RESAMPLED_FILE} not found. ---")
    print("Loading and preparing original training data...")
    train_dfs = [load_and_clean_data(f) for f in TRAIN_FILES]
    train_data = pd.concat([df for df in train_dfs if not df.empty], ignore_index=True)
    print(f"Total Training Samples loaded: {len(train_data)}")

    # Remove rare labels
    label_counts = train_data['label'].value_counts()
    rare_labels = label_counts[label_counts < 3].index
    train_data = train_data[~train_data['label'].isin(rare_labels)].reset_index(drop=True)

    # Separate X & y
    le = LabelEncoder()
    y_train = le.fit_transform(train_data['label'])
    X_train = train_data.drop('label', axis=1)

    # OHE Encoding
    X_train_encoded = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)

    print("\nOriginal class distribution:")
    print(Counter(y_train))

    # --- EXECUTE ADASYN (THE SLOW STEP) ---
    print("\nApplying ADASYN oversampling (This will take the most time!)...")
    adasyn = ADASYN(random_state=42, n_neighbors=5, n_jobs=-1) # Use n_jobs=-1 to parallelize
    X_resampled, y_resampled = adasyn.fit_resample(X_train_encoded, y_train)

    print("\nNew class distribution after ADASYN:")
    print(Counter(y_resampled))

    # --- SAVE RESAMPLED DATA (THE TIME-SAVING STEP) ---
    X_resampled['target'] = y_resampled # Combine features and target for single save
    try:
        X_resampled.to_parquet(RESAMPLED_FILE, index=False)
        print(f"\n✅ Resampled Data (X and y) successfully saved to: {RESAMPLED_FILE}")
    except Exception as e:
        print(f"Could not save resampled data: {e}. Proceeding with in-memory data.")
        X_resampled = X_resampled.drop(columns=['target'])

else:
    # --- LOAD RESAMPLED DATA (To skip the ADASYN step next time) ---
    print(f"\n--- Resampled Data Found. Loading from {RESAMPLED_FILE} ---")
    resampled_df = pd.read_parquet(RESAMPLED_FILE)

    # Re-separate X and y
    y_resampled = resampled_df['target'].values
    X_resampled = resampled_df.drop(columns=['target'])

    # Re-fit LabelEncoder on original data to ensure inverse_transform works correctly
    # You must load the original training labels to fit the encoder correctly
    train_dfs = [load_and_clean_data(f) for f in TRAIN_FILES]
    train_data_full = pd.concat([df for df in train_dfs if not df.empty], ignore_index=True)
    le = LabelEncoder()
    le.fit(train_data_full['label'])
    del train_data_full # Clean up memory

    print(f"Loaded Resampled Features Shape: {X_resampled.shape}")
    print(f"Resampled Test Samples: {len(y_resampled)}")


# --- 2. PREPARE TEST DATA FOR EVALUATION ---
test_data = load_and_clean_data(TEST_FILE)
le = LabelEncoder().fit(pd.concat([test_data['label'], train_data['label']])) # Fit on combined original labels
y_test = le.transform(test_data['label'])
X_test = test_data.drop('label', axis=1)

# A. One-Hot Encoding (OHE) - Align with RESAMPLED columns
train_cols = X_resampled.columns
X_test_encoded = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)

# Align columns and fill missing (new) columns with 0
X_test_final = X_test_encoded.reindex(columns=train_cols, fill_value=0)

# B. Numerical Feature Scaling - Fit on RESAMPLED data (since that is what the model sees)
features_to_scale = [col for col in NUMERIC_COLS if col in X_resampled.columns]

if features_to_scale:
    scaler = StandardScaler()
    # Fit scaler on the ADASYN-resampled data (X_resampled)
    X_resampled[features_to_scale] = scaler.fit_transform(X_resampled[features_to_scale])
    # Transform test data
    X_test_final[features_to_scale] = scaler.transform(X_test_final[features_to_scale])
    print("\nNumerical features scaled successfully (based on resampled data).")

X_train_final = X_resampled # Final training data is the scaled and resampled data


# --- 3. TRAIN STACKING ENSEMBLE MODEL ---
print(f"\n--- Training Stacking Ensemble Model on Resampled Data ---")
print("This may still take a while due to the large, balanced dataset.")

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)), # Limited depth for speed
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, max_depth=5)), # Limited depth for speed
    ('lr_base', LogisticRegression(max_iter=500, random_state=42, solver='liblinear'))
]

final_estimator = LogisticRegression(solver='liblinear', random_state=42)

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=2, # Reduced CV to 2 for faster training
    n_jobs=-1 # Use all available cores
)

stacked_model.fit(X_train_final, y_resampled)
print("Training complete.")

# --- 4. SAVE THE TRAINED MODEL ---
model_save_path = os.path.join(DATA_FOLDER, MODEL_FILENAME)
try:
    joblib.dump(stacked_model, model_save_path)
    print(f"\n✅ Model successfully saved to: {model_save_path}")
except Exception as e:
    print(f"Could not save model: {e}")

# --- 5. EVALUATE ON DEDICATED TEST DATA (traffic8) ---
print(f"\n--- Evaluating Model on Completely Unseen Dataset ({TEST_FILE}) ---")
y_pred = stacked_model.predict(X_test_final)

# Generate report
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Classification Report on {TEST_FILE}:")

# Get original label names for the report
target_names = le.inverse_transform(np.unique(y_test))
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

print("\nEnsemble training and testing complete.")