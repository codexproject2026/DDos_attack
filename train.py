# ============================================================
# EXPLAINABLE AI BASED INTRUSION DETECTION
# RANDOM FOREST TRAINING
# Dataset : ML-EdgeIIoT-dataset.csv
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/ML-EdgeIIoT-dataset.csv"

print("\n============================================")
print("LOADING DATASET")
print("============================================")

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded successfully.")
print("Dataset Shape:", df.shape)


# ============================================================
# 3. SELECT 20 INPUT FEATURES
# ============================================================

FEATURES = [

    "arp.opcode",
    "arp.hw.size",

    "icmp.checksum",
    "icmp.seq_le",

    "http.content_length",
    "http.response",

    "tcp.ack",
    "tcp.checksum",

    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",

    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",

    "udp.stream",
    "udp.time_delta",

    "dns.qry.name.len",

    "mqtt.len",

    "mbtcp.len"
]


print("\n============================================")
print("SELECTED 20 FEATURES")
print("============================================")

for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + ["Attack_label", "Attack_type"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns found:")
    print(missing_columns)

    raise SystemExit


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

print("\n============================================")
print("DATA PREPROCESSING")
print("============================================")

X = df[FEATURES].copy()

# Convert all selected features to numeric.
# Any non-numeric/invalid values become NaN.
for column in FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# Replace infinity with NaN
X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# Replace missing values using median
for column in FEATURES:

    median_value = X[column].median()

    # If an entire column is empty
    if pd.isna(median_value):
        median_value = 0

    X[column] = X[column].fillna(median_value)


print("Missing values after preprocessing:")
print(X.isnull().sum().sum())


# Save feature names
joblib.dump(
    FEATURES,
    "models/selected_features.pkl"
)

print("Selected features saved.")


# ============================================================
# 6. BINARY INTRUSION DETECTION
# Attack_label:
# 0 = Normal
# 1 = Attack
# ============================================================

print("\n\n============================================")
print("BINARY RANDOM FOREST MODEL")
print("NORMAL VS ATTACK")
print("============================================")

y_binary = pd.to_numeric(
    df["Attack_label"],
    errors="coerce"
)

valid_binary = y_binary.notna()

X_binary = X.loc[valid_binary].copy()
y_binary = y_binary.loc[valid_binary].astype(int)


print("\nClass Distribution:")
print(y_binary.value_counts())


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_binary,
    y_binary,

    test_size=0.20,
    random_state=42,

    # Important because dataset is imbalanced
    stratify=y_binary
)


print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ============================================================
# 8. TRAIN BINARY RANDOM FOREST
# ============================================================

print("\nTraining Binary Random Forest...")

binary_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1,

    # Helps because Attack and Normal counts are unequal
    class_weight="balanced",

    max_features="sqrt"
)


binary_model.fit(
    X_train,
    y_train
)


print("Binary model training completed.")


# ============================================================
# 9. BINARY PREDICTION
# ============================================================

y_pred = binary_model.predict(X_test)

y_probability = binary_model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. BINARY EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)


print("\n============================================")
print("BINARY MODEL RESULTS")
print("============================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc_score:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0
    )
)


# ============================================================
# 11. SAVE BINARY MODEL
# ============================================================

joblib.dump(
    binary_model,
    "models/random_forest_binary.pkl"
)

print("\nBinary model saved:")
print("models/random_forest_binary.pkl")


# ============================================================
# 12. BINARY CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Random Forest - Normal vs Attack Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/binary_confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "Binary Confusion Matrix saved:"
    " results/binary_confusion_matrix.png"
)


# ============================================================
# 13. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Random Forest (AUC = {auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Normal vs Attack"
)

plt.legend(loc="lower right")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/binary_roc_curve.png",
    dpi=300
)

plt.close()

print(
    "ROC Curve saved:"
    " results/binary_roc_curve.png"
)


# ============================================================
# 14. SAVE BINARY METRICS
# ============================================================

binary_metrics = {

    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": auc_score

}

binary_metrics_df = pd.DataFrame(
    [binary_metrics]
)

binary_metrics_df.to_csv(
    "results/binary_metrics.csv",
    index=False
)


# ============================================================
# 15. ATTACK TYPE CLASSIFICATION
# ============================================================

print("\n\n============================================")
print("ATTACK TYPE RANDOM FOREST MODEL")
print("============================================")


# Only attack rows are used here.
# Normal rows are excluded because first model already
# determines whether traffic is Normal or Attack.

attack_mask = (
    pd.to_numeric(
        df["Attack_label"],
        errors="coerce"
    ) == 1
)


X_attack = X.loc[attack_mask].copy()

y_attack = (
    df.loc[attack_mask, "Attack_type"]
    .astype(str)
    .str.strip()
)


print("\nAttack Type Distribution:")
print(y_attack.value_counts())


# ============================================================
# 16. ATTACK TYPE TRAIN TEST SPLIT
# ============================================================

X_attack_train, X_attack_test, \
y_attack_train, y_attack_test = train_test_split(

    X_attack,
    y_attack,

    test_size=0.20,

    random_state=42,

    stratify=y_attack
)


print(
    "\nAttack Training Samples :",
    len(X_attack_train)
)

print(
    "Attack Testing Samples  :",
    len(X_attack_test)
)


# ============================================================
# 17. TRAIN ATTACK TYPE RANDOM FOREST
# ============================================================

print("\nTraining Attack Type Random Forest...")


attack_model = RandomForestClassifier(

    n_estimators=250,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced",

    max_features="sqrt"
)


attack_model.fit(
    X_attack_train,
    y_attack_train
)


print("Attack Type model training completed.")


# ============================================================
# 18. ATTACK TYPE PREDICTION
# ============================================================

attack_pred = attack_model.predict(
    X_attack_test
)


# ============================================================
# 19. ATTACK TYPE METRICS
# ============================================================

attack_accuracy = accuracy_score(
    y_attack_test,
    attack_pred
)

attack_precision = precision_score(

    y_attack_test,
    attack_pred,

    average="weighted",
    zero_division=0
)

attack_recall = recall_score(

    y_attack_test,
    attack_pred,

    average="weighted",
    zero_division=0
)

attack_f1 = f1_score(

    y_attack_test,
    attack_pred,

    average="weighted",
    zero_division=0
)


print("\n============================================")
print("ATTACK TYPE MODEL RESULTS")
print("============================================")

print(
    f"Accuracy  : "
    f"{attack_accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{attack_precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{attack_recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{attack_f1 * 100:.2f}%"
)


print("\nAttack Type Classification Report:\n")

print(
    classification_report(
        y_attack_test,
        attack_pred,
        zero_division=0
    )
)


# ============================================================
# 20. SAVE ATTACK TYPE MODEL
# ============================================================

joblib.dump(
    attack_model,
    "models/random_forest_attack_type.pkl"
)

print("\nAttack Type model saved:")
print("models/random_forest_attack_type.pkl")


# ============================================================
# 21. ATTACK TYPE CONFUSION MATRIX
# ============================================================

attack_labels = sorted(
    y_attack.unique()
)

attack_cm = confusion_matrix(

    y_attack_test,
    attack_pred,

    labels=attack_labels
)


fig, ax = plt.subplots(
    figsize=(15, 13)
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=attack_cm,
    display_labels=attack_labels
)

disp.plot(

    ax=ax,

    cmap="Blues",

    xticks_rotation=90,

    values_format="d",

    colorbar=False
)


plt.title(
    "Random Forest - Attack Type Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/attack_type_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Attack Type Confusion Matrix saved:"
    " results/attack_type_confusion_matrix.png"
)


# ============================================================
# 22. SAVE ATTACK TYPE METRICS
# ============================================================

attack_metrics = {

    "Accuracy": attack_accuracy,
    "Precision Weighted": attack_precision,
    "Recall Weighted": attack_recall,
    "F1 Weighted": attack_f1

}


pd.DataFrame(
    [attack_metrics]
).to_csv(

    "results/attack_type_metrics.csv",

    index=False
)


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance":
        binary_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n============================================")
print("FEATURE IMPORTANCE")
print("============================================")

print(
    feature_importance.to_string(
        index=False
    )
)


feature_importance.to_csv(

    "results/feature_importance.csv",

    index=False
)


# ============================================================
# 24. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.barh(

    feature_importance["Feature"][::-1],

    feature_importance["Importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Network Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(

    "results/feature_importance.png",

    dpi=300,

    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n\n============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("============================================")

print("\nSaved Models:")

print(
    "1. models/random_forest_binary.pkl"
)

print(
    "2. models/random_forest_attack_type.pkl"
)

print(
    "3. models/selected_features.pkl"
)


print("\nSaved Results:")

print(
    "1. results/binary_confusion_matrix.png"
)

print(
    "2. results/binary_roc_curve.png"
)

print(
    "3. results/attack_type_confusion_matrix.png"
)

print(
    "4. results/feature_importance.png"
)

print(
    "5. results/binary_metrics.csv"
)

print(
    "6. results/attack_type_metrics.csv"
)

print(
    "7. results/feature_importance.csv"
)

print("\nNext Step:")
print(
    "Prediction + Confidence + SHAP"
)