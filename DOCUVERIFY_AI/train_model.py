# ============================================================
# DOCUVERIFY AI
# TRAINING + EDA + MODEL EVALUATION
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "docuverify_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "docuverify_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_names.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "model_metrics.pkl"
)

EDA_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "eda"
)

os.makedirs(
    EDA_FOLDER,
    exist_ok=True
)


# ============================================================
# FEATURE ORDER
# ============================================================
# IMPORTANT:
# This order MUST match app.py

FEATURE_COLUMNS = [
    "text_length",
    "text_quality",
    "metadata_score",
    "structure_score",
    "field_score",
    "suspicious_keywords",
    "repeated_lines",
    "abnormal_symbols",
    "missing_fields",
    "metadata_anomaly",
    "tampering_indicators",
    "risk_score"
]


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("                    DOCUVERIFY AI")
print("        DOCUMENT VERIFICATION ML TRAINING")
print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

print("\n[1] Checking dataset...")

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        "\nDataset not found:\n"
        f"{DATASET_PATH}\n\n"
        "Make sure docuverify_dataset.csv is inside "
        "the DOCUVERIFY_AI project folder."
    )

print("Dataset found:")
print(DATASET_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n[2] Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print("Dataset loaded successfully.")


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET DETAILS")
print("=" * 70)

print(
    "Rows    :",
    df.shape[0]
)

print(
    "Columns :",
    df.shape[1]
)

print("\nColumns:")

for column in df.columns:

    print(
        " -",
        column
    )


print("\nFirst 5 rows:")

print(
    df.head()
)


# ============================================================
# TARGET COLUMN
# ============================================================

TARGET_COLUMN = "label"

if TARGET_COLUMN not in df.columns:

    raise ValueError(
        "\nTarget column 'label' was not found.\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print(
    missing_values
)

total_missing = int(
    missing_values.sum()
)

print(
    "\nTotal missing values:",
    total_missing
)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_count = int(
    df.duplicated().sum()
)

print(
    "\nDuplicate rows:",
    duplicate_count
)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "Duplicate rows removed."
    )

else:

    print(
        "No duplicate rows found."
    )


# ============================================================
# TARGET CLEANING
# ============================================================

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE INVALID TARGET VALUES
# ============================================================

valid_labels = {
    "Fake",
    "Genuine",
    "Suspicious"
}

invalid_labels = set(
    df[TARGET_COLUMN].unique()
) - valid_labels


if invalid_labels:

    print(
        "\nWarning: Unknown target labels found:"
    )

    print(
        invalid_labels
    )

    df = df[
        df[TARGET_COLUMN].isin(
            valid_labels
        )
    ].copy()


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = (
    df[TARGET_COLUMN]
    .value_counts()
)

print(
    label_counts
)

print("\nPercentage Distribution:")

label_percentage = (
    df[TARGET_COLUMN]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)

print(
    label_percentage
)


# ============================================================
# EDA 1 - LABEL DISTRIBUTION
# ============================================================

print(
    "\n[3] Creating EDA graphs..."
)

plt.figure(
    figsize=(8, 5)
)

label_counts.plot(
    kind="bar"
)

plt.title(
    "DOCUVERIFY AI - Document Label Distribution"
)

plt.xlabel(
    "Document Label"
)

plt.ylabel(
    "Number of Documents"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_FOLDER,
        "label_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

print(
    "\n[4] Checking feature columns..."
)

missing_features = [

    feature

    for feature in FEATURE_COLUMNS

    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "\nRequired feature columns are missing:\n"
        + "\n".join(
            f" - {feature}"
            for feature in missing_features
        )
        + "\n\n"
        "Your dataset must contain the same "
        "features generated by app.py."
    )


print(
    "All required features found."
)


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for feature in FEATURE_COLUMNS:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# ============================================================
# HANDLE MISSING FEATURE VALUES
# ============================================================

df[FEATURE_COLUMNS] = (
    df[FEATURE_COLUMNS]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# REMOVE EMPTY DATASET
# ============================================================

if len(df) == 0:

    raise ValueError(
        "Dataset is empty after preprocessing."
    )


# ============================================================
# EDA 2 - RISK SCORE DISTRIBUTION
# ============================================================

if "risk_score" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df["risk_score"],
        bins=30
    )

    plt.title(
        "DOCUVERIFY AI - Risk Score Distribution"
    )

    plt.xlabel(
        "Risk Score"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "risk_score_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# EDA 3 - RISK SCORE BY LABEL
# ============================================================

if "risk_score" in df.columns:

    plt.figure(
        figsize=(8, 5)
    )

    grouped_data = []

    grouped_labels = []

    for label in [
        "Genuine",
        "Suspicious",
        "Fake"
    ]:

        values = df.loc[
            df[TARGET_COLUMN] == label,
            "risk_score"
        ]

        if len(values) > 0:

            grouped_data.append(
                values
            )

            grouped_labels.append(
                label
            )

    if grouped_data:

        plt.boxplot(
            grouped_data,
            labels=grouped_labels
        )

    plt.title(
        "DOCUVERIFY AI - Risk Score by Label"
    )

    plt.xlabel(
        "Document Label"
    )

    plt.ylabel(
        "Risk Score"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "risk_score_by_label.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# EDA 4 - CORRELATION HEATMAP
# ============================================================

if len(FEATURE_COLUMNS) >= 2:

    correlation = df[
        FEATURE_COLUMNS
    ].corr()

    plt.figure(
        figsize=(12, 9)
    )

    plt.imshow(
        correlation,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(FEATURE_COLUMNS)),
        FEATURE_COLUMNS,
        rotation=90
    )

    plt.yticks(
        range(len(FEATURE_COLUMNS)),
        FEATURE_COLUMNS
    )

    plt.title(
        "DOCUVERIFY AI - Feature Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            "correlation_heatmap.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# EDA 5 - FEATURE DISTRIBUTIONS
# ============================================================

selected_features = [

    "text_quality",

    "metadata_score",

    "structure_score",

    "field_score",

    "tampering_indicators"
]


for feature in selected_features:

    if feature not in df.columns:

        continue

    plt.figure(
        figsize=(8, 5)
    )

    for label in [
        "Genuine",
        "Suspicious",
        "Fake"
    ]:

        values = df.loc[
            df[TARGET_COLUMN] == label,
            feature
        ]

        if len(values) > 0:

            plt.hist(
                values,
                bins=20,
                alpha=0.5,
                label=label
            )

    plt.title(
        f"{feature.replace('_', ' ').title()} Distribution"
    )

    plt.xlabel(
        feature.replace(
            "_",
            " "
        ).title()
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            EDA_FOLDER,
            f"{feature}_distribution.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# PREPARE X AND Y
# ============================================================

print(
    "\n[5] Preparing training data..."
)

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    TARGET_COLUMN
].copy()


print(
    "\nFeatures used:"
)

for index, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f" {index:02d}. {feature}"
    )


# ============================================================
# LABEL ENCODING
# ============================================================

print(
    "\n[6] Encoding target labels..."
)

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


print(
    "\nEncoded classes:"
)

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f" {index} -> {class_name}"
    )


# ============================================================
# CHECK CLASS COUNT
# ============================================================

class_counts = np.bincount(
    y_encoded
)

if len(class_counts) < 2:

    raise ValueError(
        "At least two document classes are required."
    )


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print(
    "\n[7] Splitting dataset..."
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print(
    "\n[8] Training Random Forest model..."
)

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=4,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Model training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\n[9] Evaluating model..."
)

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PERFORMANCE OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "                    MODEL PERFORMANCE"
)

print(
    "=" * 70
)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 70
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(
            len(label_encoder.classes_)
        ),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(
        len(label_encoder.classes_)
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "DOCUVERIFY AI - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(label_encoder.classes_)
)

plt.xticks(
    tick_marks,
    label_encoder.classes_,
    rotation=20
)

plt.yticks(
    tick_marks,
    label_encoder.classes_
)

threshold = (
    cm.max() / 2
    if cm.size > 0
    else 0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame(

    {

        "feature":
            FEATURE_COLUMNS,

        "importance":
            model.feature_importances_

    }

)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    "\n" + "=" * 70
)

print(
    "FEATURE IMPORTANCE"
)

print(
    "=" * 70
)

print(
    feature_importance
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "DOCUVERIFY AI - Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        EDA_FOLDER,
        "feature_importance.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\n[10] Saving model files..."
)


joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    label_encoder,
    ENCODER_PATH
)


joblib.dump(
    FEATURE_COLUMNS,
    FEATURE_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_data = {

    "accuracy":
        round(
            accuracy * 100,
            2
        ),

    "precision":
        round(
            precision * 100,
            2
        ),

    "recall":
        round(
            recall * 100,
            2
        ),

    "f1_score":
        round(
            f1 * 100,
            2
        ),

    "dataset_rows":
        int(len(df)),

    "dataset_columns":
        int(len(df.columns)),

    "feature_count":
        int(len(FEATURE_COLUMNS)),

    "classes":
        list(
            label_encoder.classes_
        )
}


joblib.dump(
    metrics_data,
    METRICS_PATH
)


# ============================================================
# VERIFY SAVED FILES
# ============================================================

print(
    "\nSaved files:"
)

print(
    "✓",
    MODEL_PATH
)

print(
    "✓",
    ENCODER_PATH
)

print(
    "✓",
    FEATURE_PATH
)

print(
    "✓",
    METRICS_PATH
)


# ============================================================
# VERIFY MODEL LOAD
# ============================================================

print(
    "\n[11] Verifying saved model..."
)

try:

    test_model = joblib.load(
        MODEL_PATH
    )

    test_encoder = joblib.load(
        ENCODER_PATH
    )

    test_features = joblib.load(
        FEATURE_PATH
    )

    print(
        "✓ Model loaded successfully."
    )

    print(
        "✓ Label encoder loaded successfully."
    )

    print(
        "✓ Feature names loaded successfully."
    )

    print(
        "Saved feature order:"
    )

    for index, feature in enumerate(
        test_features,
        start=1
    ):

        print(
            f" {index:02d}. {feature}"
        )

except Exception as error:

    print(
        "Model verification failed:"
    )

    print(
        error
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "             TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "Dataset rows       :",
    len(df)
)

print(
    "Dataset columns    :",
    len(df.columns)
)

print(
    "Features used      :",
    len(FEATURE_COLUMNS)
)

print(
    "Classes            :",
    list(
        label_encoder.classes_
    )
)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Precision          : {precision * 100:.2f}%"
)

print(
    f"Recall             : {recall * 100:.2f}%"
)

print(
    f"F1 Score           : {f1 * 100:.2f}%"
)

print(
    "\nGenerated EDA files:"
)

print(
    " - label_distribution.png"
)

print(
    " - risk_score_distribution.png"
)

print(
    " - risk_score_by_label.png"
)

print(
    " - correlation_heatmap.png"
)

print(
    " - text_quality_distribution.png"
)

print(
    " - metadata_score_distribution.png"
)

print(
    " - structure_score_distribution.png"
)

print(
    " - field_score_distribution.png"
)

print(
    " - tampering_indicators_distribution.png"
)

print(
    " - confusion_matrix.png"
)

print(
    " - feature_importance.png"
)

print(
    "\nGenerated model files:"
)

print(
    " - docuverify_model.pkl"
)

print(
    " - label_encoder.pkl"
)

print(
    " - feature_names.pkl"
)

print(
    " - model_metrics.pkl"
)

print(
    "=" * 70
)