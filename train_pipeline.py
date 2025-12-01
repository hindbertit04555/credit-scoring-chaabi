import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Load cleaned data
df = pd.read_csv(
    r"C:\Users\Administrator\Desktop\chaabi\MLPART\cleaned_dataset.csv",
    low_memory=False  # silences the DtypeWarning
)

print("Columns in cleaned_dataset.csv:")
print(df.columns.tolist())

TARGET = "credit_obtenu"

# 2. Choose features (only existing columns!)
feature_cols = [
    "BANQUE",
    "AGENCE",
    "CODE_VILLE",
    "VILLE",
    "FLAG_ETRANGER_RES_MAROC",
    "FLAG_PROPRIETAIRE_LOGEMENT",
    "SEXE",
    "COUNTRY",
    "RESIDENCE",
    "PROFESSION",
    "NOMBRE_ENFANT",
    "MARITAL_STATUS",
    "has_mobile",
    "has_net",
    "has_pack",
    "has_bancassurance",
]

X = df[feature_cols]
y = df[TARGET]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Numeric vs categorical
numeric_features = ["NOMBRE_ENFANT"]  # simple numeric field
categorical_features = [col for col in feature_cols if col not in numeric_features]

numeric_transformer = "passthrough"  # RF doesn’t need scaling
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features),
    ]
)

# 5. Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"  # because credit_obtenu is probably imbalanced
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# 6. Train
clf.fit(X_train, y_train)

# 7. Evaluate quickly
probas = clf.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, probas))
print(classification_report(y_test, clf.predict(X_test)))

# 8. Save pipeline (preprocessing + model)
joblib.dump(clf, "credit_pipeline.joblib")
print("Saved credit_pipeline.joblib")
