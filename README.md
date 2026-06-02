# UNICEF Malawi – Childhood Depression Prediction

A machine learning project built for **MATH 11205: Machine Learning in Python**.  
We use the UNICEF Multiple Indicator Cluster Survey (MICS) collected in Malawi (2019–2020) to build a classification model that predicts whether a child experiences depressive feelings, and to identify the key factors associated with childhood depression.

---

## Project Overview

UNICEF's MICS survey collects child, maternal, and household data across low- and middle-income countries. This project focuses on the Malawi subset and treats depression (survey variable `FCF26`) as a binary outcome:

- **0** – No depression (*"Never"*)
- **1** – Any depression (*"A few times a year / Monthly / Weekly / Daily"*)

The goal is to deliver a well-tuned, interpretable classification model that can support government officials and health workers in understanding and addressing childhood mental health.

---

## Repository Structure

```
.
├── project.ipynb            # Main report notebook (EDA, modelling, conclusions)
├── Data_pipeline.ipynb      # Standalone data pipeline (preprocessing only)
├── Data_pipeline_refined.ipynb  # Refined pipeline with full comments & best practices
├── unicef_malawi.csv        # Dataset (not to be shared publicly – see Data Policy)
└── README.md
```

> **Note:** The extended data sources referenced in the project description (`ExtendedDataSources/`) and questionnaire documents (`Questionnaires/`) are not included here but were used to inform feature definitions.

---

## Data Pipeline

The pipeline (`Data_pipeline_refined.ipynb`) processes the raw CSV into train/test feature matrices in eight steps:

| Step | Function | Description |
|------|----------|-------------|
| 1 | — | Load raw CSV |
| 2 | `build_target` | Binarise `FCF26` into 0/1 |
| 3 | `drop_unusable_rows` | Remove rows with missing target or child age |
| 4 | `fix_skip_patterns` | Resolve survey skip-logic missings (structural NaN → known value) |
| 5 | `engineer_features` | Create composite features (e.g. `discipline_count`, `dv_acceptance_score`) |
| 6 | `train_test_split` | Stratified 80/20 split |
| 7 | `apply_all_imputations` | Geographic group imputation (train-fitted only, no leakage) |
| 8 | `encode_binary` | YES/NO strings → 0/1 integers |

The output `X_train, X_test, y_train, y_test` is then passed to an **sklearn `Pipeline`** combining a `ColumnTransformer` (scaling, ordinal encoding, one-hot encoding) with a classifier.

### Feature Groups

| Group | Examples | Preprocessing |
|-------|----------|---------------|
| Numeric | `CB3` (age), `wscore` (wealth), `CL3` (labour hours) | Median impute → StandardScaler |
| Ordinal | `CB5A` (education level), `LS1` (happiness), `HC4` (floor material) | Mode impute → OrdinalEncoder → StandardScaler |
| Binary | `FCD2A-K` (discipline methods), `DV1A-E` (DV attitudes), `HC8-HC19` (assets) | Mode impute |
| Engineered | `discipline_count`, `dv_acceptance_score`, `total_labour_hours` | Median impute → StandardScaler |
| Nominal | `HH6/HH7` (geography), `HL4` (sex), `MSTATUS` (marital status) | Mode impute → OneHotEncoder |

---

## Getting Started

### Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Pipeline

```python
from Data_pipeline_refined import build_Xy, preprocessing_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = build_Xy('unicef_malawi.csv', verbose=True)

model = Pipeline([
    ('preprocess', preprocessing_pipeline()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
print(f'Test accuracy: {model.score(X_test, y_test):.4f}')
```

### Running the Full Report

Open and run `project.ipynb` in order. The final cell converts the notebook to PDF:

```bash
jupyter nbconvert --to pdf project.ipynb
```

---

## Modelling Approach

- **Baseline model**: Logistic Regression (interpretable, establishes a performance floor)
- **Final model**: to be determined via cross-validated hyperparameter tuning in `project.ipynb`
- **Validation**: stratified train/test split + k-fold cross-validation on the training set
- **Evaluation metrics**: accuracy, precision, recall, F1-score, ROC-AUC (class imbalance considered)

---

## Data Policy

The UNICEF MICS dataset is provided under a data agreement.  
**Do not share the CSV file publicly** — keep any version control repository private and do not upload the data to any generative AI tool.

---

## Generative AI

Generative AI tools were used as a coding assistant for debugging, comment writing, and code refinement. All modelling decisions, interpretations, and written analysis are the work of the project team. A full statement is included in `project.ipynb`.

---

## Team

*Add contributor names here.*

## References

- UNICEF MICS Programme: https://mics.unicef.org  
- Malawi 2019–2020 MICS survey documentation
