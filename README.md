# Day 2 — The End-to-End Machine Learning Cycle

**Week 1 · Classical Machine Learning**

> *"Skip complex abstraction tools — focus on writing clean, step-by-step Pandas and Scikit-Learn code to master the ML lifecycle."*

---

## 🎯 Today's Learning Strategy

Today's session is structured into two simple, hands-on parts:

1. **Part 1 — Demonstration (Titanic Dataset)**:  
   Watch how the full Machine Learning lifecycle works on the classic **Titanic dataset** using explicit, line-by-line Pandas and Scikit-Learn functions (Classification task: predicting survival `survived`).

2. **Part 2 — Intern Exercise (Housing Dataset)**:  
   Apply what you saw! Take a simple **Housing dataset** (`data/housing.csv`), fill missing values, one-hot encode text features, train a model, evaluate metrics, and save your model for inference.

---

## 📂 Day 2 Workspace Layout

All materials for today live inside `days/day-2/`:

```
days/day-2/
├── README.md                           # Main tutorial guide (this document)
├── LEARNING_LOG_DAY_02.md             # Daily log submission template
├── data/
│   ├── titanic.csv                     # Titanic dataset for Part 1 demonstration (800 rows)
│   └── housing.csv                     # Housing dataset for Part 2 exercise (500 rows)
├── models/                             # Saved model files (.joblib)
├── notebooks/
│   └── 02_full_ml_cycle_tutorial.ipynb # Interactive cell-by-cell Jupyter Notebook
└── src/
    ├── titanic_ml_cycle.py             # Part 1: Full Titanic ML cycle demo script
    ├── exercise.py                     # Part 2: Intern hands-on exercise (fill in the TODOs)
    └── solution.py                     # Reference solution for exercise.py
```

---

## 📖 Part 1: Demonstration — The 6 Stages on Titanic

In [src/titanic_ml_cycle.py](file:///Users/d-23-11714/Desktop/ai_ml_internship/days/day-2/src/titanic_ml_cycle.py), we step through every phase of Machine Learning using simple code:

### 1. Load Data & Define Problem
- **Task**: Classification
- **Target ($y$)**: `survived` (1 = Survived, 0 = Did not survive)
- **Features ($X$)**: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`

```python
import pandas as pd
df = pd.read_csv("data/titanic.csv")
```

### 2. Exploratory Data Analysis (EDA)
Inspect missing values and dataset shape:
```python
print(df.isnull().sum())
print(f"Overall Survival Rate: {df['survived'].mean()*100:.1f}%")
```

### 3. Simple Data Preprocessing (Pandas)
Instead of complex pipeline abstractions, we process data step-by-step:
```python
# A. Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# B. Convert text features into numbers (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# C. Separate features and target
X = df_encoded.drop(columns=['survived'])
y = df_encoded['survived']

# D. Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# E. Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. Model Training
Train a `LogisticRegression` model directly on the scaled training data:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

### 5. Model Evaluation
Evaluate accuracy and view the confusion matrix:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

preds = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, preds)*100:.1f}%")
```

### 6. Model Serialization & Inference
Save the model with `joblib`, load it back, and predict on a sample passenger:
```python
import joblib

joblib.dump(model, "models/titanic_model.joblib")

loaded_model = joblib.load("models/titanic_model.joblib")
sample_passenger = X_test_scaled[[0]]
prediction = loaded_model.predict(sample_passenger)[0]
```

To run the full demonstration:
```bash
python days/day-2/src/titanic_ml_cycle.py
```

---

## 🛠️ Part 2: Intern Exercise — Housing Dataset

Now it's your turn!

### Your Goal:
Take the **Housing dataset** (`data/housing.csv`) and complete the simple ML lifecycle.

### Steps to Complete:

1. **Step 1 — Run Demo Script**:
   Observe how `titanic_ml_cycle.py` runs and inspect its console logs.

2. **Step 2 — Complete `exercise.py`**:
    and fill in all `# TODO` markers!

3. **Step 3 — Run & Verify**:

   *(Need help after 20 minutes? Reach out!

4. **Step 4 — Log & Submit**:
   Fill in your daily log, commit your changes to Git, and push your branch!
