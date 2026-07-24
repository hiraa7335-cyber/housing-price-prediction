import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("  DAY 2 HANDS-ON EXERCISE: ML LIFECYCLE ON HOUSING DATASET")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STAGE 1: Load Dataset
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: Load Data & Inspect ---")

    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "housing.csv"
    )

    df = pd.read_csv(data_path)

    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("\nFirst 5 rows:")
    print(df.head())

    # -------------------------------------------------------------------------
    # STAGE 2: EDA & Missing Values
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: EDA & Missing Values ---")

    print("\nMissing Values:")
    print(df.isnull().sum())

    df["age"] = df["age"].fillna(df["age"].median())

    # -------------------------------------------------------------------------
    # STAGE 3: Encoding & Train/Test Split
    # -------------------------------------------------------------------------
    print("\n--- STAGE 3: Encoding & Train/Test Split ---")

    df_encoded = pd.get_dummies(
        df,
        columns=["location"],
        dtype=int
    )

    X = df_encoded.drop(columns=["price"])
    y = df_encoded["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Train Size: {len(X_train)}")
    print(f"Test Size : {len(X_test)}")

    # -------------------------------------------------------------------------
    # STAGE 4: Train Model
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: Model Training ---")

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("Model trained successfully!")

    # -------------------------------------------------------------------------
    # STAGE 5: Evaluation
    # -------------------------------------------------------------------------
    print("\n--- STAGE 5: Evaluation ---")

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")

    # -------------------------------------------------------------------------
    # STAGE 6: Save Model
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: Save Model & Test Prediction ---")

    model_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models"
    )

    os.makedirs(model_dir, exist_ok=True)

    save_path = os.path.join(
        model_dir,
        "my_housing_model.joblib"
    )

    joblib.dump(model, save_path)

    loaded_model = joblib.load(save_path)

    sample_house = X_test.iloc[[0]]

    predicted_price = loaded_model.predict(sample_house)[0]

    actual_price = y_test.iloc[0]

    print(f"\nPredicted Price : ${predicted_price:,.2f}")
    print(f"Actual Price    : ${actual_price:,.2f}")

    print("\n" + "=" * 70)
    print("CONGRATULATIONS: You successfully completed the ML lifecycle!")
    print("=" * 70)


if __name__ == "__main__":
    main()