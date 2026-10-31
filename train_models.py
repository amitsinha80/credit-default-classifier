"""Train five classifiers on the UCI 'Default of Credit Card Clients' dataset."""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation import compute_metrics, positive_class_scores

RAW_DATA_PATH = "data/default of credit card clients.xls"
TEST_DATA_PATH = "test_data.csv"
MODEL_DIR = Path("model")
TARGET = "default_next_month"
RANDOM_STATE = 17  # keep the same number you chose in part 1


def load_and_clean(path):
    """Read the raw UCI file and apply the cleaning decided during exploration."""
    df = pd.read_excel(path, header=1, engine="calamine")  # row 0 holds X1..X23 labels
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": TARGET, "PAY_0": "PAY_1"})

    # Undocumented codes -> the 'others' category
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def build_models():
    """Return the five classifiers; scale-sensitive ones get a StandardScaler."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
    }


def model_file_name(name):
    """'Logistic Regression' -> 'logistic_regression.pkl'"""
    return name.lower().replace(" ", "_") + ".pkl"


def main():
    df = load_and_clean(RAW_DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Features: {X.shape[1]}")
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Default rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    # Save the held-out test set (features + target) for the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved {TEST_DATA_PATH}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    results = []

    for name, model in build_models().items():
        start = time.time()
        model.fit(X_train, y_train)  # train on the training set only
        train_secs = time.time() - start

        # Evaluate on the held-out test set
        y_pred = model.predict(X_test)
        y_score = positive_class_scores(model, X_test)
        results.append({"Model": name, **compute_metrics(y_test, y_pred, y_score)})

        joblib.dump(model, MODEL_DIR / model_file_name(name))
        print(f"{name:<20} trained in {train_secs:6.1f} s -> saved {model_file_name(name)}")

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    results_df.to_csv(MODEL_DIR / "metrics.csv")

    print("\nTest-set comparison:")
    print(results_df.to_string())


if __name__ == "__main__":
    main()