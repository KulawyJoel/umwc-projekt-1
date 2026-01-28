import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    X = df.drop("label", axis=1)
    y = df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))
    joblib.dump(clf, out_dir / "model.joblib")
    print(f"✔ Model zapisany. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
