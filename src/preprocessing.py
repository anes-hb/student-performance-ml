import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_split(path):
    df = pd.read_csv(path)

    # Binary target: pass if G3 >=10
    df["pass"] = (df["G3"] >= 10).astype(int)

    X = df.drop(["G3", "pass"], axis=1)
    y = df["pass"]

    X = pd.get_dummies(X, drop_first=True)  # encode categorical

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val