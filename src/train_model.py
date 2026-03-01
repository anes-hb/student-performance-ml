from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from preprocessing import load_and_split

X_train, X_val, y_train, y_val = load_and_split("../data/student.csv")

model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

preds = model.predict(X_val)
print("Validation F1:", f1_score(y_val, preds))