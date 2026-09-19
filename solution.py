import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

id_col = "candidate_id"

target_cols = [c for c in train.columns if c not in test.columns]
target_col = target_cols[0] if target_cols else train.columns[-1]

features = [c for c in train.columns if c in test.columns and c != id_col]

X = train[features].copy()
y = train[target_col].copy()
X_test = test[features].copy()

# Categorical text handling
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[col] = oe.fit_transform(X[col].astype(str).values.reshape(-1, 1))
    X_test[col] = oe.transform(X_test[col].astype(str).values.reshape(-1, 1))

# Hyperparameters update for higher accuracy
if len(np.unique(y.dropna())) < 20:
    model = HistGradientBoostingClassifier(
        max_iter=300, 
        learning_rate=0.05, 
        max_depth=6, 
        l2_regularization=1.0, 
        random_state=42
    )
    model.fit(X, y)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.predict(X_test)
else:
    model = HistGradientBoostingRegressor(
        max_iter=300, 
        learning_rate=0.05, 
        max_depth=6, 
        l2_regularization=1.0, 
        random_state=42
    )
    model.fit(X, y)
    scores = model.predict(X_test)

test_res = test[[id_col]].copy()
test_res['score'] = scores

test_res = test_res.sort_values(by='score', ascending=False).reset_index(drop=True)

top_500 = test_res.head(500).copy()
top_500['rank'] = range(1, len(top_500) + 1)

submission = top_500[['rank', 'candidate_id']]
submission.to_csv("submission.csv", index=False)
print("Updated high-accuracy submission.csv successfully created!")