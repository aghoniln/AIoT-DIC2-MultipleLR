"""
50 Startup Problem (Lasso Regression)
穷举 1~4 个特征的所有组合, 找出测试集 MSE 最小的 Lasso 模型
"""
import sys
from itertools import combinations

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 80)

# Step 1: Load data from URL
url = ("https://raw.githubusercontent.com/harimittapalli/"
       "Mulitple-Linear-Reggression/master/50_Startups.csv")
try:
    data = pd.read_csv(url)
except Exception as e:
    sys.exit(f"[错误] 数据载入失败: {e}")
print(data.head())
print("\nDataset shape:", data.shape)
data.info()

# Step 2: Preprocess data and one-hot encode State
data = pd.get_dummies(data, columns=["State"], drop_first=True, dtype=float)
print(data.head())
print("\nColumns after encoding:", list(data.columns))

X = data.drop(columns=["Profit"])
y = data["Profit"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# Step 3: Evaluate Lasso models for different numbers of features
results = []
for num_features in range(1, 5):
    best_mse, best_features = float("inf"), []
    for features in combinations(X.columns, num_features):
        feature_list = list(features)
        model = Lasso(alpha=1.0, max_iter=100000)
        model.fit(X_train[feature_list], y_train)
        mse = mean_squared_error(y_test, model.predict(X_test[feature_list]))
        if mse < best_mse:
            best_mse, best_features = mse, feature_list
    results.append({"num_features": num_features,
                    "features": best_features, "mse": best_mse})

results_df = pd.DataFrame(results)
print(results_df[["num_features", "features", "mse"]].to_string(index=False))

# Step 4: Best model summary
best_result = results_df.loc[results_df["mse"].idxmin()]
print("\nBest overall feature set:")
print("Number of features:", best_result["num_features"])
print("Features:", best_result["features"])
print("MSE:", best_result["mse"])

# Step 5: Final prediction and MSE
final_features = best_result["features"]
final_model = Lasso(alpha=1.0, max_iter=100000)
final_model.fit(X_train[final_features], y_train)
y_pred = final_model.predict(X_test[final_features])
final_mse = mean_squared_error(y_test, y_pred)
print(f"\nFinal test MSE: {final_mse:.4f}")
print(f"Final test RMSE: {final_mse ** 0.5:.4f}")
print(f"Selected features: {final_features}")
print("Coefficients:", dict(zip(final_features, final_model.coef_.round(4))))
