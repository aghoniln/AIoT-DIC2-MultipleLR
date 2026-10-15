"""
Part 2: Forward Selection 特征选择
"""
import sys
from io import StringIO

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate

URL = ("https://gist.githubusercontent.com/GaneshSparkz/"
       "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
       "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv")

# Step 1: 载入数据
try:
    response = requests.get(URL, timeout=15)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
except Exception as e:
    sys.exit(f"[错误] 数据获取/解析失败: {e}")

X = data.drop("Profit", axis=1)
y = data["Profit"]
X_encoded = pd.get_dummies(X, columns=["State"], drop_first=True, dtype=float)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0)


# Step 3: 评估函数
def evaluate_performance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


# Step 2: 前向选择
def forward_selection(X_train, X_test, y_train, y_test, max_features):
    selected, results = [], []
    for i in range(1, max_features + 1):
        best = (np.inf, -np.inf, None)  # (rmse, r2, feature)
        for f in X_train.columns.difference(selected):
            cols = selected + [f]
            m = LinearRegression().fit(X_train[cols], y_train)
            rmse, r2 = evaluate_performance(y_test, m.predict(X_test[cols]))
            if rmse < best[0]:  # RMSE 最小 ⇔ R² 最大 (同一测试集)
                best = (rmse, r2, f)
        selected.append(best[2])
        results.append({"Number of Features": i,
                        "Selected Features": ", ".join(selected),
                        "RMSE": best[0], "R-squared": best[1]})
    return results


# Step 4: 1~5 个特征
n_max = min(5, X_train.shape[1])
results_df = pd.DataFrame(
    forward_selection(X_train, X_test, y_train, y_test, n_max))

# Step 5: Markdown 表格
md = tabulate(results_df, headers="keys", tablefmt="pipe",
              showindex=False, floatfmt=(".0f", "", ".6f", ".6f"))
print(md)
with open("forward_selection.md", "w", encoding="utf-8") as fp:
    fp.write(md + "\n")

# 图表
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(results_df["Number of Features"], results_df["RMSE"], marker="o")
ax[0].set(title="RMSE by Number of Features",
          xlabel="Number of Features", ylabel="RMSE")
ax[1].plot(results_df["Number of Features"], results_df["R-squared"], marker="o")
ax[1].set(title="R-squared by Number of Features",
          xlabel="Number of Features", ylabel="R-squared")
plt.tight_layout()
plt.savefig("forward_selection.png", dpi=120)
print("图已保存: forward_selection.png")
