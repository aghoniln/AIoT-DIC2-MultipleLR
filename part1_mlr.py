"""
Part 1: 50 Startups 多元线性回归 (CRISP-DM)
"""
import sys
from io import StringIO

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

URL = ("https://github.com/harimittapalli/Mulitple-Linear-Reggression/"
       "raw/master/50_Startups.csv")
REQUIRED = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]

# ---------- Step 1: 数据获取 (Data Understanding) ----------
try:
    response = requests.get(URL, timeout=15)
except requests.RequestException as e:
    sys.exit(f"[错误] 网络请求失败: {e}")
if response.status_code != 200:
    sys.exit(f"[错误] 获取数据失败, status code: {response.status_code}")

try:
    data = pd.read_csv(StringIO(response.text))
except Exception as e:
    sys.exit(f"[错误] CSV 解析失败: {e}")

missing = set(REQUIRED) - set(data.columns)
if missing:
    sys.exit(f"[错误] 缺少字段: {missing}")

print("前 5 行:\n", data.head())
print("列名:", list(data.columns))
print("缺失值:\n", data.isnull().sum())
data = data.dropna()

# ---------- Step 2: 数据准备 ----------
X = data.drop("Profit", axis=1)
y = data["Profit"]
# drop='first' 直接丢弃一个虚拟变量列, 避免虚拟变量陷阱
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), ["State"])],
    remainder="passthrough")
X_enc = ct.fit_transform(X)
print("编码后特征:", list(ct.get_feature_names_out()))

# ---------- Step 3: 切分 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=0)

# ---------- Step 4: 建模 ----------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ---------- Step 5: 评估 ----------
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"R-squared: {r2:.6f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# ---------- 可视化 ----------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color="tab:blue")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", label="y = x")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs. Predicted Profit")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
print("图已保存: actual_vs_predicted.png")
