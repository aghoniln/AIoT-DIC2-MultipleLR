"""
CRISP-DM: 多元线性回归预测初创公司利润 (50_Startups.csv)
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://raw.githubusercontent.com/mahesh147/"
       "Multiple-Linear-Regression/master/50_Startups.csv")
LOCAL = os.path.join(BASE, "50_Startups.csv")

# ===== 1. 业务理解 =====
print("=" * 60)
print("1. 业务理解: 根据 R&D / 管理 / 市场支出与州别 预测 Profit")
print("   成功标准: 测试集 R² 高 (>0.9), MAE / RMSE 低")

# ===== 2. 数据理解 =====
print("=" * 60)
if os.path.exists(LOCAL):
    df = pd.read_csv(LOCAL)
else:
    df = pd.read_csv(URL)
    df.to_csv(LOCAL, index=False)
print("2. 数据理解")
print(df.head())
print("列名:", list(df.columns))
print("形状:", df.shape)
print(df.dtypes)
print(df.describe())
print("State 分布:\n", df["State"].value_counts())
print("缺失值:\n", df.isnull().sum())

num_cols = ["R&D Spend", "Administration", "Marketing Spend"]
for c in num_cols:  # IQR 异常值检测
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"   {c}: IQR 异常值 {n} 个")
print("与 Profit 的相关系数:\n", df[num_cols + ["Profit"]].corr()["Profit"])

# ===== 3. 数据准备 =====
print("=" * 60)
print("3. 数据准备")
df = df.dropna()
X = df[num_cols + ["State"]]
y = df["Profit"]
# drop='first' 避免虚拟变量陷阱 (多重共线性)
pre = ColumnTransformer(
    [("state", OneHotEncoder(drop="first"), ["State"])],
    remainder="passthrough")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
print(f"   训练集 {X_train.shape}, 测试集 {X_test.shape}")

# ===== 4. 建模 =====
print("=" * 60)
print("4. 建模: LinearRegression")
model = Pipeline([("pre", pre), ("reg", LinearRegression())])
model.fit(X_train, y_train)
names = model.named_steps["pre"].get_feature_names_out()
reg = model.named_steps["reg"]
print("   截距:", round(reg.intercept_, 2))
for n, c in zip(names, reg.coef_):
    print(f"   {n:35s} {c:12.4f}")
from sklearn.model_selection import KFold
cv = cross_val_score(model, X, y, scoring="r2",
                     cv=KFold(5, shuffle=True, random_state=0))
print(f"   5 折交叉验证 R²: {cv.mean():.4f} ± {cv.std():.4f}")

# ===== 5. 评估 =====
print("=" * 60)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
metrics = {
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mse,
    "RMSE": float(np.sqrt(mse)),
    "CV_R2_mean": float(cv.mean()),
}
print("5. 评估 (测试集)")
for k, v in metrics.items():
    print(f"   {k:10s} {v:,.4f}")
print(pd.DataFrame({"实际": y_test.values, "预测": y_pred.round(2)}))

# ===== 6. 部署 =====
print("=" * 60)
joblib.dump(model, os.path.join(BASE, "startup_profit_model.joblib"))
with open(os.path.join(BASE, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("6. 部署: 模型已保存 startup_profit_model.joblib, 指标已保存 metrics.json")
sample = pd.DataFrame([{"R&D Spend": 160000, "Administration": 130000,
                        "Marketing Spend": 300000, "State": "New York"}])
print("   示例预测:", round(model.predict(sample)[0], 2))
