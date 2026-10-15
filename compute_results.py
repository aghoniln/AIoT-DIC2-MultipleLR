"""
Precompute all results for the 50 Startups website.
5 feature-selection methods x k = 1..5  ->  web/data.js
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, ElasticNet, lasso_path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "50_Startups.csv")
URL = ("https://raw.githubusercontent.com/harimittapalli/"
       "Mulitple-Linear-Reggression/master/50_Startups.csv")

raw = pd.read_csv(CSV) if os.path.exists(CSV) else pd.read_csv(URL)
data = pd.get_dummies(raw, columns=["State"], drop_first=True, dtype=float)
X = data.drop(columns=["Profit"])
y = data["Profit"]
FEATS = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_train)
Xs_train = pd.DataFrame(scaler.transform(X_train), columns=FEATS)


def vif(Xdf):
    """Variance inflation factor for each column of Xdf."""
    out = []
    for c in Xdf.columns:
        others = [o for o in Xdf.columns if o != c]
        if not others:
            out.append(1.0)
            continue
        r2 = LinearRegression().fit(Xdf[others], Xdf[c]).score(Xdf[others], Xdf[c])
        out.append(float(1 / (1 - r2)) if r2 < 1 else float("inf"))
    return out


def coef_stats(m, cols):
    """OLS standard errors, t-stats, p-values, 95% CI and VIF (training set)."""
    Xm = np.column_stack([np.ones(len(X_train)), X_train[cols].values])
    resid = y_train.values - m.predict(X_train[cols])
    dof = len(y_train) - Xm.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xm.T @ Xm)))
    beta = np.r_[m.intercept_, m.coef_]
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    tc = stats.t.ppf(0.975, dof)
    vifs = [None] + vif(X_train[cols])
    return [{
        "feature": name, "coef": float(b), "stderr": float(s), "tstat": float(tt),
        "pvalue": float(pp), "ci_lower": float(b - tc * s), "ci_upper": float(b + tc * s),
        "vif": v,
    } for name, b, s, tt, pp, v in zip(["(Intercept)"] + cols, beta, se, t, p, vifs)]


def fit_eval(cols):
    cols = list(cols)
    m = LinearRegression().fit(X_train[cols], y_train)
    pred = m.predict(X_test[cols])
    res = y_test.values - pred
    order = np.argsort(-np.abs(res))
    outliers = [{
        "rank": r + 1, "index": int(y_test.index[i]) + 1,
        "actual": float(y_test.values[i]), "predicted": float(pred[i]),
        "residual": float(res[i]), "abs_error": float(abs(res[i])),
        "pct_error": float(abs(res[i]) / y_test.values[i] * 100),
    } for r, i in enumerate(order)]
    n, k = len(y_test), len(cols)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    # training-set F statistic (overall significance)
    nt = len(y_train)
    r2t = r2_score(y_train, m.predict(X_train[cols]))
    F = (r2t / k) / ((1 - r2t) / (nt - k - 1))
    return {
        "features": cols,
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "r2": float(r2),
        "adj_r2": float(1 - (1 - r2) * (n - 1) / (n - k - 1)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "f_stat": float(F),
        "f_p": float(stats.f.sf(F, k, nt - k - 1)),
        "train_r2": float(r2t),
        "intercept": float(m.intercept_),
        "coef": {c: float(v) for c, v in zip(cols, m.coef_)},
        "actual": [float(v) for v in y_test],
        "pred": [float(v) for v in pred],
        "coef_stats": coef_stats(m, cols),
        "outliers": outliers,
    }


def rmse_of(cols):
    m = LinearRegression().fit(X_train[list(cols)], y_train)
    return np.sqrt(mean_squared_error(y_test, m.predict(X_test[list(cols)])))


# ---- rankings ----
def lasso_rank():
    alphas, coefs, _ = lasso_path(Xs_train.values, y_train.values, n_alphas=500)
    order = []
    for j in range(coefs.shape[1] - 1, -1, -1)[::-1]:  # large -> small alpha
        for i in np.nonzero(coefs[:, j])[0]:
            if FEATS[i] not in order:
                order.append(FEATS[i])
    order += [f for f in FEATS if f not in order]
    return order


def enet_rank():
    m = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=100000).fit(Xs_train, y_train)
    return [FEATS[i] for i in np.argsort(-np.abs(m.coef_))]


def forward_sets():
    sel, out = [], []
    for _ in FEATS:
        best = min((f for f in FEATS if f not in sel),
                   key=lambda f: rmse_of(sel + [f]))
        sel.append(best)
        out.append(sel.copy())
    return out


def rfe_sets():
    return [[f for f, s in zip(FEATS, RFE(LinearRegression(), n_features_to_select=k)
                               .fit(Xs_train, y_train).support_) if s]
            for k in range(1, len(FEATS) + 1)]


def exhaustive_sets():
    return [list(min(combinations(FEATS, k), key=rmse_of))
            for k in range(1, len(FEATS) + 1)]


lr, er = lasso_rank(), enet_rank()
methods = {
    "lasso": [lr[:k] for k in range(1, 6)],
    "forward": forward_sets(),
    "rfe": rfe_sets(),
    "elasticnet": [er[:k] for k in range(1, 6)],
    "exhaustive": exhaustive_sets(),
}
results = {name: [fit_eval(s) for s in sets] for name, sets in methods.items()}

for name, rs in results.items():
    print(f"{name:11s}", " | ".join(f"k={i+1}:{r['rmse']:.2f}" for i, r in enumerate(rs)))

# ---- dataset / EDA ----
num = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
eda = {
    "rows": raw.to_dict(orient="records"),
    "describe": raw[num].describe().round(2).to_dict(),
    "corr": raw[num].corr().round(4).to_dict(),
    "state_counts": raw["State"].value_counts().to_dict(),
    "state_profit": raw.groupby("State")["Profit"].mean().round(2).to_dict(),
    "missing": int(raw.isnull().sum().sum()),
    "n_train": len(y_train), "n_test": len(y_test),
}

out = {"features": FEATS, "results": results, "eda": eda}
os.makedirs(os.path.join(BASE, "web"), exist_ok=True)
with open(os.path.join(BASE, "web", "data.js"), "w", encoding="utf-8") as f:
    f.write("window.STARTUP_DATA = " + json.dumps(out) + ";\n")
print("wrote web/data.js")
