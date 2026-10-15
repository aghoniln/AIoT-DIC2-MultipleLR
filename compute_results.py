"""
Precompute results for DIC-2 Multiple Linear Regression & Feature Selection Explorer:
Dual Dataset Engine:
  1. 50 Startups Profit Prediction (Primary Topic)
  2. Boston Housing Valuation (Applied Reusable Workflow / Skill)
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


def coef_stats(m, cols, X_tr, y_tr):
    """OLS standard errors, t-stats, p-values, 95% CI and VIF on training set."""
    Xm = np.column_stack([np.ones(len(X_tr)), X_tr[cols].values])
    resid = y_tr.values - m.predict(X_tr[cols])
    dof = len(y_tr) - Xm.shape[1]
    if dof <= 0:
        dof = 1
    sigma2 = (resid @ resid) / dof
    try:
        inv = np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(np.diag(sigma2 * inv))
    except Exception:
        se = np.ones(Xm.shape[1]) * 1e-4

    beta = np.r_[m.intercept_, m.coef_]
    t = beta / np.where(se == 0, 1e-9, se)
    p = 2 * stats.t.sf(np.abs(t), dof)
    tc = stats.t.ppf(0.975, dof)
    vifs = [None] + vif(X_tr[cols])
    return [{
        "feature": name,
        "coef": float(b),
        "stderr": float(s),
        "tstat": float(tt),
        "pvalue": float(pp),
        "ci_lower": float(b - tc * s),
        "ci_upper": float(b + tc * s),
        "vif": v,
    } for name, b, s, tt, pp, v in zip(["(Intercept)"] + cols, beta, se, t, p, vifs)]


def run_pipeline(name, title, raw_df, X, y, target_col, target_unit, random_state=0, max_k=5):
    feats = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler().fit(X_train)
    Xs_train = pd.DataFrame(scaler.transform(X_train), columns=feats)

    def rmse_of(cols):
        m = LinearRegression().fit(X_train[list(cols)], y_train)
        return float(np.sqrt(mean_squared_error(y_test, m.predict(X_test[list(cols)]))))

    def fit_eval(cols):
        cols = list(cols)
        m = LinearRegression().fit(X_train[cols], y_train)
        pred = m.predict(X_test[cols])
        res = y_test.values - pred
        order = np.argsort(-np.abs(res))
        outliers = [{
            "rank": r + 1,
            "index": int(y_test.index[i]) + 1 if isinstance(y_test.index[i], (int, np.integer)) else i + 1,
            "actual": float(y_test.values[i]),
            "predicted": float(pred[i]),
            "residual": float(res[i]),
            "abs_error": float(abs(res[i])),
            "pct_error": float(abs(res[i]) / abs(y_test.values[i]) * 100) if y_test.values[i] != 0 else 0.0,
        } for r, i in enumerate(order[:10])]

        n, k = len(y_test), len(cols)
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        nt = len(y_train)
        r2t = r2_score(y_train, m.predict(X_train[cols]))
        f_den = ((1 - r2t) / (nt - k - 1)) if (nt - k - 1) > 0 and (1 - r2t) != 0 else 1e-9
        F = (r2t / k) / f_den if k > 0 else 0.0

        return {
            "features": cols,
            "rmse": float(np.sqrt(mse)),
            "mse": float(mse),
            "r2": float(r2),
            "adj_r2": float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if (n - k - 1) > 0 else float(r2),
            "mae": float(mean_absolute_error(y_test, pred)),
            "f_stat": float(F),
            "f_p": float(stats.f.sf(F, k, nt - k - 1)) if (nt - k - 1) > 0 else 0.0,
            "train_r2": float(r2t),
            "intercept": float(m.intercept_),
            "coef": {c: float(v) for c, v in zip(cols, m.coef_)},
            "actual": [float(v) for v in y_test],
            "pred": [float(v) for v in pred],
            "coef_stats": coef_stats(m, cols, X_train, y_train),
            "outliers": outliers,
        }

    # 1. Lasso ranking
    alphas, coefs, _ = lasso_path(Xs_train.values, y_train.values, n_alphas=500)
    lasso_order = []
    for j in range(coefs.shape[1] - 1, -1, -1)[::-1]:
        for i in np.nonzero(coefs[:, j])[0]:
            if feats[i] not in lasso_order:
                lasso_order.append(feats[i])
    lasso_order += [f for f in feats if f not in lasso_order]

    # 2. ElasticNet ranking
    enet = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=100000).fit(Xs_train, y_train)
    enet_order = [feats[i] for i in np.argsort(-np.abs(enet.coef_))]

    # 3. Forward SFS
    forward_sets = []
    sel = []
    for _ in range(max_k):
        candidates = [f for f in feats if f not in sel]
        if not candidates:
            break
        best = min(candidates, key=lambda f: rmse_of(sel + [f]))
        sel.append(best)
        forward_sets.append(sel.copy())

    # 4. RFE
    rfe_sets = []
    for k in range(1, max_k + 1):
        selector = RFE(LinearRegression(), n_features_to_select=k).fit(Xs_train, y_train)
        rfe_sets.append([f for f, s in zip(feats, selector.support_) if s])

    # 5. Exhaustive Search
    exhaustive_sets = [
        list(min(combinations(feats, k), key=rmse_of))
        for k in range(1, max_k + 1)
    ]

    methods = {
        "lasso": [lasso_order[:k] for k in range(1, max_k + 1)],
        "forward": forward_sets,
        "rfe": rfe_sets,
        "elasticnet": [enet_order[:k] for k in range(1, max_k + 1)],
        "exhaustive": exhaustive_sets,
    }

    results = {m_name: [fit_eval(s) for s in sets] for m_name, sets in methods.items()}

    # EDA
    num_cols = list(raw_df.select_dtypes(include=[np.number]).columns)
    corr_dict = raw_df[num_cols].corr().round(4).to_dict()

    eda = {
        "rows": raw_df.head(60).to_dict(orient="records"),
        "total_rows": len(raw_df),
        "describe": raw_df[num_cols].describe().round(2).to_dict(),
        "corr": corr_dict.get(target_col, {}),
        "missing": int(raw_df.isnull().sum().sum()),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "target": target_col,
        "target_unit": target_unit,
    }

    return {
        "id": name,
        "title": title,
        "features": feats,
        "results": results,
        "eda": eda,
    }


# ===================== 1. 50 Startups =====================
csv_startup = os.path.join(BASE, "50_Startups.csv")
raw_startup = pd.read_csv(csv_startup)
data_startup = pd.get_dummies(raw_startup, columns=["State"], drop_first=True, dtype=float)
X_startup = data_startup.drop(columns=["Profit"])
y_startup = data_startup["Profit"]

startup_bundle = run_pipeline(
    name="startup",
    title="50 Startups: Venture Profit Prediction",
    raw_df=raw_startup,
    X=X_startup,
    y=y_startup,
    target_col="Profit",
    target_unit="$",
    random_state=0,
    max_k=5
)
# Add extra startup state metrics
startup_bundle["eda"]["state_counts"] = raw_startup["State"].value_counts().to_dict()
startup_bundle["eda"]["state_profit"] = raw_startup.groupby("State")["Profit"].mean().round(2).to_dict()


# ===================== 2. Boston Housing =====================
csv_boston = os.path.join(BASE, "BostonHousing.csv")
if not os.path.exists(csv_boston):
    boston_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    raw_boston = pd.read_csv(boston_url)
    raw_boston.to_csv(csv_boston, index=False)
else:
    raw_boston = pd.read_csv(csv_boston)

X_boston = raw_boston.drop(columns=["medv"])
y_boston = raw_boston["medv"]

boston_bundle = run_pipeline(
    name="boston",
    title="Boston Housing: Median Property Valuation",
    raw_df=raw_boston,
    X=X_boston,
    y=y_boston,
    target_col="medv",
    target_unit="$1k",
    random_state=42,
    max_k=5
)

# Feature dictionary for Boston
boston_bundle["feature_meta"] = {
    "rm": "Average number of rooms per dwelling",
    "lstat": "% Lower status of the population",
    "ptratio": "Pupil-teacher ratio by town",
    "dis": "Weighted distances to five Boston employment centres",
    "crim": "Per capita crime rate by town",
    "tax": "Full-value property-tax rate per $10,000",
    "nox": "Nitric oxides concentration (parts per 10 million)",
    "chas": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "b": "1000(Bk - 0.63)^2 where Bk is proportion of Black residents",
    "zn": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "indus": "Proportion of non-retail business acres per town",
    "age": "Proportion of owner-occupied units built prior to 1940",
    "rad": "Index of accessibility to radial highways",
}

# Combine into master output
master_data = {
    "meta": {
        "title": "DIC-2 Multiple Linear Regression & Feature Selection Explorer",
        "description": "CRISP-DM standard workflow with comparative feature selection on 50 Startups and Boston Housing",
        "problems": ["startup", "boston"]
    },
    "problems": {
        "startup": startup_bundle,
        "boston": boston_bundle
    },
    # Backwards compatibility alias for startup results
    "features": startup_bundle["features"],
    "results": startup_bundle["results"],
    "eda": startup_bundle["eda"]
}

# Write to web/data.js and root data.js
for p in [os.path.join(BASE, "web", "data.js"), os.path.join(BASE, "data.js")]:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("window.STARTUP_DATA = " + json.dumps(master_data) + ";\n")
    print(f"Wrote {p}")

print("Successfully precomputed dual-problem datasets!")
