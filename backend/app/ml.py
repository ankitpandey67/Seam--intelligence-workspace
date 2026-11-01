"""Loading, cleaning, EDA, data mining (clusters, rules), AutoML, forecasting, and safe query plans."""
from __future__ import annotations
import io, json, os, re, sqlite3, tempfile, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

COLORS = ["#0E6B6B", "#B7791F", "#3D5A80", "#7A9E7E", "#A44A3F", "#8D99AE"]


# ---------- helpers ----------
def is_num(s): return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
def is_dt(s): return pd.api.types.is_datetime64_any_dtype(s)
def is_id(c): return c == "id" or c.endswith("_id")
def num_cols(df): return [c for c in df.columns if is_num(df[c]) and not is_id(c)]
def cat_cols(df): return [c for c in df.columns if not is_num(df[c]) and not is_dt(df[c])]


def fig_json(fig, height=320):
    fig.update_layout(template="simple_white", colorway=COLORS, height=height,
                      margin=dict(l=50, r=20, t=50, b=40), title_font_size=14)
    return json.loads(fig.to_json())


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def guess_metric(df):
    nums = num_cols(df)
    for key in ("revenue", "sales", "amount", "total", "profit", "value", "price", "cost"):
        for c in nums:
            if key in c:
                return c
    return nums[0] if nums else None


# ---------- ingestion ----------
def _sqlite_tables(con):
    names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    return {n: pd.read_sql(f'select * from "{n}"', con) for n in names}


def load_file(name: str, raw: bytes) -> dict[str, pd.DataFrame]:
    base, _, ext = name.rpartition(".")
    ext = ext.lower()
    if ext == "csv":
        return {base: pd.read_csv(io.BytesIO(raw))}
    if ext in ("xlsx", "xls"):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        return {(base if len(sheets) == 1 else f"{base}_{s}"): d for s, d in sheets.items()}
    if ext == "json":
        data = json.loads(raw)
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
            data = lists[0] if lists else [data]
        return {base: pd.json_normalize(data)}
    if ext == "sql":  # SQL dump (CREATE TABLE / INSERT) executed in a throwaway in-memory SQLite
        con = sqlite3.connect(":memory:")
        con.executescript(raw.decode("utf-8", "ignore"))
        return _sqlite_tables(con)
    if ext in ("db", "sqlite"):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            f.write(raw)
        con = sqlite3.connect(f.name)
        try:
            return _sqlite_tables(con)
        finally:
            con.close(); os.unlink(f.name)
    raise ValueError(f"Unsupported file type: .{ext}")


# ---------- cleaning ----------
def clean(df: pd.DataFrame):
    rep = {"rows_before": len(df), "actions": []}
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    n = len(df)
    df = df.drop_duplicates()
    if n - len(df):
        rep["actions"].append(f"Removed {n - len(df)} duplicate rows")
    for c in list(df.columns):
        if is_num(df[c]) or is_dt(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        s = df[c].astype("string").str.strip()
        s = s.mask(s.str.lower().isin(["", "nan", "none", "null", "n/a", "na"]))
        num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
        if s.notna().any() and num.notna().sum() >= 0.9 * s.notna().sum():
            df[c] = num
            rep["actions"].append(f"Converted '{c}' to numeric")
            continue
        if any(k in c for k in ("date", "time", "_at")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(s, errors="coerce")
            if d.notna().mean() >= 0.8:
                df[c] = d
                rep["actions"].append(f"Parsed '{c}' as dates")
                continue
        df[c] = s.astype(object).where(s.notna(), np.nan)
    for c in df.columns:
        miss = int(df[c].isna().sum())
        if not miss:
            continue
        if is_num(df[c]):
            df[c] = df[c].fillna(df[c].median())
            rep["actions"].append(f"Filled {miss} missing in '{c}' with median")
        elif not is_dt(df[c]):
            df[c] = df[c].fillna("Unknown")
            rep["actions"].append(f"Filled {miss} missing in '{c}' with 'Unknown'")
    for c in num_cols(df):
        if df[c].nunique() > 20:
            lo, hi = df[c].quantile([0.01, 0.99])
            clipped = int(((df[c] < lo) | (df[c] > hi)).sum())
            df[c] = df[c].clip(lo, hi)
            if clipped:
                rep["actions"].append(f"Capped {clipped} outliers in '{c}' (1st-99th percentile)")
    rep["rows_after"] = len(df)
    return df.reset_index(drop=True), rep


def auto_merge(dfs: dict[str, pd.DataFrame]):
    """Join tables on shared *_id columns, starting from the largest (fact) table."""
    names = sorted(dfs, key=lambda n: -len(dfs[n]))
    master, joins, rest = dfs[names[0]], [], names[1:]
    changed = True
    while rest and changed:
        changed = False
        for n in list(rest):
            keys = [c for c in dfs[n].columns if is_id(c) and c in master.columns]
            if keys:
                right = dfs[n].drop_duplicates(subset=keys)
                master = master.merge(right, on=keys, how="left", suffixes=("", f"_{n}"))
                joins.append(f"{names[0]} + {n} on {', '.join(keys)}")
                rest.remove(n); changed = True
    for n in rest:
        joins.append(f"'{n}' shares no ID column with the others, so it was not joined")
    return master, joins


# ---------- EDA ----------
def eda(df: pd.DataFrame):
    nums = num_cols(df)
    cols = [{"name": c, "type": "number" if is_num(df[c]) else "date" if is_dt(df[c]) else "text",
             "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique())} for c in df.columns]
    figs, metric = [], guess_metric(df)
    dts = [c for c in df.columns if is_dt(df[c])]
    if dts and metric:
        m = df.groupby(df[dts[0]].dt.to_period("M").dt.to_timestamp())[metric].sum().reset_index()
        figs.append(fig_json(px.line(m, x=dts[0], y=metric, title=f"Monthly {metric}", markers=True)))
    for c in nums[:4]:
        figs.append(fig_json(px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")))
    if len(nums) >= 2:
        corr = df[nums].corr().round(2)
        figs.append(fig_json(px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                       title="Correlation"), height=380))
    for c in [c for c in cat_cols(df) if not is_id(c) and 2 <= df[c].nunique() <= 25][:3]:
        vc = df[c].value_counts().head(12).reset_index()
        vc.columns = [c, "count"]
        figs.append(fig_json(px.bar(vc, x=c, y="count", title=f"{c}: most common values")))
    desc = df[nums].describe().T.round(2).reset_index().rename(columns={"index": "column"}) if nums else pd.DataFrame()
    return {"rows": len(df), "columns": cols, "describe": records(desc), "figures": figs,
            "missing_total": int(df.isna().sum().sum())}


# ---------- clustering ----------
def cluster(df: pd.DataFrame):
    ent = "customer_id" if "customer_id" in df.columns else None
    if ent:
        nums, metric = num_cols(df), guess_metric(df)
        X = df.groupby(ent)[nums].mean()
        X["records"] = df.groupby(ent).size()
        if metric:
            X[f"{metric}_total"] = df.groupby(ent)[metric].sum()
    else:
        X = df[num_cols(df)].copy()
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 2 or len(X) < 10:
        raise ValueError("Need at least 10 rows and 2 numeric columns to find segments.")
    Z = StandardScaler().fit_transform(X)
    best = None
    for k in range(2, min(7, len(X) // 3 + 1)):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sc = silhouette_score(Z, km.labels_, sample_size=min(2000, len(X)), random_state=42)
        if best is None or sc > best[0]:
            best = (sc, k, km)
    sc, k, km = best
    labels = km.labels_
    segs = []
    for i in range(k):
        mask = labels == i
        z = km.cluster_centers_[i]
        top = np.argsort(-np.abs(z))[:2]
        desc = ", ".join(f"{'high' if z[j] > 0 else 'low'} {X.columns[j]}" for j in top)
        segs.append({"name": f"Segment {i + 1}", "size": int(mask.sum()), "share": round(float(mask.mean()), 3),
                     "label": desc, "profile": {c: round(float(X.loc[mask, c].mean()), 2) for c in X.columns}})
    P = PCA(2, random_state=0).fit_transform(Z)
    plot = pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "segment": [f"Segment {l + 1}" for l in labels]})
    plot = plot.sample(min(1500, len(plot)), random_state=0)
    fig = fig_json(px.scatter(plot, x="PC1", y="PC2", color="segment", opacity=0.7,
                              title=f"{k} segments (silhouette {sc:.2f})"), 380)
    return {"k": k, "silhouette": round(float(sc), 3), "entity": ent or "row", "segments": segs, "figure": fig}


# ---------- association rules ----------
def rules(df: pd.DataFrame):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    basket = next((c for c in df.columns if is_id(c) and any(k in c for k in ("transaction", "order", "invoice", "basket"))), None)
    item = next((c for c in ("product_name", "product", "item", "category") if c in df.columns), None)
    if basket and item and df[basket].nunique() < len(df):
        tx = df.groupby(basket)[item].apply(lambda s: sorted(set(map(str, s)))).tolist()
        source = f"'{item}' bought together within each '{basket}'"
    else:
        cats = [c for c in cat_cols(df) if not is_id(c) and 2 <= df[c].nunique() <= 20]
        if len(cats) < 2:
            return {"rules": [], "message": "No basket or categorical columns to mine.", "source": ""}
        tx = df[cats].astype(str).apply(lambda r: [f"{c}={v}" for c, v in r.items()], axis=1).tolist()
        source = "co-occurring values of " + ", ".join(cats)
    tx = [t for t in tx if len(t) > 0][:20000]
    onehot = pd.DataFrame(TransactionEncoder().fit(tx).transform(tx), columns=sorted({i for t in tx for i in t}))
    out = pd.DataFrame()
    for sup in (0.05, 0.02, 0.01, 0.005):
        fi = apriori(onehot, min_support=sup, use_colnames=True, max_len=3)
        if len(fi) < 2:
            continue
        try:
            r = association_rules(fi, metric="lift", min_threshold=1.0)
        except TypeError:
            r = association_rules(fi, num_itemsets=len(onehot), metric="lift", min_threshold=1.0)
        if len(r) >= 3:
            out = r; break
        out = r
    if out.empty:
        return {"rules": [], "message": "No strong patterns found.", "source": source}
    out = out[out["confidence"] >= 0.2].sort_values(["lift", "confidence"], ascending=False)
    out = out[~out.apply(lambda r: frozenset(r.antecedents | r.consequents), axis=1).duplicated()].head(12)
    rows = [{"if_bought": ", ".join(sorted(a)), "then_also": ", ".join(sorted(c)), "support": round(s, 3),
             "confidence": round(cf, 3), "lift": round(l, 2)}
            for a, c, s, cf, l in zip(out.antecedents, out.consequents, out.support, out.confidence, out.lift)]
    return {"rules": rows, "message": "", "source": source, "n_transactions": len(tx)}


# ---------- AutoML ----------
def automl(df: pd.DataFrame, target: str, max_rows: int = 15000):
    if target not in df.columns:
        raise ValueError(f"Unknown column '{target}'.")
    d = df.dropna(subset=[target])
    if is_dt(d[target]):
        raise ValueError("Pick a numeric or categorical target, not a date.")
    if len(d) < 30:
        raise ValueError("Need at least 30 rows with a value in the target column.")
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    y, X = d[target], d.drop(columns=[target])
    drop = [c for c in X if is_id(c) or X[c].nunique() <= 1 or (not is_num(X[c]) and not is_dt(X[c]) and X[c].nunique() > 30)]
    X = X.drop(columns=drop)
    for c in [c for c in X if is_dt(X[c])]:
        X[c + "_year"], X[c + "_month"], X[c + "_dow"] = X[c].dt.year, X[c].dt.month, X[c].dt.dayofweek
        X = X.drop(columns=c)
    cats = [c for c in X if not is_num(X[c])]
    numeric = [c for c in X if c not in cats]
    if numeric:
        X[numeric] = X[numeric].fillna(X[numeric].median())
    X = pd.get_dummies(X.astype({c: str for c in cats}), columns=cats, dtype=float)
    origin = [c if c in numeric else next((k for k in sorted(cats, key=len, reverse=True) if c.startswith(k + "_")), c)
              for c in X.columns]
    X.columns = [re.sub(r"[\[\]<>]", "_", c) for c in X.columns]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left after removing IDs.")

    regression = is_num(y) and y.nunique() > 15
    if regression:
        yv, scoring, metric_name = y.values.astype(float), "r2", "R²"
        models = {"Ridge regression": make_pipeline(StandardScaler(), Ridge()),
                  "Random forest": RandomForestRegressor(150, n_jobs=-1, random_state=0),
                  "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=-1)}
    else:
        le = LabelEncoder()
        yv = le.fit_transform(y.astype(str))
        if len(le.classes_) < 2:
            raise ValueError("Target has only one value.")
        scoring, metric_name = "f1_weighted", "F1 (weighted)"
        models = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
                  "Random forest": RandomForestClassifier(150, n_jobs=-1, random_state=0),
                  "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=-1)}
    board = []
    for name, m in models.items():
        try:
            board.append({"model": name, "score": round(float(cross_val_score(m, X, yv, cv=3, scoring=scoring).mean()), 4)})
        except Exception:
            continue
    if not board:
        raise ValueError("No model could be trained on this data.")
    board.sort(key=lambda r: -r["score"])
    best = board[0]["model"]
    strat = yv if not regression and np.bincount(yv).min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.2, random_state=0, stratify=strat)
    model = models[best].fit(Xtr, ytr)
    pred = model.predict(Xte)
    if regression:
        hold = {"R²": round(float(r2_score(yte, pred)), 3), "MAE": round(float(mean_absolute_error(yte, pred)), 2)}
        sample = pd.DataFrame({"actual": yte[:8].round(2), "predicted": pred[:8].round(2)})
    else:
        hold = {"accuracy": round(float(accuracy_score(yte, pred)), 3),
                "F1": round(float(f1_score(yte, pred, average="weighted")), 3)}
        sample = pd.DataFrame({"actual": le.inverse_transform(yte[:8]), "predicted": le.inverse_transform(pred[:8])})
    last = model[-1] if hasattr(model, "steps") else model
    imp = last.feature_importances_ if hasattr(last, "feature_importances_") else np.abs(last.coef_).reshape(-1, X.shape[1]).mean(0)
    s = pd.Series(imp, index=origin).groupby(level=0).sum()
    s = (s / s.sum()).sort_values(ascending=False).head(10)
    imps = [{"feature": k, "importance": round(float(v), 4)} for k, v in s.items()]
    fig = fig_json(px.bar(pd.DataFrame(imps)[::-1], x="importance", y="feature", orientation="h",
                          title=f"What drives {target} ({best})"), 340)
    warn = ""
    if regression and hold["R²"] > 0.98:
        warn = (f"Near-perfect fit: '{target}' is probably calculated from other columns "
                f"(for example revenue = price x quantity), so the top drivers describe its formula, not hidden causes.")
    return {"warning": warn, "task": "regression" if regression else "classification", "target": target, "best": best,
            "metric_name": metric_name, "leaderboard": board, "holdout": hold, "importances": imps,
            "figure": fig, "predictions": records(sample), "n_rows": len(d), "n_features": len(set(origin))}


# ---------- forecasting ----------
def forecast(df: pd.DataFrame, periods: int = 3):
    dts = [c for c in df.columns if is_dt(df[c])]
    metric = guess_metric(df)
    if not dts or not metric:
        raise ValueError("Forecasting needs a date column and a numeric sales-like column.")
    m = df.groupby(df[dts[0]].dt.to_period("M").dt.to_timestamp())[metric].sum().sort_index()
    if len(m) < 6:
        raise ValueError("Need at least 6 months of history to forecast.")
    seasonal = len(m) >= 24

    def feats(idx, t):
        cols = [t]
        if seasonal:
            cols.append(np.eye(12)[idx.month - 1])
        return np.column_stack(cols)

    t = np.arange(len(m))
    lr = LinearRegression().fit(feats(m.index, t), m.values)
    sd = float(np.std(m.values - lr.predict(feats(m.index, t))))
    fidx = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    fv = lr.predict(feats(fidx, np.arange(len(m), len(m) + periods)))
    fig = go.Figure()
    fig.add_scatter(x=m.index, y=m.values, name="Actual", mode="lines+markers")
    fig.add_scatter(x=fidx, y=fv, name="Forecast", mode="lines+markers", line=dict(dash="dash"))
    fig.add_scatter(x=list(fidx) + list(fidx[::-1]), y=list(fv + 1.28 * sd) + list((fv - 1.28 * sd)[::-1]),
                    fill="toself", opacity=0.2, line=dict(width=0), name="80% range", hoverinfo="skip")
    fig.update_layout(title=f"{metric}: next {periods} months")
    return {"metric": metric, "seasonal": seasonal, "figure": fig_json(fig, 360),
            "forecast": [{"month": d.strftime("%Y-%m"), "value": round(float(v), 2)} for d, v in zip(fidx, fv)]}


# ---------- natural-language query plans (executed safely, never exec'd) ----------
SYN = {"product": ["product_name", "product", "item"], "customer": ["customer_id", "customer"],
       "region": ["region", "city", "state", "country"], "city": ["city"], "category": ["category", "segment"],
       "channel": ["channel"], "tier": ["tier"]}


def plan_heuristic(q: str, df: pd.DataFrame):
    ql, cols, nums = q.lower(), list(df.columns), num_cols(df)
    mentioned = [c for c in cols if c.replace("_", " ") in ql]
    metric = next((c for c in mentioned if c in nums), None)
    if not metric and re.search(r"sales|revenue|sold|selling|income|amount|spend|profit", ql):
        metric = guess_metric(df)
    by = next((c for c in mentioned if c != metric and not is_dt(df[c])), None)
    if not by and re.search(r"month|trend|over time|by date|per year", ql):
        by = next((c for c in cols if is_dt(df[c])), None)
    if not by:
        for word, keys in SYN.items():
            if word in ql:
                by = next((c for k in keys for c in cols if k in c and c != metric), None)
                if by:
                    break
    agg = "mean" if re.search(r"average|avg|mean", ql) else "count" if (re.search(r"how many|number of|count", ql) or not metric) else "sum"
    n = re.search(r"(?:top|best|bottom|worst)\s*(\d+)", ql)
    return {"by": by, "metric": metric, "agg": agg,
            "n": int(n.group(1)) if n else (5 if re.search(r"top|best", ql) else 10),
            "ascending": bool(re.search(r"worst|lowest|least|bottom|fewest", ql)),
            "drivers": bool(re.search(r"factor|influenc|why|driver|affect|impact|caus|explain", ql))}


def plan_valid(plan, df):
    if not isinstance(plan, dict):
        return None
    by = plan.get("by") if plan.get("by") in df.columns else None
    metric = plan.get("metric") if plan.get("metric") in num_cols(df) else None
    agg = plan.get("agg") if plan.get("agg") in ("sum", "mean", "count") else ("sum" if metric else "count")
    try:
        n = max(1, min(50, int(plan.get("n", 10))))
    except (TypeError, ValueError):
        n = 10
    if not by and not (plan.get("drivers") and metric):
        return None
    return {"by": by, "metric": metric, "agg": agg, "n": n,
            "ascending": bool(plan.get("ascending")), "drivers": bool(plan.get("drivers"))}


def run_plan(df: pd.DataFrame, p: dict):
    by, metric, agg = p.get("by"), p.get("metric"), p.get("agg", "sum")
    if not by:
        return None
    key = df[by].dt.to_period("M").dt.to_timestamp() if is_dt(df[by]) else df[by]
    g = df.groupby(key)
    t = g.size().rename("count") if agg == "count" or not metric else g[metric].agg(agg).rename(f"{agg}_{metric}")
    val = t.name
    if is_dt(df[by]):
        t = t.sort_index().reset_index()
        t.columns = [by, val]
        fig = px.line(t, x=by, y=val, markers=True, title=f"{val} over time")
    else:
        t = t.sort_values(ascending=p.get("ascending", False)).head(p.get("n", 10)).reset_index()
        t.columns = [by, val]
        t = t.round(2)
        fig = px.bar(t, x=by, y=val, title=f"{'Bottom' if p.get('ascending') else 'Top'} {len(t)} {by} by {val}")
        fig.update_xaxes(type="category")
    return {"table": records(t), "figure": fig_json(fig), "value_col": val}
