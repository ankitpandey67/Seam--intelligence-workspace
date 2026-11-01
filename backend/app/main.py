"""Seam API. Run: uvicorn app.main:app --reload  (from the backend/ folder)."""
import hashlib
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import llm, ml, rag, sample_data, storage

app = FastAPI(title="Seam")
S = {"raw": {}, "clean": {}, "reports": {}, "master": None, "joins": [], "docs": [], "rag": rag.Index(),
     "prepared_signature": None, "prepared_columns": []}


class Target(BaseModel):
    target: str


class Question(BaseModel):
    question: str


def master():
    if S["master"] is None:
        raise HTTPException(400, "Load data and run 'Clean and combine' first.")
    return S["master"]


def summary():
    return [{"name": n, "rows": len(d), "columns": len(d.columns)} for n, d in S["raw"].items()]


def _dataset_signature():
    h = hashlib.sha256()
    for name, df in sorted(S["raw"].items()):
        h.update(name.encode("utf-8"))
        h.update(str(len(df)).encode("utf-8"))
        h.update(str(tuple(df.shape)).encode("utf-8"))
        head = df.head(10).to_json(orient="records", date_format="iso")
        h.update(head.encode("utf-8"))
    return h.hexdigest()


@app.get("/api/health")
def health():
    return {"status": "ok", "datasets": summary(), "docs": S["docs"], "prepared": S["master"] is not None}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    for f in files:
        raw = await f.read()
        storage.save(f.filename, raw)
        ext = f.filename.rsplit(".", 1)[-1].lower()
        try:
            if ext in ("txt", "md"):
                S["docs"].append(f.filename)
                S["rag"].add(raw.decode("utf-8", "ignore"), f.filename)
            else:
                S["raw"].update(ml.load_file(f.filename, raw))
        except Exception as e:
            raise HTTPException(400, f"Could not read {f.filename}: {e}")
    S["master"] = None
    S["prepared_signature"] = None
    S["prepared_columns"] = []
    return {"datasets": summary(), "docs": S["docs"]}


@app.post("/api/sample")
def sample():
    S["raw"].update(sample_data.make())
    S["master"] = None
    S["prepared_signature"] = None
    S["prepared_columns"] = []
    return {"datasets": summary(), "docs": S["docs"]}


@app.post("/api/reset")
def reset():
    S.update(raw={}, clean={}, reports={}, master=None, joins=[], docs=[], prepared_signature=None, prepared_columns=[])
    S["rag"] = rag.Index()
    return {"datasets": [], "docs": []}


@app.post("/api/prepare")
def prepare():
    if not S["raw"]:
        raise HTTPException(400, "Upload at least one dataset first.")
    signature = _dataset_signature()
    if S["master"] is not None and S["prepared_signature"] == signature:
        return {"reports": S["reports"], "joins": S["joins"], "rows": len(S["master"]),
                "columns": S["prepared_columns"]}

    S["rag"] = rag.Index() if not S["docs"] else S["rag"]
    for n, d in S["raw"].items():
        S["clean"][n], S["reports"][n] = ml.clean(d)
    S["master"], S["joins"] = ml.auto_merge(S["clean"])
    m = S["master"]
    for n, d in S["clean"].items():
        S["rag"].add(f"Dataset {n}: {len(d)} rows. Columns: " + "; ".join(
            f"{c} ({d[c].dtype}, {d[c].nunique()} unique)" for c in d.columns), f"dataset:{n}")
    columns = [{"name": c, "type": "number" if ml.is_num(m[c]) else "date" if ml.is_dt(m[c]) else "text"}
               for c in m.columns]
    S["prepared_signature"] = signature
    S["prepared_columns"] = columns
    return {"reports": S["reports"], "joins": S["joins"], "rows": len(m), "columns": columns}


@app.get("/api/eda")
def eda():
    return ml.eda(master())


def guard(fn, *a):
    try:
        return fn(*a)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/cluster")
def cluster():
    return guard(ml.cluster, master())


@app.post("/api/rules")
def rules():
    return guard(ml.rules, master())


@app.post("/api/predict")
def predict(t: Target):
    return guard(ml.automl, master(), t.target)


@app.post("/api/forecast")
def forecast():
    return guard(ml.forecast, master())


@app.post("/api/ask")
def ask(q: Question):
    return llm.ask(q.question, master(), S["rag"])


@app.post("/api/insights")
def insights():
    df, facts = master(), {}
    metric = ml.guess_metric(df)
    for key, fn in (("segments", lambda: [{k: s[k] for k in ("name", "size", "share", "label")} for s in ml.cluster(df)["segments"]]),
                    ("rules", lambda: ml.rules(df)["rules"][:3]),
                    ("forecast", lambda: ml.forecast(df)),
                    ("drivers", lambda: ml.automl(df, metric)["importances"][:5] if metric else None)):
        try:
            facts[key] = fn()
        except Exception:
            facts[key] = None
    if facts.get("forecast"):
        facts["forecast"] = {"metric": facts["forecast"]["metric"], "forecast": facts["forecast"]["forecast"]}
    plan = ml.plan_valid(ml.plan_heuristic("top selling products by revenue", df), df)
    res = ml.run_plan(df, plan) if plan and plan["by"] else None
    if res and metric:
        tbl, vc = res["table"], res["value_col"]
        total = float(df[metric].sum()) or 1.0
        facts["top"] = {"leader": str(tbl[0][plan["by"]]), "metric": metric, "share": tbl[0][vc] / total,
                        "top5": tbl[:5]}
    return llm.insights(facts)


FRONT = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", StaticFiles(directory=FRONT, html=True), name="ui")
