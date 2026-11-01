"""Streamlit interface for the Seam data-mining workbench."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import llm, ml, rag, storage


st.set_page_config(page_title="Seam | Data intelligence", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    :root {--seam-teal:#0c8b8a; --seam-ink:#132130;}
    .stApp {background:radial-gradient(ellipse at 75% -12%,rgba(12,139,138,.12),transparent 42%);}
    .block-container {max-width:1480px;padding-top:2rem;padding-bottom:3.5rem;}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#091821 0%,#102631 100%);border-right:1px solid rgba(255,255,255,.08);}
    [data-testid="stSidebar"] * {color:#e8f1f5;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.45rem .7rem;border-radius:.7rem;transition:background .16s ease;}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(255,255,255,.08);}
    [data-testid="stMetric"] {background:linear-gradient(145deg,rgba(12,139,138,.10),rgba(128,128,128,.045));padding:1.05rem 1.1rem;border:1px solid rgba(12,139,138,.14);border-radius:16px;box-shadow:0 12px 30px rgba(13,36,47,.055);}
    [data-testid="stMetricLabel"] {font-size:.82rem;font-weight:650;}
    [data-testid="stMetricValue"] {letter-spacing:-.04em;}
    div[data-testid="stPlotlyChart"] {background:rgba(255,255,255,.62);border:1px solid rgba(128,128,128,.13);border-radius:16px;padding:.65rem;box-shadow:0 10px 28px rgba(13,36,47,.045);}
    [data-testid="stFileUploader"] section {background:linear-gradient(140deg,rgba(12,139,138,.08),rgba(255,255,255,.45));border:1px dashed rgba(12,139,138,.48);border-radius:18px;padding:1.1rem;}
    [data-testid="stFileUploader"] section:hover {border-color:#0c8b8a;background:rgba(12,139,138,.10);}
    .stButton > button {border-radius:11px;min-height:2.55rem;font-weight:650;transition:transform .16s ease,box-shadow .16s ease;}
    .stButton > button:hover {transform:translateY(-1px);box-shadow:0 8px 18px rgba(12,139,138,.16);border-color:#0c8b8a;}
    .stTextInput input {border-radius:12px;}
    h1 {letter-spacing:-.045em!important;}
    h2,h3 {letter-spacing:-.025em!important;}
    .seam-brand {display:flex;align-items:center;gap:12px;padding:.5rem 0 1.15rem;}
    .seam-logo {width:44px;height:44px;border-radius:15px;background:linear-gradient(145deg,#38d3c4,#087d80);display:grid;place-items:center;box-shadow:0 9px 24px rgba(22,198,181,.24);flex:none;}
    .seam-logo svg {width:29px;height:29px;}
    .seam-brand-name {font-size:1.48rem;font-weight:800;letter-spacing:-.055em;line-height:1.05;color:#fff;}
    .seam-brand-tag {font-size:.62rem;font-weight:700;letter-spacing:.16em;color:#9eb8c2;margin-top:5px;}
    .seam-answer {padding:1.1rem 1.2rem;border:1px solid rgba(12,139,138,.19);border-left:4px solid #0c8b8a;border-radius:15px;background:linear-gradient(135deg,rgba(12,139,138,.09),rgba(255,255,255,.42));line-height:1.75;white-space:pre-wrap;}
    .seam-kicker {color:#0c8b8a;font-size:.73rem;font-weight:750;letter-spacing:.13em;text-transform:uppercase;}
    .seam-visual {padding:.6rem;border:1px solid rgba(12,139,138,.12);border-radius:16px;background:rgba(255,255,255,.35);margin-bottom:1rem;}
    @media (prefers-color-scheme:dark) {
      div[data-testid="stPlotlyChart"] {background:rgba(18,30,40,.75);}
      .seam-answer {background:linear-gradient(135deg,rgba(12,139,138,.16),rgba(18,30,40,.5));}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    provider_default = os.getenv(
        "SEAM_LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama"
    )
    if provider_default == "openai":
        provider_default = "openai-compatible"
    if provider_default not in {"ollama", "anthropic", "openai-compatible"}:
        provider_default = "ollama"
    defaults = {
        "raw": {},
        "clean": {},
        "reports": {},
        "master": None,
        "joins": [],
        "documents": {},
        "rag_index": rag.Index(),
        "processed_uploads": set(),
        "upload_widget_version": 0,
        "dashboard_visuals": [],
        "dashboard_filters": [],
        "dashboard_pending_filters": None,
        "dashboard_pending_kpis": None,
        "llm_provider": provider_default,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def language_model_config() -> dict:
    provider = st.session_state.get("llm_provider", "ollama")
    default_models = {
        "ollama": "qwen2.5:7b",
        "anthropic": os.getenv("SEAM_MODEL", "claude-sonnet-4-5-20250929"),
        "openai-compatible": os.getenv("SEAM_MODEL", "gpt-4o-mini"),
    }
    if provider == "openai":
        provider = "openai-compatible"
    if provider not in default_models:
        provider = "ollama"
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    return {
        "provider": provider,
        "model": st.session_state.get(f"llm_model_{provider}", default_models[provider]),
        "api_key": st.session_state.get(f"llm_key_{provider}", os.getenv(key_env, "")),
        "base_url": st.session_state.get(
            f"llm_url_{provider}",
            "http://localhost:11434" if provider == "ollama" else "https://api.openai.com/v1",
        ),
    }


def reset_prepared() -> None:
    st.session_state.clean = {}
    st.session_state.reports = {}
    st.session_state.master = None
    st.session_state.joins = []
    for key in ("segment_result", "pattern_result", "model_result", "forecast_result", "answer_result"):
        st.session_state.pop(key, None)


def prepare_data() -> None:
    if not st.session_state.raw:
        st.error("Upload at least one dataset before preparing the workspace.")
        return
    try:
        cleaned, reports = {}, {}
        for name, frame in st.session_state.raw.items():
            cleaned[name], reports[name] = ml.clean(frame)
        master, joins = ml.auto_merge(cleaned)
        index = rag.Index()
        for name, text in st.session_state.documents.items():
            index.add(text, name)
        for name, frame in cleaned.items():
            profile = "; ".join(
                f"{column} ({frame[column].dtype}, {frame[column].nunique()} unique)"
                for column in frame.columns
            )
            index.add(f"Dataset {name}: {len(frame)} rows. Columns: {profile}", f"dataset:{name}")
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not prepare datasets: {exc}")
        return

    st.session_state.clean = cleaned
    st.session_state.reports = reports
    st.session_state.master = master
    st.session_state.joins = joins
    st.session_state.rag_index = index


@st.cache_data(show_spinner=False)
def cached_eda(frame: pd.DataFrame) -> dict:
    return ml.eda(frame)


def show_figure(figure: dict | None) -> None:
    if figure:
        st.plotly_chart(go.Figure(figure), width="stretch")


def show_table(rows) -> None:
    if rows is None:
        return
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not frame.empty:
        st.dataframe(frame, width="stretch", hide_index=True)


def build_visual(frame: pd.DataFrame, spec: dict):
    kind, x, y = spec["kind"], spec.get("x"), spec.get("y")
    title = spec.get("title") or "Untitled visual"
    if kind == "Table":
        return frame.head(500)
    if kind == "Histogram":
        if not y or y not in frame.columns or not ml.is_num(frame[y]):
            raise ValueError("Histogram needs a numeric field.")
        return px.histogram(frame, x=y, nbins=30, title=title, color_discrete_sequence=["#0c8b8a"])
    if kind == "Scatter":
        if not x or not y or not ml.is_num(frame[x]) or not ml.is_num(frame[y]):
            raise ValueError("Scatter plot needs two numeric fields.")
        sample = frame.sample(min(5000, len(frame)), random_state=0) if len(frame) > 5000 else frame
        return px.scatter(sample, x=x, y=y, title=title, opacity=.72, color_discrete_sequence=["#0c8b8a"])
    if kind == "Box":
        if not y or not ml.is_num(frame[y]):
            raise ValueError("Box plot needs a numeric measure.")
        return px.box(frame, x=x if x and x != "(none)" else None, y=y, title=title,
                      color_discrete_sequence=["#0c8b8a"])
    if kind == "Pie":
        if not x or x not in frame.columns:
            raise ValueError("Pie chart needs a category field.")
        grouped = frame[x].fillna("Unknown").astype(str).value_counts().head(12).rename_axis(x).reset_index(name="rows")
        return px.pie(grouped, names=x, values="rows", title=title, hole=.48)
    if kind in ("Bar", "Line"):
        if not x or x not in frame.columns:
            raise ValueError(f"{kind} chart needs an x-axis field.")
        group_key = frame[x].dt.to_period("M").dt.to_timestamp() if ml.is_dt(frame[x]) else frame[x]
        if y and y != "(row count)" and y in frame.columns and ml.is_num(frame[y]):
            grouped = frame.groupby(group_key, dropna=False, observed=True)[y].agg(spec.get("aggregation", "sum")).reset_index()
            value_name = f"{spec.get('aggregation', 'sum')} {y}"
        else:
            grouped = frame.groupby(group_key, dropna=False, observed=True).size().reset_index(name="rows")
            value_name = "rows"
        grouped.columns = [x, value_name]
        if ml.is_dt(grouped[x]):
            grouped = grouped.sort_values(x)
        else:
            grouped = grouped.sort_values(value_name, ascending=False).head(30)
        if kind == "Line":
            return px.line(grouped, x=x, y=value_name, title=title, markers=True,
                           color_discrete_sequence=["#0c8b8a"])
        return px.bar(grouped, x=x, y=value_name, title=title,
                      color_discrete_sequence=["#0c8b8a"])
    raise ValueError(f"Unsupported visual type: {kind}")


def validate_dashboard_definition(definition: object, frame: pd.DataFrame) -> tuple[list[dict], list[str], list[str]]:
    if not isinstance(definition, dict) or definition.get("version") != 1:
        raise ValueError("Dashboard definition must be a version 1 JSON object.")

    visuals = definition.get("visuals")
    filters = definition.get("filters", [])
    kpis = definition.get("kpis", [])
    if not isinstance(visuals, list) or len(visuals) > 50:
        raise ValueError("Dashboard must contain a list of no more than 50 visuals.")
    if not isinstance(filters, list) or not all(isinstance(item, str) for item in filters):
        raise ValueError("Dashboard filters must be a list of field names.")
    if not isinstance(kpis, list) or not all(isinstance(item, str) for item in kpis):
        raise ValueError("Dashboard KPIs must be a list of field names.")

    columns = set(frame.columns)
    numeric = {column for column in frame.columns if ml.is_num(frame[column])}
    if len(set(filters)) != len(filters) or any(item not in columns for item in filters):
        raise ValueError("Dashboard filters contain duplicate or unknown fields.")
    if len(kpis) > 6 or len(set(kpis)) != len(kpis) or any(item not in numeric for item in kpis):
        raise ValueError("Dashboard KPIs must contain up to six unique numeric fields.")

    kinds = {"Bar", "Line", "Scatter", "Histogram", "Pie", "Box", "Table"}
    validated_visuals = []
    for spec in visuals:
        if not isinstance(spec, dict) or spec.get("kind") not in kinds:
            raise ValueError("Dashboard contains an invalid visual.")
        if not isinstance(spec.get("title"), str) or not spec["title"].strip():
            raise ValueError("Each dashboard visual needs a title.")

        kind, x, y = spec["kind"], spec.get("x"), spec.get("y")
        if kind in {"Bar", "Line", "Pie"} and x not in columns:
            raise ValueError(f"{kind} visual has an unknown category field.")
        if kind in {"Bar", "Line"}:
            if y != "(row count)" and y not in numeric:
                raise ValueError(f"{kind} visual needs a numeric measure or row count.")
            if spec.get("aggregation") not in {"sum", "mean", "count"}:
                raise ValueError(f"{kind} visual has an unsupported aggregation.")
        elif kind == "Scatter" and (x not in numeric or y not in numeric):
            raise ValueError("Scatter visual needs two numeric fields.")
        elif kind == "Histogram" and y not in numeric:
            raise ValueError("Histogram visual needs a numeric field.")
        elif kind == "Box" and (
            y not in numeric or (x not in columns and x != "(none)")
        ):
            raise ValueError("Box visual needs a numeric measure and an optional category field.")
        validated_visuals.append(spec)

    return validated_visuals, filters, kpis


def filtered_dashboard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    for column in st.session_state.dashboard_filters:
        if column not in result.columns:
            continue
        if ml.is_dt(result[column]):
            valid = result[column].dropna()
            if valid.empty:
                continue
            start, end = valid.min().date(), valid.max().date()
            chosen = st.date_input(f"{column.replace('_', ' ').title()} range", (start, end), key=f"dashboard_date_{column}")
            if isinstance(chosen, tuple) and len(chosen) == 2:
                result = result[result[column].dt.date.between(chosen[0], chosen[1])]
        else:
            values = sorted(result[column].fillna("Unknown").astype(str).unique().tolist())
            if len(values) <= 100:
                chosen_values = st.multiselect(
                    column.replace("_", " ").title(), values, default=values,
                    key=f"dashboard_values_{column}",
                )
                result = result[result[column].fillna("Unknown").astype(str).isin(chosen_values)]
            else:
                st.caption(f"Filter {column} by exact text")
                search = st.text_input(f"{column.replace('_', ' ').title()} contains", key=f"dashboard_text_{column}")
                if search:
                    result = result[result[column].astype(str).str.contains(search, case=False, na=False)]
    return result


def load_uploads(files) -> None:
    for uploaded in files:
        raw = uploaded.getvalue()
        fingerprint = hashlib.sha256(uploaded.name.encode("utf-8") + raw).hexdigest()
        if fingerprint in st.session_state.processed_uploads:
            continue
        try:
            storage.save(uploaded.name, raw)
            suffix = Path(uploaded.name).suffix.lower()
            if suffix in (".txt", ".md"):
                text = raw.decode("utf-8", "ignore")
                st.session_state.documents[uploaded.name] = text
                st.session_state.rag_index.add(text, uploaded.name)
            else:
                tables = ml.load_file(uploaded.name, raw)
                if not tables:
                    raise ValueError("No tables were found in this file.")
                st.session_state.raw.update(tables)
            st.session_state.processed_uploads.add(fingerprint)
            reset_prepared()
            st.success(f"Imported {uploaded.name}")
        except Exception as exc:
            st.error(f"Could not import {uploaded.name}: {exc}")


def page_data() -> None:
    st.title("Data")
    st.caption("Bring your datasets and supporting documents into the Seam workspace.")
    files = st.file_uploader(
        "Drop files here or browse",
        type=["csv", "xlsx", "xls", "json", "sql", "db", "sqlite", "txt", "md"],
        accept_multiple_files=True,
        help="Supports CSV, Excel, JSON, SQL dump, SQLite, TXT, and Markdown files.",
        key=f"seam_upload_{st.session_state.upload_widget_version}",
    )
    col_upload, col_prepare, col_reset = st.columns([1.2, 1.4, 1])
    with col_upload:
        if st.button("Import selected files", type="primary", width="stretch", disabled=not files):
            load_uploads(files)
    with col_prepare:
        if st.button("Clean and combine", width="stretch", disabled=not st.session_state.raw):
            with st.spinner("Cleaning datasets and finding shared IDs…"):
                prepare_data()
    with col_reset:
        if st.button("Clear workspace", width="stretch"):
            st.session_state.raw = {}
            st.session_state.clean = {}
            st.session_state.reports = {}
            st.session_state.master = None
            st.session_state.joins = []
            st.session_state.documents = {}
            st.session_state.processed_uploads = set()
            st.session_state.upload_widget_version += 1
            st.session_state.rag_index = rag.Index()
            for key in ("segment_result", "pattern_result", "model_result", "forecast_result", "answer_result"):
                st.session_state.pop(key, None)
            st.rerun()

    datasets = [
        {"Dataset": name, "Rows": len(frame), "Columns": len(frame.columns)}
        for name, frame in st.session_state.raw.items()
    ]
    if datasets:
        st.subheader("Loaded datasets")
        show_table(datasets)
    elif not st.session_state.documents:
        st.info("No dataset uploaded yet. Upload a file above to begin; no sample or other dataset is loaded automatically.")
    if st.session_state.documents:
        st.subheader("Reference documents")
        st.write(", ".join(st.session_state.documents))
    if st.session_state.master is not None:
        st.subheader("Preparation report")
        st.success(
            f"Prepared {len(st.session_state.master):,} rows and "
            f"{len(st.session_state.master.columns)} columns."
        )
        for join in st.session_state.joins:
            st.write(f"- {join}")
        for name, report in st.session_state.reports.items():
            with st.expander(f"{name}: {report['rows_before']:,} → {report['rows_after']:,} rows"):
                if report["actions"]:
                    for action in report["actions"]:
                        st.write(f"- {action}")
                else:
                    st.write("No cleaning changes were required.")


def page_overview() -> None:
    st.title("Seam intelligence workspace")
    st.caption("A home for exploring, modeling, and understanding your business data.")
    master = st.session_state.master
    if master is None:
        if st.session_state.raw:
            st.info("Your uploaded datasets are waiting to be cleaned and combined.")
            show_table([
                {"Dataset": name, "Rows": len(frame), "Columns": len(frame.columns)}
                for name, frame in st.session_state.raw.items()
            ])
        else:
            st.info("No dataset uploaded yet. Seam will only display and analyze data that you add to this workspace.")
        if st.button("Open data workspace", type="primary"):
            st.session_state.page = "Data"
            st.rerun()
        return

    columns = [
        {"name": name, "type": "number" if ml.is_num(master[name]) else "date" if ml.is_dt(master[name]) else "text"}
        for name in master.columns
    ]
    eda = cached_eda(master)
    metric = ml.guess_metric(master)
    kpis = st.columns(4)
    kpis[0].metric("Rows analyzed", f"{len(master):,}")
    kpis[1].metric("Fields", len(master.columns))
    kpis[2].metric("Numeric measures", sum(ml.is_num(master[c]) for c in master.columns))
    kpis[3].metric("Missing values", f"{int(master.isna().sum().sum()):,}")
    st.subheader("Data signals")
    if eda["figures"]:
        charts = st.columns(2)
        for i, figure in enumerate(eda["figures"][:2]):
            with charts[i % 2]:
                show_figure(figure)
    else:
        st.info("No numeric or date fields were found for charts.")
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Analysis tools")
        st.write("Explore distributions, discover segments, mine buying patterns, train predictions, or ask questions.")
        if st.button("Open analytics", type="primary"):
            st.session_state.page = "Analytics"
            st.rerun()
    with right:
        st.subheader("Dataset profile")
        profile = pd.DataFrame(columns).head(8)
        show_table(profile)
        if metric:
            st.caption(f"Suggested measure: **{metric}**")


def page_dashboard() -> None:
    st.title("Dashboard studio")
    st.caption("Build a shareable, interactive dashboard from the dataset you prepared.")
    master = st.session_state.master
    numeric = [column for column in master.columns if ml.is_num(master[column])]
    pending_filters = st.session_state.pop("dashboard_pending_filters", None)
    if pending_filters is not None:
        st.session_state["dashboard_filter_fields"] = [
            name for name in pending_filters if name in master.columns
        ]
    pending_kpis = st.session_state.pop("dashboard_pending_kpis", None)
    if pending_kpis is not None:
        st.session_state["dashboard_kpis"] = [
            name for name in pending_kpis if name in numeric
        ]
    with st.expander("Dashboard filters", expanded=bool(st.session_state.dashboard_filters)):
        st.session_state.dashboard_filters = st.multiselect(
            "Choose fields viewers can filter",
            master.columns.tolist(),
            default=[name for name in st.session_state.dashboard_filters if name in master.columns],
            key="dashboard_filter_fields",
        )
    filtered = filtered_dashboard_frame(master)
    if filtered.empty:
        st.warning("These filters match no rows. Adjust the selections to see dashboard results.")
        return

    st.subheader("KPI cards")
    kpi_fields = st.multiselect(
        "Pick numeric measures to summarize",
        numeric,
        default=[name for name in (ml.guess_metric(master),) if name in numeric],
        max_selections=6,
        key="dashboard_kpis",
    )
    if kpi_fields:
        cards = st.columns(min(len(kpi_fields) + 1, 4))
        cards[0].metric("Filtered rows", f"{len(filtered):,}")
        for index, field in enumerate(kpi_fields[:3], start=1):
            cards[index].metric(f"Total {field.replace('_', ' ').title()}", f"{filtered[field].sum():,.2f}")
        for field in kpi_fields[3:]:
            st.metric(f"Total {field.replace('_', ' ').title()}", f"{filtered[field].sum():,.2f}")

    st.divider()
    st.subheader("Add a visual")
    with st.container(border=True):
        left, middle, right = st.columns([1, 1, 1])
        with left:
            kind = st.selectbox("Visual type", ["Bar", "Line", "Scatter", "Histogram", "Pie", "Box", "Table"])
            title = st.text_input("Visual title", placeholder="For example: Revenue by region", key="dashboard_new_title")
        with middle:
            x_options = master.columns.tolist()
            x = st.selectbox("Category / X-axis", x_options, index=0, key="dashboard_new_x")
            aggregation = st.selectbox("Aggregation", ["sum", "mean", "count"], key="dashboard_new_aggregation")
        with right:
            y_options = ["(row count)"] + numeric
            y = st.selectbox("Measure / Y-axis", y_options, index=1 if numeric else 0, key="dashboard_new_y")
            if kind == "Scatter":
                y = st.selectbox("Scatter Y-axis", numeric or ["(no numeric fields)"], key="dashboard_scatter_y")
                x = st.selectbox("Scatter X-axis", numeric or ["(no numeric fields)"], key="dashboard_scatter_x")
            elif kind in ("Histogram", "Box"):
                y = st.selectbox("Numeric field", numeric or ["(no numeric fields)"], key="dashboard_distribution_y")
        if st.button("Add visual to dashboard", type="primary", key="dashboard_add_visual"):
            spec = {
                "kind": kind, "title": title.strip() or f"{kind} visualization",
                "x": x, "y": y, "aggregation": aggregation,
            }
            if kind in ("Scatter", "Histogram", "Box") and not numeric:
                st.error(f"{kind} needs one or more numeric fields.")
            else:
                st.session_state.dashboard_visuals.append(spec)
                st.rerun()

    visual_specs = st.session_state.dashboard_visuals
    st.divider()
    st.subheader(f"Dashboard visuals · {len(visual_specs)}")
    if not visual_specs:
        st.info("Add a chart above to start building your dashboard.")
    else:
        for start in range(0, len(visual_specs), 2):
            visual_columns = st.columns(2)
            for offset, spec in enumerate(visual_specs[start:start + 2]):
                index = start + offset
                with visual_columns[offset], st.container(border=True):
                    st.markdown(f"**{spec['title']}**")
                    try:
                        result = build_visual(filtered, spec)
                        if isinstance(result, pd.DataFrame):
                            show_table(result)
                        else:
                            show_figure(result.to_dict())
                    except (ValueError, KeyError) as exc:
                        st.warning(f"Could not render this visual: {exc}")
                    if st.button("Remove visual", key=f"dashboard_remove_{index}"):
                        st.session_state.dashboard_visuals.pop(index)
                        st.rerun()

    dashboard_json = json.dumps(
        {"version": 1, "visuals": visual_specs, "filters": st.session_state.dashboard_filters,
         "kpis": kpi_fields},
        indent=2,
    )
    st.download_button(
        "Download dashboard definition",
        dashboard_json,
        file_name="seam-dashboard.json",
        mime="application/json",
    )
    dashboard_file = st.file_uploader("Load a saved dashboard definition", type=["json"], key="dashboard_import")
    if dashboard_file is not None:
        try:
            definition = json.loads(dashboard_file.getvalue())
            valid_visuals, filters, kpis = validate_dashboard_definition(definition, master)
            st.session_state.dashboard_visuals = valid_visuals
            st.session_state.dashboard_pending_filters = filters
            st.session_state.dashboard_pending_kpis = kpis
            st.success(f"Loaded {len(valid_visuals)} dashboard visual(s).")
            st.rerun()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            st.error(f"Could not load this dashboard definition: {exc}")


def page_analytics() -> None:
    st.title("Analytics")
    with st.spinner("Computing dataset profile and charts…"):
        result = cached_eda(st.session_state.master)
    charts = st.columns(2)
    for index, figure in enumerate(result["figures"]):
        with charts[index % 2]:
            show_figure(figure)
    st.subheader("Summary statistics")
    show_table(result["describe"])
    st.subheader("Field profile")
    show_table(result["columns"])


def page_segments() -> None:
    st.title("Customer segments")
    st.caption("K-means selects a cluster count using silhouette score.")
    if st.button("Find segments", type="primary"):
        try:
            with st.spinner("Finding customer groups…"):
                st.session_state.segment_result = ml.cluster(st.session_state.master)
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("segment_result")
    if result:
        st.metric("Silhouette score", result["silhouette"])
        show_figure(result["figure"])
        rows = [
            {"Segment": segment["name"], "Customers": segment["size"], "Share": f'{segment["share"]:.0%}',
             "Stands out": segment["label"], **segment["profile"]}
            for segment in result["segments"]
        ]
        show_table(rows)


def page_patterns() -> None:
    st.title("Buying patterns")
    st.caption("Lift above 1 indicates items co-occur more often than chance.")
    if st.button("Mine patterns", type="primary"):
        try:
            with st.spinner("Mining association rules…"):
                st.session_state.pattern_result = ml.rules(st.session_state.master)
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("pattern_result")
    if result:
        if result.get("source"):
            st.caption(f"Mined from {result['source']}")
        if result["rules"]:
            show_table(result["rules"])
        else:
            st.info(result.get("message", "No strong patterns found."))


def page_prediction() -> None:
    st.title("Prediction studio")
    master = st.session_state.master
    targets = [column for column in master.columns if not ml.is_dt(master[column]) and not ml.is_id(column)]
    if not targets:
        st.warning("There are no eligible target columns in this dataset.")
        return
    default_metric = ml.guess_metric(master)
    target = st.selectbox("Choose a target", targets, index=targets.index(default_metric) if default_metric in targets else 0)
    train_col, forecast_col = st.columns(2)
    with train_col:
        if st.button("Train and compare models", type="primary", width="stretch"):
            try:
                with st.spinner("Training candidate models…"):
                    st.session_state.model_result = ml.automl(master, target)
            except ValueError as exc:
                st.error(str(exc))
    with forecast_col:
        if st.button("Forecast next 3 months", width="stretch"):
            try:
                with st.spinner("Building the forecast…"):
                    st.session_state.forecast_result = ml.forecast(master)
            except ValueError as exc:
                st.error(str(exc))
    result = st.session_state.get("model_result")
    if result:
        if result["warning"]:
            st.warning(result["warning"])
        st.subheader(f"Best model: {result['best']}")
        metrics = st.columns(len(result["holdout"]) + 1)
        metrics[0].metric("Task", result["task"])
        for col, (name, value) in zip(metrics[1:], result["holdout"].items()):
            col.metric(name, value)
        show_figure(result["figure"])
        show_table(result["leaderboard"])
        st.subheader("Sample predictions")
        show_table(result["predictions"])
    forecast_result = st.session_state.get("forecast_result")
    if forecast_result:
        st.subheader(f"Forecast: {forecast_result['metric']}")
        show_figure(forecast_result["figure"])
        show_table(forecast_result["forecast"])


def page_ask() -> None:
    st.title("Ask Seam")
    st.caption("Ask in your own words: explore the data, compare values, explain a trend, or ask about uploaded reference documents.")
    st.markdown('<div class="seam-kicker">Try questions beyond charts and totals</div>', unsafe_allow_html=True)
    st.caption("What does this dataset tell me? · Why might revenue have changed? · Explain these columns · What does the attached report say?")
    with st.form("ask_form"):
        question = st.text_input("Your question", placeholder="Ask Seam anything about your uploaded data or documents…")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        try:
            with st.spinner("Interpreting your question…"):
                st.session_state.answer_result = llm.ask(
                    question.strip(), st.session_state.master, st.session_state.rag_index,
                    language_model_config(),
                )
        except (llm.LLMProviderError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not answer this question: {exc}")
    answer = st.session_state.get("answer_result")
    if answer:
        safe_answer = html.escape(answer["answer"])
        st.markdown(f'<div class="seam-answer">{safe_answer}</div>', unsafe_allow_html=True)
        if answer.get("plan"):
            st.caption(f"Interpreted plan: {json.dumps(answer['plan'], ensure_ascii=False)}")
        elif not answer.get("llm"):
            st.caption("The answer uses only verifiable uploaded-data summaries. Configure and test a provider in the sidebar for broader conversational answers.")
        if answer.get("sources"):
            st.caption("Context sources: " + ", ".join(sorted(set(answer["sources"]))))
        show_figure(answer.get("figure"))
        show_figure(answer.get("drivers_figure"))
        show_table(answer.get("table"))
    st.divider()
    if st.button("Generate business insights"):
        facts = {}
        master = st.session_state.master
        metric = ml.guess_metric(master)
        with st.spinner("Computing recommendations…"):
            try:
                facts["segments"] = [
                    {key: segment[key] for key in ("name", "size", "share", "label")}
                    for segment in ml.cluster(master)["segments"]
                ]
            except ValueError:
                facts["segments"] = None
            try:
                facts["rules"] = ml.rules(master)["rules"][:3]
            except ValueError:
                facts["rules"] = None
            try:
                forecast = ml.forecast(master)
                facts["forecast"] = {"metric": forecast["metric"], "forecast": forecast["forecast"]}
            except ValueError:
                facts["forecast"] = None
            if metric:
                try:
                    facts["drivers"] = ml.automl(master, metric)["importances"][:5]
                except ValueError:
                    facts["drivers"] = None
            plan = ml.plan_valid(ml.plan_heuristic("top selling products by revenue", master), master)
            top_result = ml.run_plan(master, plan) if plan and plan["by"] else None
            if top_result and metric:
                table, value_col = top_result["table"], top_result["value_col"]
                total = float(master[metric].sum()) or 1.0
                facts["top"] = {
                    "leader": str(table[0][plan["by"]]), "metric": metric,
                    "share": table[0][value_col] / total, "top5": table[:5],
                }
        try:
            insights = llm.insights(facts, language_model_config())
        except llm.LLMProviderError as exc:
            st.error(f"Could not generate insights: {exc}")
            return
        for bullet in insights["bullets"]:
            st.write(f"- {bullet}")
        st.caption("Generated with the language model." if insights["llm"] else "Generated from computed dataset facts.")


def main() -> None:
    initialize_state()
    with st.sidebar:
        st.markdown(
            """
            <div class="seam-brand">
              <div class="seam-logo" aria-hidden="true">
                <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M24.8 7.4H13.1a5.1 5.1 0 0 0 0 10.2h5.8a5.1 5.1 0 0 1 0 10.2H7.2" stroke="white" stroke-width="3.2" stroke-linecap="round"/>
                  <circle cx="7.1" cy="7.3" r="2.2" fill="#D9FFF8"/>
                  <circle cx="24.8" cy="25.2" r="2.2" fill="#D9FFF8"/>
                </svg>
              </div>
              <div><div class="seam-brand-name">Seam</div><div class="seam-brand-tag">DATA INTELLIGENCE</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        if st.session_state.master is None:
            st.caption(f"Workspace · {len(st.session_state.raw)} uploaded dataset(s), not prepared")
        else:
            st.caption(f"Workspace · {len(st.session_state.master):,} prepared rows")
        pages = ["Overview", "Data", "Dashboard studio", "Analytics", "Segments", "Patterns", "Prediction", "Ask Seam"]
        if "page" not in st.session_state:
            st.session_state.page = "Overview"
        selected = st.radio("Workspace", pages, key="page", label_visibility="collapsed")
        with st.expander("Language model setup"):
            provider_names = {
                "ollama": "Ollama · local",
                "anthropic": "Anthropic API",
                "openai-compatible": "OpenAI-compatible API",
            }
            provider = st.selectbox(
                "Provider", list(provider_names), format_func=provider_names.get, key="llm_provider"
            )
            model_defaults = {
                "ollama": "qwen2.5:7b",
                "anthropic": os.getenv("SEAM_MODEL", "claude-sonnet-4-5-20250929"),
                "openai-compatible": os.getenv("SEAM_MODEL", "gpt-4o-mini"),
            }
            st.text_input("Model name", value=model_defaults[provider], key=f"llm_model_{provider}")
            if provider != "ollama":
                key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
                st.text_input(
                    "API key", value=os.getenv(key_env, ""),
                    type="password", key=f"llm_key_{provider}",
                )
            default_url = "http://localhost:11434" if provider == "ollama" else "https://api.openai.com/v1"
            url = st.text_input("API base URL", value=default_url, key=f"llm_url_{provider}")
            st.caption(
                "Ollama keeps model inference on this computer; install Ollama and pull the selected model first."
                if provider == "ollama" else "Your prompts and dataset context are sent to the selected provider."
            )
            if st.button("Test language model", key="llm_test"):
                try:
                    reply = llm.complete(
                        "Reply with OK only.",
                        "Connection check.",
                        max_tokens=8,
                        config={
                            "provider": provider,
                            "model": st.session_state[f"llm_model_{provider}"],
                            "api_key": st.session_state.get(f"llm_key_{provider}", ""),
                            "base_url": url,
                        },
                    )
                    if reply:
                        st.success(f"Connected: {reply.strip()[:40]}")
                    else:
                        st.error("No response. Check that the selected provider and API key are configured.")
                except llm.LLMProviderError as exc:
                    st.error(str(exc))
        st.markdown("---")
        st.caption("Private session workspace")

    if selected == "Data":
        page_data()
        return
    if selected == "Dashboard studio":
        if st.session_state.master is None:
            st.title("Dashboard studio")
            st.info("Upload and prepare a dataset to create a dashboard.")
            if st.button("Open data workspace", type="primary"):
                st.session_state.page = "Data"
                st.rerun()
            return
        page_dashboard()
        return
    if selected == "Overview":
        page_overview()
        return
    if st.session_state.master is None:
        st.title(selected)
        if st.session_state.raw:
            st.info("Only your uploaded datasets are shown. Clean and combine them to enable this analysis.")
            show_table([
                {"Dataset": name, "Rows": len(frame), "Columns": len(frame.columns)}
                for name, frame in st.session_state.raw.items()
            ])
        else:
            st.info("No dataset uploaded yet. Upload your own data from the Data page to begin.")
        if st.button("Open data workspace", type="primary"):
            st.session_state.page = "Data"
            st.rerun()
        return
    if selected == "Analytics":
        page_analytics()
    elif selected == "Segments":
        page_segments()
    elif selected == "Patterns":
        page_patterns()
    elif selected == "Prediction":
        page_prediction()
    elif selected == "Ask Seam":
        page_ask()


if __name__ == "__main__":
    main()
