"""LLM orchestration with Anthropic, OpenAI-compatible, and local Ollama providers."""
import json
import os
import re
from urllib.parse import urljoin

import requests
from . import ml

MODEL = os.getenv("SEAM_MODEL", "claude-sonnet-5-5")

PLAN_SYS = ('Convert a business question about a table into a JSON query plan. Reply with JSON only: '
            '{"by": column or null, "metric": numeric column or null, "agg": "sum"|"mean"|"count", "n": integer, '
            '"ascending": boolean, "drivers": boolean}. Set "drivers" true when the question asks what factors '
            'influenced the metric. Use only the column names provided.')
ANS_SYS = ("You are a careful business analyst. Answer the question in under 140 words using ONLY the numbers in the "
           "provided results and context. Name specific products, segments, or factors. If the results do not "
           "answer the question, say what is missing. Do not invent figures.")
GENERAL_ANS_SYS = (
    "You are Seam, a helpful data analyst. Answer the user's question directly, even when it is not asking for "
    "a sum, ranking, or grouping. Use the supplied dataset profile, small row sample, and retrieved document "
    "context as evidence. Do not claim that a sample represents the whole dataset. Never invent dataset facts, "
    "values, columns, or sources. If the supplied evidence is insufficient, explain what is missing and suggest "
    "a useful follow-up. If the user asks general knowledge unrelated to the uploaded data, answer briefly from "
    "general knowledge and clearly say it is not derived from their dataset. Keep the answer concise."
)
INS_SYS = ("You are a business analyst. From the facts, write 4-6 insights as lines starting with '- '. Each line "
           "states a finding with its number, then a concrete recommendation. Use only the provided facts.")


class LLMProviderError(RuntimeError):
    """Raised when a configured language-model provider cannot answer."""


def _config_value(config, name: str, default: str = "") -> str:
    if config is not None:
        value = config.get(name)
        if value is not None:
            return str(value).strip()
    return os.getenv(name.upper(), default).strip()


def complete(system: str, user: str, max_tokens: int = 600, config=None):
    default_provider = os.getenv("SEAM_LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
    provider = _config_value(config, "provider", default_provider).lower()
    default_model = "qwen2.5:7b" if provider == "ollama" else MODEL
    model = _config_value(config, "model", default_model)
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = _config_value(config, "api_key") or os.getenv(key_env, "").strip()
    base_url = _config_value(config, "base_url", os.getenv("SEAM_LLM_BASE_URL", ""))
    if provider in ("anthropic", "openai", "openai-compatible") and not api_key:
        return None
    try:
        if provider == "anthropic":
            import anthropic

            try:
                response = anthropic.Anthropic(api_key=api_key).messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except anthropic.APIError as exc:
                raise LLMProviderError(f"Anthropic model request failed: {exc}") from exc
            return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        if provider in ("openai", "openai-compatible"):
            endpoint = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
            response = requests.post(
                urljoin(endpoint + "/", "chat/completions"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": max_tokens,
                },
                timeout=(10, 120),
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        if provider == "ollama":
            endpoint = (base_url or "http://localhost:11434").rstrip("/")
            response = requests.post(
                urljoin(endpoint + "/", "api/chat"),
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=(5, 180),
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    except (requests.RequestException, ValueError, KeyError, ImportError) as exc:
        raise LLMProviderError(f"{provider} model request failed: {exc}") from exc
    raise LLMProviderError(f"Unsupported language-model provider: {provider}")


def llm_enabled(config=None):
    default_provider = os.getenv("SEAM_LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")
    provider = _config_value(config, "provider", default_provider).lower()
    if provider == "ollama":
        return True
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    return bool(_config_value(config, "api_key") or os.getenv(key_env))


def schema_text(df):
    lines = []
    for c in df.columns:
        kind = "date" if ml.is_dt(df[c]) else "number" if ml.is_num(df[c]) else "text"
        ex = ", ".join(map(str, df[c].dropna().unique()[:4]))
        lines.append(f"- {c} ({kind}) e.g. {ex}")
    return "\n".join(lines)


def _fmt(v):
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)


def _general_facts(df, rag, q: str):
    profile = []
    for column in df.columns:
        series = df[column]
        item = {
            "column": column,
            "type": "date" if ml.is_dt(series) else "number" if ml.is_num(series) else "text",
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique()),
        }
        if ml.is_dt(series):
            valid_dates = series.dropna()
            if not valid_dates.empty:
                item["date_range"] = [valid_dates.min().isoformat(), valid_dates.max().isoformat()]
        elif ml.is_num(series):
            item["statistics"] = {
                key: round(float(value), 3)
                for key, value in series.describe().items()
                if key in ("mean", "min", "max", "50%") and value == value
            }
        else:
            item["common_values"] = series.dropna().astype(str).value_counts().head(5).to_dict()
        profile.append(item)
    sample = df.head(5).astype(object).where(df.head(5).notna(), None).to_dict(orient="records")
    context = rag.query(q, 4)
    return {
        "question": q,
        "dataset": {"rows": len(df), "columns": profile, "sample_rows": sample},
        "retrieved_context": [{"source": item["source"], "text": item["text"][:1000]} for item in context],
    }, context


def ask(q: str, df, rag, config=None):
    plan = None
    raw = complete(PLAN_SYS, f"Columns:\n{schema_text(df)}\n\nQuestion: {q}", 200, config)
    if raw:
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            plan = ml.plan_valid(json.loads(m.group(0)), df) if m else None
        except json.JSONDecodeError:
            plan = None
    planner = "llm" if plan else "rules"
    plan = plan or ml.plan_valid(ml.plan_heuristic(q, df), df)
    if not plan:
        facts, ctx = _general_facts(df, rag, q)
        answer = complete(GENERAL_ANS_SYS, json.dumps(facts, default=str), 600, config)
        used_llm = answer is not None
        if not answer:
            dataset = facts["dataset"]
            numeric = [
                f"{item['column']}: mean {item['statistics']['mean']:,.2f}, range "
                f"{item['statistics']['min']:,.2f}–{item['statistics']['max']:,.2f}"
                for item in dataset["columns"] if item.get("statistics")
            ][:3]
            categories = [
                f"{item['column']}: {', '.join(item['common_values'])}"
                for item in dataset["columns"] if item.get("common_values")
            ][:3]
            answer = (
                f"I don't have the language model configured, so I can't reliably interpret every open-ended "
                f"question. Facts I can verify: your uploaded dataset contains {len(df):,} rows and "
                f"{len(df.columns)} columns."
            )
            if numeric:
                answer += " Numeric summaries: " + "; ".join(numeric) + "."
            if categories:
                answer += " Common text values: " + "; ".join(categories) + "."
            if ctx:
                excerpts = " ".join(
                    f"{item['source']}: {item['text'][:250].strip()}" for item in ctx[:2]
                )
                answer += " Relevant uploaded reference: " + excerpts
            else:
                answer += " Configure a language model in the sidebar for open-ended, contextual explanations."
        return {"answer": answer, "plan": None, "planner": "general", "table": None,
                "figure": None, "drivers_figure": None, "sources": [c["source"] for c in ctx],
                "llm": used_llm}
    res = ml.run_plan(df, plan)
    drivers = None
    if plan["drivers"] and plan["metric"]:
        try:
            drivers = ml.automl(df, plan["metric"])
        except ValueError:
            drivers = None
    ctx = rag.query(q, 3)
    facts = {"question": q, "plan": plan, "table": res["table"] if res else None,
             "drivers": drivers["importances"][:6] if drivers else None,
             "model_quality": drivers["holdout"] if drivers else None,
             "caveat": drivers["warning"] if drivers else None,
             "context": [c["text"][:500] for c in ctx]}
    answer = complete(ANS_SYS, json.dumps(facts, default=str), 400, config)
    if not answer:
        parts = []
        if res:
            vc, by = res["value_col"], plan["by"]
            rows = "; ".join(f"{r[by]} ({_fmt(r[vc])})" for r in res["table"][:5])
            parts.append(f"{vc} by {by}: {rows}.")
        if drivers:
            top = ", ".join(f"{d['feature']} ({d['importance']:.0%})" for d in drivers["importances"][:4])
            parts.append(f"Biggest influences on {plan['metric']}: {top}. "
                         f"The {drivers['best']} model explains this with {drivers['metric_name']} "
                         f"{list(drivers['holdout'].values())[0]} on held-out data. {drivers['warning']}")
        answer = " ".join(parts)
    return {"answer": answer, "plan": plan, "planner": planner, "table": res["table"] if res else None,
            "figure": res["figure"] if res else None,
            "drivers_figure": drivers["figure"] if drivers else None,
            "sources": [c["source"] for c in ctx], "llm": llm_enabled(config)}


def insights(facts: dict, config=None):
    text = complete(INS_SYS, json.dumps(facts, default=str), 700, config)
    if text:
        bullets = [re.sub(r"^[-*\d.\s]+", "", l).strip() for l in text.splitlines() if l.strip()[:1] in "-*0123456789"]
        if bullets:
            return {"bullets": bullets, "llm": True}
    b = []
    t = facts.get("top")
    if t:
        b.append(f"{t['leader']} leads {t['metric']} with {t['share']:.0%} of the total. Protect its stock and promotion.")
    s = facts.get("segments")
    if s:
        big = max(s, key=lambda x: x["size"])
        b.append(f"The largest segment ({big['share']:.0%} of customers) shows {big['label']}. Tailor offers to it first.")
    for r in (facts.get("rules") or [])[:2]:
        b.append(f"Buyers of {r['if_bought']} also buy {r['then_also']} (lift {r['lift']}). Bundle or cross-sell them.")
    f = facts.get("forecast")
    if f:
        b.append(f"Projected {f['metric']} next month: {f['forecast'][0]['value']:,.0f}. Plan inventory against it.")
    d = facts.get("drivers")
    if d:
        b.append(f"{d[0]['feature']} is the strongest driver of {facts['top']['metric'] if t else 'the target'} "
                 f"({d[0]['importance']:.0%} of model importance).")
    return {"bullets": b, "llm": False}
