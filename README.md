# Seam — data intelligence workspace

Seam is a Streamlit application for importing datasets, cleaning and joining tables, exploring data, finding segments and buying patterns, training prediction models, forecasting, and asking questions in plain language.

## Run locally

From the project root:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), choose **Data**, upload your files, and click **Clean and combine** to enable the analytics pages. Seam does not load sample or other datasets automatically.

### Configure a language model

The sidebar's **Language model setup** supports three provider types:

- **Ollama · local** — install [Ollama](https://ollama.com), start it, and download a model, for example `ollama pull qwen2.5:7b`. The default endpoint is `http://localhost:11434`; model inference stays local.
- **Anthropic API** — provide an Anthropic API key and model name.
- **OpenAI-compatible API** — provide an API key, model name, and optionally a compatible API base URL.

Use **Test language model** in the sidebar to check the selected provider. A hosted provider sends prompts, a small dataset sample, and retrieved document context to that service. Configure credentials in the sidebar or through the environment variables in `.env.example`.

Ask Seam supports conversational questions, not just totals or rankings. It uses uploaded dataset summaries and relevant document excerpts as evidence; it does not promise facts that are not in the supplied context.

## Run with Docker

From the project root:

```powershell
docker build -t seam .
docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY=your-key seam
```

Then open [http://localhost:8501](http://localhost:8501).

## Features

| Feature | Implementation |
|---|---|
| CSV, Excel, JSON, SQL dump, and SQLite input | `app.ml.load_file` |
| Cleaning and multi-table joins | `app.ml.clean` and `app.ml.auto_merge` |
| EDA charts and summary statistics | `app.ml.eda` |
| Customer segments | K-means clustering with silhouette-based cluster selection |
| Buying patterns | MLxtend Apriori association rules |
| Prediction | Ridge/Logistic, Random Forest, and XGBoost model comparison |
| Forecasting | Monthly trend model, with seasonality when sufficient history exists |
| Natural-language analysis and open-ended Q&A grounded in dataset profiles and uploaded documents | `app.llm` and `app.rag` |
| Interactive dashboard builder with KPI cards, filters, charts, and portable dashboard definitions | `app/streamlit_app.py` |
| Streamlit product interface | `app/streamlit_app.py` |

The original FastAPI endpoints remain available through `app.main` for API use. The default local and Docker launch path is Streamlit.

## Notes

- Each Streamlit browser session keeps its datasets in session state; they are not shared across users.
- Uploaded files are saved locally, and can optionally be sent to S3 when `S3_BUCKET` is configured.
- For multi-user persistence, add authenticated storage and a database before deploying this as a shared service.
