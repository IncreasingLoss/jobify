"""
Jobify — FastAPI Routes
All endpoints, reads index.html from disk.
"""

import io
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    File,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from functions import (
    check_ollama,
    generate_google_query,
    lock,
    run_classify,
    run_scrape,
    state,
)

app = FastAPI(title="Jobify")

# Path to index.html — must be in the same folder as this file
HTML_PATH = Path(__file__).parent / "front_end.html"


# ─── Serve Frontend ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists():
        return HTMLResponse(
            "<h1>index.html not found</h1><p>Place it next to api.py</p>",
            status_code=500,
        )
    return HTML_PATH.read_text(encoding="utf-8")


# ─── Ollama ─────────────────────────────────────────────────────────────────

@app.get("/api/ollama-status")
def ollama_status():
    with lock:
        return {
            "connected": state["ollama_connected"],
            "models": state["ollama_models"],
            "error": state["ollama_error"],
        }


@app.post("/api/generate-query")
async def generate_query(req: Request):
    p = await req.json()
    kw = p.get("keywords", "").strip()
    loc = p.get("location", "").strip()
    if not kw:
        return JSONResponse({"error": "Keywords required"}, status_code=400)
    model = p.get(
        "model",
        state["ollama_models"][0] if state["ollama_models"] else "qwen3:8b",
    )
    q = generate_google_query(kw, loc, model)
    if q:
        return {"query": q}
    return JSONResponse({"error": "Failed to generate query"}, status_code=500)


# ─── Scraping ───────────────────────────────────────────────────────────────

@app.post("/api/scrape")
async def start_scrape(req: Request):
    with lock:
        if state["scrape_status"] == "running":
            return JSONResponse(
                {"error": "Scrape already running"}, status_code=409
            )
    p = await req.json()
    kw = p.get("keywords", "").strip()
    if not kw:
        return JSONResponse({"error": "Keywords required"}, status_code=400)
    threading.Thread(target=run_scrape, args=(p,), daemon=True).start()
    return {"ok": True}


@app.get("/api/scrape-status")
def scrape_status():
    with lock:
        return {
            "status": state["scrape_status"],
            "progress": state["scrape_progress"],
            "site_status": state["scrape_site_status"],
            "total_raw": state["scrape_total_raw"],
            "total_deduped": state["scrape_total_deduped"],
            "duplicates": state["scrape_duplicates"],
        }


@app.get("/api/scrape-data")
def scrape_data(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    with lock:
        df = state["scraped_df"]
    if df is None or df.empty:
        return {"data": [], "total": 0, "page": 1, "pages": 0}
    total = len(df)
    pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    slice_df = df.iloc[start : start + per_page]
    data = []
    for _, row in slice_df.iterrows():
        r = {}
        for c in row.index:
            v = row[c]
            if pd.isna(v):
                r[c] = None
            elif isinstance(v, (np.integer,)):
                r[c] = int(v)
            elif isinstance(v, (np.floating,)):
                r[c] = float(v)
            else:
                r[c] = str(v)
        data.append(r)
    return {"data": data, "total": total, "page": page, "pages": pages}


@app.post("/api/load-csv")
async def load_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if "title" not in df.columns:
            return JSONResponse(
                {"error": "CSV must have a 'title' column"}, status_code=400
            )
        raw = len(df)
        if "description" not in df.columns:
            df["description"] = np.nan
        df["description"] = df["description"].replace(
            r"^\s*$", np.nan, regex=True
        )
        df["_tk"] = df["title"].str.strip().str.lower()
        df["_ck"] = (
            df["company"].str.strip().str.lower()
            if "company" in df.columns
            else ""
        )
        df = df.drop_duplicates(subset=["_tk", "_ck"], keep="first")
        df = df.drop(columns=["_tk", "_ck"])
        cols = [
            c
            for c in [
                "id",
                "site",
                "job_url",
                "title",
                "company",
                "location",
                "date_posted",
                "emails",
                "description",
                "company_url",
                "job_type",
            ]
            if c in df.columns
        ]
        df = df.loc[:, cols]
        with lock:
            state["scraped_df"] = df
            state["scrape_total_raw"] = raw
            state["scrape_total_deduped"] = len(df)
            state["scrape_duplicates"] = raw - len(df)
            state["scrape_status"] = "done"
            state["scrape_progress"] = f"Loaded {len(df)} jobs from CSV"
        return {
            "ok": True,
            "total_raw": raw,
            "total_deduped": len(df),
            "duplicates": raw - len(df),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Classification ─────────────────────────────────────────────────────────

@app.post("/api/classify")
async def start_classify(req: Request):
    with lock:
        if state["classify_status"] == "running":
            return JSONResponse(
                {"error": "Classification already running"}, status_code=409
            )
        if state["scraped_df"] is None or state["scraped_df"].empty:
            return JSONResponse(
                {
                    "error": "No scraped data. Search or load CSV first.",
                },
                status_code=400,
            )
    p = await req.json()
    threading.Thread(target=run_classify, args=(p,), daemon=True).start()
    return {"ok": True}


@app.get("/api/classify-status")
def classify_status():
    with lock:
        return {
            "status": state["classify_status"],
            "progress": state["classify_progress"],
            "current": state["classify_current"],
            "total": state["classify_total"],
            "target_type": state["classify_target_type"],
            "stats": state["classify_stats"],
        }


@app.get("/api/classify-data")
def classify_data(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    min_match: int = Query(0, ge=0, le=100),
    only_matching: bool = Query(False),
    show_rejected: bool = Query(False),
):
    with lock:
        df = state["classified_df"]
    if df is None or df.empty:
        return {"data": [], "total": 0, "page": 1, "pages": 0}

    filtered = df.copy()

    # Filter: only show matches unless "show rejected" is checked
    if not show_rejected:
        filtered = filtered[filtered["is_right_jobtype"] == "yes"]

    # Filter: skills matching
    if "skills_matching" in filtered.columns:
        if only_matching:
            filtered = filtered[filtered["skills_matching"].notna()]
        filtered = filtered[filtered["skills_matching"].fillna(-1) >= min_match]

    total = len(filtered)
    pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    slice_df = filtered.iloc[start : start + per_page]
    data = []
    for _, row in slice_df.iterrows():
        r = {}
        for c in row.index:
            v = row[c]
            if pd.isna(v):
                r[c] = None
            elif isinstance(v, (np.integer,)):
                r[c] = int(v)
            elif isinstance(v, (np.floating,)):
                r[c] = float(v)
            else:
                r[c] = str(v)
        data.append(r)
    return {"data": data, "total": total, "page": page, "pages": pages}


# ─── Exports ────────────────────────────────────────────────────────────────

@app.get("/api/export/scraped")
def export_scraped():
    with lock:
        df = state["scraped_df"]
    if df is None or df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=jobs_scraped.csv"
        },
    )


@app.get("/api/export/final")
def export_final(
    min_match: int = Query(0, ge=0, le=100),
    only_matching: bool = Query(False),
    show_rejected: bool = Query(False),
):
    with lock:
        df = state["classified_df"]
    if df is None or df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)

    filtered = df.copy()

    if not show_rejected:
        filtered = filtered[filtered["is_right_jobtype"] == "yes"]

    if "skills_matching" in filtered.columns:
        if only_matching:
            filtered = filtered[filtered["skills_matching"].notna()]
        filtered = filtered[filtered["skills_matching"].fillna(-1) >= min_match]
    if "skills_matching" in filtered.columns:
        filtered = filtered.sort_values("skills_matching", ascending=False)

    buf = io.StringIO()
    filtered.to_csv(buf, index=False)
    return Response(
        buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=jobs_final.csv"
        },
    )


# ─── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Jobify UI on http://localhost:8000")
    threading.Thread(target=check_ollama, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")