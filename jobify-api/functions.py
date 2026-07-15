"""
Jobify — Core Functions
Scraping, Ollama communication, classification logic, shared state.
"""

import os, json, time, logging, threading
import urllib, urllib.request, urllib.error
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from jobspy import scrape_jobs
except ImportError:
    print("ERROR: jobspy not installed. Run: pip install jobspy")
    raise SystemExit(1)

# ─── Glassdoor Noise Filter ──────────────────────────────────────────────────
class _SilenceGlassdoorNoise(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("location not parsed" in msg or "response status code 400" in msg)

logging.getLogger("JobSpy:Glassdoor").addFilter(_SilenceGlassdoorNoise())

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Shared State ────────────────────────────────────────────────────────────
lock = threading.Lock()
state = {
    "ollama_connected": False,
    "ollama_models": [],
    "ollama_error": None,
    "scrape_status": "idle",
    "scrape_progress": "",
    "scrape_site_status": {},
    "scrape_total_raw": 0,
    "scrape_total_deduped": 0,
    "scrape_duplicates": 0,
    "scraped_df": None,
    "classify_status": "idle",
    "classify_progress": "",
    "classify_current": 0,
    "classify_total": 0,
    "classify_stats": {"matched": 0, "rejected": 0, "error": 0},
    "classify_target_type": None,
    "classified_df": None,
}

OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Target-Type Prompt Templates ────────────────────────────────────────────
# Each prompt tells the AI exactly what the SELECTED type looks like
# and asks it to answer yes/no — not to freely classify.

TYPE_PROMPTS = {
    "fulltime": (
        "You verify whether a German job posting is a genuine FULL-TIME position (Vollzeit).\n"
        "A full-time job means:\n"
        "- 35-40 hours per week (Vollzeit)\n"
        "- No requirement for university enrollment\n"
        "- NOT a working student (Werkstudent), internship (Praktikum), or part-time (Teilzeit) role\n"
        "- Regular employment, not a student or temporary placement\n\n"
        "If you are not sure, answer \"no\".\n\n"
        "Respond with ONLY this JSON and nothing else:\n"
        "{\"is_right_jobtype\":\"yes\" or \"no\"}"
    ),
    "parttime": (
        "You verify whether a German job posting is a genuine PART-TIME position (Teilzeit).\n"
        "A part-time job means:\n"
        "- Around 20 hours per week (Teilzeit)\n"
        "- No requirement for university enrollment\n"
        "- NOT a working student (Werkstudent), internship (Praktikum), or full-time (Vollzeit) role\n\n"
        "If you are not sure, answer \"no\".\n\n"
        "Respond with ONLY this JSON and nothing else:\n"
        "{\"is_right_jobtype\":\"yes\" or \"no\"}"
    ),
    "working_student": (
        "You verify whether a German job posting is a genuine WORKING STUDENT (Werkstudent) position.\n"
        "A working student job means:\n"
        "- Part-time, around 10-20 hours per week\n"
        "- Requires current university enrollment (eingeschrieben, immatrikuliert, laufendes Studium)\n"
        "- Flexible schedule compatible with lectures and exams\n"
        "- Often hourly pay (Werkstudentenvergütung)\n"
        "- NOT a full-time role, regular part-time without student requirement, or internship (Praktikum)\n\n"
        "If you are not sure, answer \"no\".\n\n"
        "Respond with ONLY this JSON and nothing else:\n"
        "{\"is_right_jobtype\":\"yes\" or \"no\"}"
    ),
    "internship": (
        "You verify whether a German job posting is a genuine INTERNSHIP (Praktikum).\n"
        "An internship means:\n"
        "- 20-40 hours per week\n"
        "- Fixed-term placement for practical experience\n"
        "- Listed as Praktikum, Praktikant, Intern, or internship\n"
        "- May be mandatory (Pflichtpraktikum) or voluntary (Freiwilliges Praktikum)\n"
        "- NOT a full-time permanent role, working student (Werkstudent), or regular part-time\n\n"
        "If you are not sure, answer \"no\".\n\n"
        "Respond with ONLY this JSON and nothing else:\n"
        "{\"is_right_jobtype\":\"yes\" or \"no\"}"
    ),
}

SKILLS_APPEND = (
    "\nYou also receive the candidate's skills and background. "
    "Rate the fit between the candidate's skills and the job's requirements "
    "as \"skills_matching\": an integer 0-100 (0 = no overlap, 100 = excellent match). "
    "Use the full range, not just round numbers. Also consider the skills the applicant "
    "is missing, not just the matches. Average out how much is missing compared to "
    "what skills match, and the skill level.\n\n"
    "Respond with ONLY this JSON and nothing else:\n"
    "{\"is_right_jobtype\":\"yes\" or \"no\",\"skills_matching\":integer}"
)

# ─── Ollama Helpers ──────────────────────────────────────────────────────────

def ollama_chat(system_prompt, user_prompt, model="qwen3:8b", timeout=120, retries=2):
    """Send a chat request to Ollama, return {ok, error, data}."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
        "think": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            return {"ok": True, "error": None, "data": json.loads(content)}
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            return {"ok": False, "error": f"HTTP {e.code}: {err}", "data": None}
        except urllib.error.URLError as e:
            return {"ok": False, "error": f"Cannot reach Ollama: {e}", "data": None}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return {"ok": False, "error": f"Unparseable output: {e}", "data": None}
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "data": None}
    return {"ok": False, "error": "Max retries exceeded", "data": None}


def check_ollama():
    """Probe Ollama /api/tags and update state with model list."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = sorted([m["name"] for m in data.get("models", [])])
        with lock:
            state["ollama_connected"] = True
            state["ollama_models"] = models
            state["ollama_error"] = None
    except Exception as e:
        with lock:
            state["ollama_connected"] = False
            state["ollama_models"] = []
            state["ollama_error"] = str(e)


def generate_google_query(keywords, location, model):
    """Ask Ollama to write a natural-language Google Jobs search query."""
    prompt = (
        f"Generate a single, effective Google search query to find recent job postings.\n"
        f"Keywords: {keywords}\nLocation: {location}\n"
        f"Include time constraint like 'since yesterday' or 'past week'.\n"
        f"Return ONLY the search query text, nothing else."
    )
    result = ollama_chat("You are a job search expert.", prompt, model, timeout=30)
    if result["ok"] and result["data"]:
        q = (
            result["data"]
            if isinstance(result["data"], str)
            else result["data"].get("query", result["data"].get("text", ""))
        )
        return str(q).strip().strip('"')
    return None


# ─── Scraping Helpers ────────────────────────────────────────────────────────

def _is_transient(exc):
    """Heuristic: DNS / connection blip vs real error."""
    msg = str(exc).lower()
    return type(exc).__name__ in ("ConnectionError", "TLSClientExeption") or any(
        s in msg
        for s in (
            "nameresolutionerror",
            "getaddrinfo failed",
            "no such host",
            "failed to resolve",
            "max retries exceeded",
            "temporary failure",
        )
    )


def _resolve_site(site, loc_vars, term_vars, hours_old, country="Germany"):
    """Try location × search_term variants for one site. Return (df, status_str)."""
    for loc in loc_vars:
        for term in term_vars:
            attempt = 0
            while True:
                try:
                    df = scrape_jobs(
                        site_name=[site],
                        search_term=term,
                        location=loc,
                        results_wanted=10000,
                        hours_old=hours_old,
                        country_indeed=country,
                        linkedin_fetch_description=True,
                        proxies=None,
                        verbose=0,
                    )
                except Exception as e:
                    if _is_transient(e) and attempt < 1:
                        attempt += 1
                        time.sleep(3)
                        continue
                    return pd.DataFrame(), f"error: {e}"
                if df is not None and not df.empty:
                    return df, f"done ({len(df)} jobs)"
                return pd.DataFrame(), "no results"
    return pd.DataFrame(), "no results"


def _resolve_google(query, results_wanted=50):
    """Scrape Google Jobs with a single query string."""
    try:
        df = scrape_jobs(
            site_name=["google"],
            google_search_term=query,
            results_wanted=results_wanted,
            proxies=None,
            verbose=0,
        )
        if df is not None and not df.empty:
            return df, f"done ({len(df)} jobs)"
    except Exception as e:
        return pd.DataFrame(), f"error: {e}"
    return pd.DataFrame(), "no results"


def _make_search_variants(kw):
    """Expand a keyword string into multiple search-term variants."""
    kw = kw.strip()
    return [
        f"Werkstudent {kw}",
        f"working student {kw}",
        f'"working student" ({kw})',
        kw,
        kw.replace(" ", " OR "),
    ]


def _make_loc_variants(loc):
    """Expand a location string into multiple variants."""
    loc = loc.strip()
    variants = [loc, f"{loc}, Germany"]
    if any(c in loc.lower() for c in ["münchen", "munich"]):
        variants += [
            f"{loc}, Bayern",
            f"{loc}, Bavaria",
            "Munich, Bavaria, Germany",
        ]
    seen = set()
    return [v for v in variants if not (v.lower() in seen or seen.add(v.lower()))]


def run_scrape(params):
    """Background-thread target: scrape all selected sites, deduplicate, store in state."""
    keywords = params["keywords"]
    location = params["location"]
    sites = params["sites"]
    hours_old = params.get("hours_old", 240)
    google_query = params.get("google_query", "")

    with lock:
        state["scrape_status"] = "running"
        state["scrape_progress"] = "Starting..."
        ss = {}
        for s in sites:
            ss[s] = "pending"
        if google_query:
            ss["google"] = "pending"
        state["scrape_site_status"] = ss
        state["scrape_total_raw"] = 0
        state["scrape_total_deduped"] = 0
        state["scrape_duplicates"] = 0
        state["scraped_df"] = None

    search_vars = _make_search_variants(keywords)
    loc_vars = _make_loc_variants(location)
    all_dfs = []

    for site in sites:
        with lock:
            state["scrape_site_status"][site] = "running"
            state["scrape_progress"] = f"Scraping {site}..."
        df, status = _resolve_site(site, loc_vars, search_vars, hours_old)
        with lock:
            state["scrape_site_status"][site] = status
            if not df.empty:
                all_dfs.append(df)

    if google_query:
        with lock:
            state["scrape_site_status"]["google"] = "running"
            state["scrape_progress"] = "Scraping Google..."
        gdf, gstatus = _resolve_google(google_query)
        with lock:
            state["scrape_site_status"]["google"] = gstatus
            if not gdf.empty:
                all_dfs.append(gdf)

    if all_dfs:
        jobs = pd.concat(all_dfs, ignore_index=True)
        raw_count = len(jobs)

        if "description" not in jobs.columns:
            jobs["description"] = np.nan
        jobs["description"] = jobs["description"].replace(
            r"^\s*$", np.nan, regex=True
        )

        # Deduplicate by title + company (case-insensitive)
        jobs["_tk"] = jobs["title"].str.strip().str.lower()
        jobs["_ck"] = jobs["company"].str.strip().str.lower()
        jobs = jobs.drop_duplicates(subset=["_tk", "_ck"], keep="first")
        jobs = jobs.drop(columns=["_tk", "_ck"])

        # Keep only relevant columns
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
            if c in jobs.columns
        ]
        jobs = jobs.loc[:, cols]

        # Save to disk
        jobs.to_csv(os.path.join(RESULTS_DIR, "jobs_scraped.csv"), index=False)

        with lock:
            state["scraped_df"] = jobs
            state["scrape_total_raw"] = raw_count
            state["scrape_total_deduped"] = len(jobs)
            state["scrape_duplicates"] = raw_count - len(jobs)
            state["scrape_status"] = "done"
            state["scrape_progress"] = (
                f"Done — {raw_count} raw → {len(jobs)} unique"
            )
    else:
        with lock:
            state["scraped_df"] = pd.DataFrame()
            state["scrape_status"] = "done"
            state["scrape_progress"] = "No jobs found across any site."


# ─── Classification Helpers ──────────────────────────────────────────────────

def _norm_yes_no(v):
    """Normalize a value to 'yes' or 'no'."""
    v = str(v or "").strip().lower()
    return "yes" if v in ("yes", "true", "1") else "no"


def _classify_one(title, company, job_type, description, skills_text, model, target_type):
    """Classify a single job row via Ollama against a specific target type."""
    desc = (description or "(no description)")[:4000]

    # Build system prompt based on target type
    system_prompt = TYPE_PROMPTS[target_type]
    if skills_text:
        system_prompt += SKILLS_APPEND

    user_prompt = (
        f"Title: {title or '(?)'}\n"
        f"Company: {company or '(?)'}\n"
        f"Job type (as listed by the site, may be missing or wrong): {job_type or '(?)'}\n"
        f"Description:\n{desc}"
    )

    if skills_text:
        user_prompt = (
            f"Candidate skills:\n{skills_text}\n\n"
            f"Job posting:\n{user_prompt}"
        )

    r = ollama_chat(system_prompt, user_prompt, model, timeout=90)

    if not r["ok"]:
        return {"is_right_jobtype": None, "skills_matching": None}

    d = r["data"]
    sm = d.get("skills_matching")
    try:
        sm = max(0, min(100, int(float(sm)))) if sm is not None else None
    except (TypeError, ValueError):
        sm = None

    return {
        "is_right_jobtype": _norm_yes_no(d.get("is_right_jobtype")),
        "skills_matching": sm,
    }


def _fmt_skills(skills_list):
    """Format a list of {name, level, area} dicts into text for the LLM prompt."""
    lines = []
    for s in skills_list:
        name = s.get("name", "")
        lvl = s.get("level", 50)
        area = s.get("area", "")
        lines.append(f"- {name}: {lvl}% ({area})")
    return "\n".join(lines)


def run_classify(params):
    """Background-thread target: classify all scraped jobs against a target type."""
    with lock:
        df = state["scraped_df"]
    if df is None or df.empty:
        with lock:
            state["classify_status"] = "error"
            state["classify_progress"] = "No scraped data to classify."
        return

    target_type = params.get("target_type", "working_student")
    if target_type not in TYPE_PROMPTS:
        with lock:
            state["classify_status"] = "error"
            state["classify_progress"] = f"Unknown target type: {target_type}"
        return

    skills_json = params.get("skills")
    model = params.get("model", "qwen3:8b")
    max_workers = params.get("max_workers", 6)
    skills_text = _fmt_skills(skills_json) if skills_json else None
    total = len(df)

    # Detect column names
    title_col = next(
        (c for c in ["title", "job_title"] if c in df.columns), None
    )
    desc_col = next(
        (c for c in ["description", "job_description"] if c in df.columns), None
    )
    comp_col = next(
        (c for c in ["company", "company_name"] if c in df.columns), None
    )
    jt_col = next((c for c in ["job_type"] if c in df.columns), None)

    with lock:
        state["classify_status"] = "running"
        state["classify_current"] = 0
        state["classify_total"] = total
        state["classify_progress"] = f"Classifying 0/{total}..."
        state["classify_stats"] = {"matched": 0, "rejected": 0, "error": 0}
        state["classify_target_type"] = target_type

    results_map = {}

    def _do(idx, row):
        t = str(row.get(title_col, "")) if title_col else ""
        c = str(row.get(comp_col, "")) if comp_col else ""
        jt = str(row.get(jt_col, "")) if jt_col else ""
        d = (
            str(row.get(desc_col, ""))[:1500]
            if desc_col and pd.notna(row.get(desc_col))
            else ""
        )
        r = _classify_one(t, c, jt, d, skills_text, model, target_type)
        return idx, r

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_do, i, row): i for i, row in df.iterrows()}
        done_count = 0
        for f in as_completed(futures):
            idx, r = f.result()
            results_map[idx] = r
            done_count += 1
            with lock:
                state["classify_current"] = done_count
                state["classify_progress"] = f"Classifying {done_count}/{total}..."

    # Reassemble in original order
    results = [results_map[i] for i in df.index]
    df["is_right_jobtype"] = [r["is_right_jobtype"] for r in results]
    if skills_text:
        df["skills_matching"] = [r["skills_matching"] for r in results]

    n_yes = (df["is_right_jobtype"] == "yes").sum()
    n_no = (df["is_right_jobtype"] == "no").sum()
    n_err = df["is_right_jobtype"].isna().sum()

    # Save classified to disk
    df.to_csv(os.path.join(RESULTS_DIR, "jobs_classified.csv"), index=False)
    final_df = df[df["is_right_jobtype"] == "yes"].copy()
    if "skills_matching" in final_df.columns and not final_df.empty:
        final_df = final_df.sort_values("skills_matching", ascending=False)
    final_df.to_csv(os.path.join(RESULTS_DIR, "jobs_final.csv"), index=False)

    with lock:
        state["classified_df"] = df
        state["classify_status"] = "done"
        state["classify_progress"] = (
            f"Done — {n_yes} matched, {n_no} rejected, {n_err} errors"
        )
        state["classify_stats"] = {
            "matched": int(n_yes),
            "rejected": int(n_no),
            "error": int(n_err),
        }