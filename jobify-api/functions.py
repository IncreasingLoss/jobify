"""
Jobify — Core Functions
Scraping, Ollama communication, classification logic, shared state.
"""

import os, json, time, re, logging, threading, subprocess
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
SKILLS_FILE = os.path.join(RESULTS_DIR, "skills_profile.csv")

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
    "classify_used_skills": False,
    "classified_df": None,
}

OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Target-Type Prompt Templates ────────────────────────────────────────────
TYPE_PROMPTS = {
    "fulltime": (
        "You verify whether a German job posting is a genuine FULL-TIME position (Vollzeit).\n"
        "A full-time job means:\n"
        "- 35-40 hours per week (Vollzeit)\n"
        "- No requirement for university enrollment\n"
        "- NOT a working student (Werkstudent), internship (Praktikum), or part-time (Teilzeit) role\n"
        "- Regular employment, not a student or temporary placement\n\n"
        'If you are not sure, answer "no".\n\n'
        'Respond with ONLY this JSON and nothing else:\n'
        '{"is_right_jobtype":"yes" or "no"}'
    ),
    "parttime": (
        "You verify whether a German job posting is a genuine PART-TIME position (Teilzeit).\n"
        "A part-time job means:\n"
        "- Around 20 hours per week (Teilzeit)\n"
        "- No requirement for university enrollment\n"
        "- NOT a working student (Werkstudent), internship (Praktikum), or full-time (Vollzeit) role\n\n"
        'If you are not sure, answer "no".\n\n'
        'Respond with ONLY this JSON and nothing else:\n'
        '{"is_right_jobtype":"yes" or "no"}'
    ),
    "working_student": (
        "You verify whether a German job posting is a genuine WORKING STUDENT (Werkstudent) position.\n"
        "A working student job means:\n"
        "- Part-time, around 10-20 hours per week\n"
        "- Requires current university enrollment (eingeschrieben, immatrikuliert, laufendes Studium)\n"
        "- Flexible schedule compatible with lectures and exams\n"
        "- Often hourly pay (Werkstudentenvergütung)\n"
        "- NOT a full-time role, regular part-time without student requirement, or internship (Praktikum)\n\n"
        'If you are not sure, answer "no".\n\n'
        'Respond with ONLY this JSON and nothing else:\n'
        '{"is_right_jobtype":"yes" or "no"}'
    ),
    "internship": (
        "You verify whether a German job posting is a genuine INTERNSHIP (Praktikum).\n"
        "An internship means:\n"
        "- 20-40 hours per week\n"
        "- Fixed-term placement for practical experience\n"
        "- Listed as Praktikum, Praktikant, Intern, or internship\n"
        "- May be mandatory (Pflichtpraktikum) or voluntary (Freiwilliges Praktikum)\n"
        "- NOT a full-time permanent role, working student (Werkstudent), or regular part-time\n\n"
        'If you are not sure, answer "no".\n\n'
        'Respond with ONLY this JSON and nothing else:\n'
        '{"is_right_jobtype":"yes" or "no"}'
    ),
}

SKILLS_APPEND = (
    "\nYou also receive the candidate's skills and background. "
    "Rate the fit between the candidate's skills and the job's requirements "
    'as "skills_matching": an integer 0-100 (0 = no overlap, 100 = excellent match). '
    "Use the full range, not just round numbers. Also consider the skills the applicant "
    "is missing, not just the matches. Average out how much is missing compared to "
    "what skills match, and the skill level.\n\n"
    'Respond with ONLY this JSON and nothing else:\n'
    '{"is_right_jobtype":"yes" or "no","skills_matching":integer}'
)


# ─── VRAM & Worker Calculation ───────────────────────────────────────────────

def get_vram_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            total_mb = 0
            free_mb = 0
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        total_mb += int(float(parts[0]))
                        free_mb += int(float(parts[1]))
                    except ValueError:
                        continue
            if total_mb > 0:
                return {"total_mb": total_mb, "free_mb": free_mb}
    except Exception:
        pass
    return None


def get_model_info(model_name):
    try:
        payload = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/show", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        details = data.get("details", {})
        model_info = data.get("model_info", {})
        param_count = model_info.get("general.parameter_count", 0)
        param_size_str = details.get("parameter_size", "0B")
        quant = details.get("quantization_level", "Q4_K_M")
        if param_count == 0 and param_size_str:
            ps = param_size_str.strip().upper()
            if ps.endswith("B"):
                param_count = float(ps[:-1]) * 1e9
            elif ps.endswith("M"):
                param_count = float(ps[:-1]) * 1e6
            elif ps.endswith("T"):
                param_count = float(ps[:-1]) * 1e12
        return {
            "parameter_count": int(param_count),
            "parameter_size_str": param_size_str,
            "quantization": quant,
        }
    except Exception:
        return None


def calculate_max_workers(model_name):
    vram = get_vram_info()
    if not vram:
        return {"workers": 1, "vram": None, "reason": "No GPU detected via nvidia-smi"}
    minfo = get_model_info(model_name)
    if not minfo:
        return {"workers": 1, "vram": vram, "reason": f"Could not read model info for {model_name}"}
    params_b = minfo["parameter_count"] / 1e9
    q = minfo["quantization"].upper()
    bpp = 0.6
    if "Q2" in q: bpp = 0.35
    elif "Q3" in q: bpp = 0.45
    elif "Q4" in q: bpp = 0.6
    elif "Q5" in q: bpp = 0.75
    elif "Q6" in q: bpp = 0.85
    elif "Q8" in q: bpp = 1.0
    elif "F16" in q or "FP16" in q: bpp = 2.0
    elif "F32" in q or "FP32" in q: bpp = 4.0
    model_vram_gb = params_b * bpp
    kv_per_worker_gb = max(0.15, params_b * 0.04)
    total_gb = vram["total_mb"] / 1024
    available = total_gb - model_vram_gb - 1.0
    if available <= 0:
        return {"workers": 1, "vram": vram,
                "reason": f"Model needs ~{model_vram_gb:.1f}GB, only {total_gb:.1f}GB total",
                "model_vram_gb": round(model_vram_gb, 1)}
    workers = int(available / kv_per_worker_gb)
    workers = max(1, min(workers, 30))
    return {"workers": workers, "vram": vram,
            "reason": f"{params_b:.1f}B params @ {q} = ~{model_vram_gb:.1f}GB model, "
                      f"~{kv_per_worker_gb:.2f}GB/worker KV, {available:.1f}GB available",
            "model_vram_gb": round(model_vram_gb, 1),
            "kv_per_worker_gb": round(kv_per_worker_gb, 3)}


# ─── Ollama Helpers ──────────────────────────────────────────────────────────

def ollama_chat(system_prompt, user_prompt, model="qwen3:8b", timeout=120, retries=2):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json", "stream": False,
        "options": {"temperature": 0.0}, "think": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
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


def ollama_chat_text(system_prompt, user_prompt, model="qwen3:8b", timeout=30):
    """Send a chat request WITHOUT forcing JSON format. Returns raw text or None."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
        "think": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("message", {}).get("content", "").strip() or None
    except Exception:
        return None


def check_ollama():
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
    prompt = (
        f"Generate a single, effective Google search query to find recent job postings.\n"
        f"Keywords: {keywords}\nLocation: {location}\n"
        f"Include time constraint like 'since yesterday' or 'past week'.\n"
        f"Return ONLY the search query text, nothing else."
    )
    return ollama_chat_text("You are a job search expert.", prompt, model, timeout=30)


# ─── AI Variant Generation ───────────────────────────────────────────────────

def _parse_variant_list(text):
    """Parse a list of strings from Ollama output. Handles JSON arrays, numbered lists, bullet lists, comma-separated."""
    if not text:
        return []
    text = text.strip()

    # Try JSON array first
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            arr = json.loads(text[start:end])
            if isinstance(arr, list):
                items = []
                for item in arr:
                    s = str(item).strip().strip('"').strip("'").strip()
                    if s and not s.isdigit():
                        items.append(s)
                if items:
                    return items
        except (json.JSONDecodeError, TypeError):
            pass

    # Try numbered list: "1. item" or "1) item"
    numbered = re.findall(r"(?:^\d+[.)\s]+)(.+)$", text, re.MULTILINE)
    if numbered:
        items = [m[1].strip().strip('"').strip("'").strip() for m in numbered]
        items = [s for s in items if s and not s.isdigit()]
        if items:
            return items

    # Try bullet list: "- item" or "* item"
    bullets = re.findall(r"^[•*\-]\s+(.+)$", text, re.MULTILINE)
    if bullets:
        items = [m.strip().strip('"').strip("'").strip() for m in bullets]
        items = [s for s in items if s and not s.isdigit()]
        if items:
            return items

    # Fallback: split by newlines, then commas
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.isdigit():
            continue
        # Strip leading number/bullet
        line = re.sub(r"^[\d.)\-\*•\s]+", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if line:
            items.append(line)
    if not items:
        for part in text.split(","):
            part = part.strip().strip('"').strip("'").strip()
            if part and not part.isdigit():
                items.append(part)

    return items


def generate_variants(keywords, location, language, model):
    """Use Ollama to generate location and search term variations in multiple languages."""

    # --- Location variants ---
    loc_prompt = (
        f"Generate exactly 20 location string variations for job search engines.\n"
        f"Original location: \"{location}\"\n"
        f"Generate variations in BOTH {language} AND English.\n\n"
        f"Include all of these formats:\n"
        f'- City only (both languages): "München", "Munich"\n'
        f'- City + country: "München, Germany", "Munich, Germany"\n'
        f'- City + state/region (both languages): "München, Bayern", "Munich, Bavaria"\n'
        f'- City + state + country: "Munich, Bavaria, Germany"\n'
        f'- With dashes: "München-Germany", "Munich-Germany"\n'
        f'- With "near": "near Munich", "near München"\n'
        f'- Common abbreviations and local spellings\n'
        f'- Umlaut and non-umlaut versions\n\n'
        f"Return ONLY a JSON array of strings. No explanation, no markdown, no numbering.\n"
        f'Example: ["München", "Munich", "München, Germany", "Munich, Germany", ...]'
    )
    loc_text = ollama_chat_text(
        "You are a job search localization expert. Output ONLY valid JSON arrays.",
        loc_prompt, model, timeout=30
    )
    loc_variants = _parse_variant_list(loc_text) if loc_text else []

    # --- Search term variants ---
    term_prompt = (
        f"Generate exactly 20 search keyword variations for job search engines.\n"
        f"Original keywords: \"{keywords}\"\n"
        f"Generate variations in BOTH {language} AND English.\n\n"
        f"Include all of these:\n"
        f"- Exact keywords as-is\n"
        f'- Common misspellings: "analyst" → also "analist", "scientist" → "scientis"\n'
        f"- Translations to {language} (if different from English)\n"
        f"- Related/adjacent job titles: broader and narrower roles\n"
        f'- Combined with OR: "data scientist OR data analyst"\n'
        f'- Abbreviated forms\n'
        f"- Compound variations\n\n"
        f"Do NOT include employment type prefixes like \"Werkstudent\" or \"working student\" "
        f"or \"Praktikum\" — those are handled separately by the system.\n\n"
        f"Return ONLY a JSON array of strings. No explanation, no markdown, no numbering.\n"
        f'Example: ["Data Scientist", "Datenwissenschaftler", "Data Analyst", ...]'
    )
    term_text = ollama_chat_text(
        "You are a job search expert. Output ONLY valid JSON arrays.",
        term_prompt, model, timeout=30
    )
    term_variants = _parse_variant_list(term_text) if term_text else []

    # Fallbacks if AI failed
    if not loc_variants:
        loc_variants = _fallback_loc_variants(location)
    if not term_variants:
        term_variants = _fallback_term_variants(keywords)

    return {
        "location_variants": loc_variants[:20],
        "search_variants": term_variants[:20],
    }


def _fallback_loc_variants(loc):
    """Simple hardcoded fallback if AI variant generation fails."""
    loc = loc.strip()
    variants = [loc, f"{loc}, Germany"]
    low = loc.lower()
    if any(c in low for c in ["münchen", "munich"]):
        variants += ["München", "Munich", "münchen", "munich",
                     "München, Bayern", "Munich, Bavaria",
                     "München, Germany", "Munich, Germany",
                     "Munich, Bavaria, Germany", "München, Bayern, Germany",
                     "München-Germany", "Munich-Germany",
                     "near Munich", "near München",
                     "Bayern", "Bavaria"]
    elif any(c in low for c in ["berlin"]):
        variants += ["Berlin", "berlin", "Berlin, Germany",
                     "Berlin, Brandenburg", "near Berlin"]
    elif any(c in low for c in ["hamburg"]):
        variants += ["Hamburg", "hamburg", "Hamburg, Germany",
                     "Hamburg, Niedersachsen", "near Hamburg"]
    elif any(c in low for c in ["frankfurt"]):
        variants += ["Frankfurt", "Frankfurt am Main", "Frankfurt, Germany",
                     "Frankfurt, Hessen", "near Frankfurt"]
    else:
        variants += [loc, f"{loc}, Germany", f"near {loc}"]
    seen = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def _fallback_term_variants(kw):
    """Simple hardcoded fallback if AI variant generation fails."""
    kw = kw.strip()
    return [
        kw,
        f"Werkstudent {kw}",
        f"working student {kw}",
        f'"working student" ({kw})',
        kw.replace(" ", " OR "),
    ]


# ─── Skills Persistence ──────────────────────────────────────────────────────

def save_skills_to_file(skills_list):
    if not skills_list:
        return False
    try:
        df = pd.DataFrame(skills_list)
        df.to_csv(SKILLS_FILE, index=False)
        return True
    except Exception:
        return False


def load_skills_from_file():
    if not os.path.isfile(SKILLS_FILE):
        return None
    try:
        df = pd.read_csv(SKILLS_FILE)
        if "name" not in df.columns:
            return None
        result = []
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            lvl = 50
            if pd.notna(row.get("level")):
                try:
                    lvl = max(0, min(100, int(float(row["level"]))))
                except (TypeError, ValueError):
                    pass
            area = "other"
            if pd.notna(row.get("area")):
                area = str(row["area"]).strip() or "other"
            result.append({"name": name, "level": lvl, "area": area})
        return result if result else None
    except Exception:
        return None


def parse_skills_csv_bytes(content_bytes):
    text = content_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    for delim in ["|", ",", "\t"]:
        lines = text.split("\n")
        parsed = []
        for line in lines:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            cells = [c.strip() for c in line.split(delim)]
            if len(cells) < 2:
                continue
            header_words = {"name", "skill", "skill_name", "skill_lvl", "level", "area", "0"}
            name = None
            name_idx = -1
            for idx, cell in enumerate(cells):
                c = cell.strip()
                if not c or c.lower() in header_words or c.isdigit():
                    continue
                name = c
                name_idx = idx
                break
            if not name:
                continue
            level = 50
            for idx, cell in enumerate(cells):
                if idx == name_idx:
                    continue
                c = cell.strip().replace("%", "")
                try:
                    v = int(float(c))
                    if 0 <= v <= 100:
                        level = v
                        break
                except (TypeError, ValueError):
                    pass
            area = "other"
            for idx in range(len(cells) - 1, -1, -1):
                if idx == name_idx:
                    continue
                c = cells[idx].strip()
                if not c or c.isdigit():
                    continue
                try:
                    int(float(c.replace("%", "")))
                    continue
                except (TypeError, ValueError):
                    pass
                if c.lower() in header_words:
                    continue
                area = c
                break
            parsed.append({"name": name, "level": level, "area": area})
        if parsed:
            return parsed
    return []


# ─── Scraping Helpers ────────────────────────────────────────────────────────

def _is_transient(exc):
    msg = str(exc).lower()
    return type(exc).__name__ in ("ConnectionError", "TLSClientExeption") or any(
        s in msg for s in (
            "nameresolutionerror", "getaddrinfo failed", "no such host",
            "failed to resolve", "max retries exceeded", "temporary failure",
        )
    )


def _resolve_site(site, loc_vars, term_vars, hours_old, country="Germany"):
    """Try ALL combinations of location × search_term. Return (df, status_str)."""
    results = []
    for loc in loc_vars:
        for term in term_vars:
            attempt = 0
            while True:
                try:
                    df = scrape_jobs(
                        site_name=[site], search_term=term, location=loc,
                        results_wanted=10000, hours_old=hours_old,
                        country_indeed=country, linkedin_fetch_description=True,
                        proxies=None, verbose=0,
                    )
                except Exception as e:
                    if _is_transient(e) and attempt < 1:
                        attempt += 1
                        time.sleep(3)
                        continue
                    # Continue trying other combinations instead of giving up
                    break
                if df is not None and not df.empty:
                    results.append(df)
    if results:
        combined = pd.concat(results, ignore_index=True)
        # Deduplicate within this site
        if "title" in combined.columns and "company" in combined.columns:
            combined["_tk"] = combined["title"].str.strip().str.lower()
            combined["_ck"] = combined["company"].str.strip().str.lower()
            combined = combined.drop_duplicates(subset=["_tk", "_ck"], keep="first")
            combined = combined.drop(columns=["_tk", "_ck"])
        return combined, f"done ({len(combined)} jobs)"
    return pd.DataFrame(), "no results"


def _resolve_google(query, results_wanted=50):
    try:
        df = scrape_jobs(
            site_name=["google"], google_search_term=query,
            results_wanted=results_wanted, proxies=None, verbose=0,
        )
        if df is not None and not df.empty:
            return df, f"done ({len(df)} jobs)"
    except Exception as e:
        return pd.DataFrame(), f"error: {e}"
    return pd.DataFrame(), "no results"


def run_scrape(params):
    keywords = params["keywords"]
    location = params["location"]
    sites = params["sites"]
    hours_old = params.get("hours_old", 240)
    google_query = params.get("google_query", "")

    # Use AI-generated variants if provided, otherwise fallback
    loc_vars = params.get("location_variants") or _fallback_loc_variants(location)
    term_vars = params.get("search_variants") or _fallback_term_variants(keywords)

    with lock:
        state["scrape_status"] = "running"
        state["scrape_progress"] = "Starting..."
        ss = {s: "pending" for s in sites}
        if google_query:
            ss["google"] = "pending"
        state["scrape_site_status"] = ss
        state["scrape_total_raw"] = 0
        state["scrape_total_deduped"] = 0
        state["scrape_duplicates"] = 0
        state["scraped_df"] = None

    all_dfs = []

    for site in sites:
        with lock:
            state["scrape_site_status"][site] = "running"
            state["scrape_progress"] = f"Scraping {site} ({len(loc_vars)}×{len(term_vars)} variants)..."
        df, status = _resolve_site(site, loc_vars, term_vars, hours_old)
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
        jobs["description"] = jobs["description"].replace(r"^\s*$", np.nan, regex=True)
        jobs["_tk"] = jobs["title"].str.strip().str.lower()
        jobs["_ck"] = jobs["company"].str.strip().str.lower()
        jobs = jobs.drop_duplicates(subset=["_tk", "_ck"], keep="first")
        jobs = jobs.drop(columns=["_tk", "_ck"])
        cols = [c for c in [
            "id", "site", "job_url", "title", "company", "location",
            "date_posted", "emails", "description", "company_url", "job_type",
        ] if c in jobs.columns]
        jobs = jobs.loc[:, cols]
        jobs.to_csv(os.path.join(RESULTS_DIR, "jobs_scraped.csv"), index=False)
        with lock:
            state["scraped_df"] = jobs
            state["scrape_total_raw"] = raw_count
            state["scrape_total_deduped"] = len(jobs)
            state["scrape_duplicates"] = raw_count - len(jobs)
            state["scrape_status"] = "done"
            state["scrape_progress"] = f"Done — {raw_count} raw → {len(jobs)} unique"
    else:
        with lock:
            state["scraped_df"] = pd.DataFrame()
            state["scrape_status"] = "done"
            state["scrape_progress"] = "No jobs found across any site."


# ─── Classification Helpers ──────────────────────────────────────────────────

def _norm_yes_no(v):
    v = str(v or "").strip().lower()
    return "yes" if v in ("yes", "true", "1") else "no"


def _classify_one(title, company, job_type, description, skills_text, model, target_type):
    desc = (description or "(no description)")[:4000]
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
        user_prompt = f"Candidate skills:\n{skills_text}\n\nJob posting:\n{user_prompt}"
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
    lines = []
    for s in skills_list:
        lines.append(f"- {s.get('name', '')}: {s.get('level', 50)}% ({s.get('area', '')})")
    return "\n".join(lines)


def run_classify(params):
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
    used_skills = skills_text is not None
    total = len(df)
    title_col = next((c for c in ["title", "job_title"] if c in df.columns), None)
    desc_col = next((c for c in ["description", "job_description"] if c in df.columns), None)
    comp_col = next((c for c in ["company", "company_name"] if c in df.columns), None)
    jt_col = next((c for c in ["job_type"] if c in df.columns), None)
    with lock:
        state["classify_status"] = "running"
        state["classify_current"] = 0
        state["classify_total"] = total
        state["classify_progress"] = f"Classifying 0/{total}..."
        state["classify_stats"] = {"matched": 0, "rejected": 0, "error": 0}
        state["classify_target_type"] = target_type
        state["classify_used_skills"] = used_skills
    results_map = {}

    def _do(idx, row):
        t = str(row.get(title_col, "")) if title_col else ""
        c = str(row.get(comp_col, "")) if comp_col else ""
        jt = str(row.get(jt_col, "")) if jt_col else ""
        d = (str(row.get(desc_col, ""))[:1500]
             if desc_col and pd.notna(row.get(desc_col)) else "")
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
    results = [results_map[i] for i in df.index]
    df["is_right_jobtype"] = [r["is_right_jobtype"] for r in results]
    if used_skills:
        df["skills_matching"] = [r["skills_matching"] for r in results]
    n_yes = (df["is_right_jobtype"] == "yes").sum()
    n_no = (df["is_right_jobtype"] == "no").sum()
    n_err = df["is_right_jobtype"].isna().sum()
    df.to_csv(os.path.join(RESULTS_DIR, "jobs_classified.csv"), index=False)
    final_df = df[df["is_right_jobtype"] == "yes"].copy()
    if used_skills and "skills_matching" in final_df.columns and not final_df.empty:
        final_df = final_df.sort_values("skills_matching", ascending=False)
    final_df.to_csv(os.path.join(RESULTS_DIR, "jobs_final.csv"), index=False)
    with lock:
        state["classified_df"] = df
        state["classify_status"] = "done"
        state["classify_progress"] = f"Done — {n_yes} matched, {n_no} rejected, {n_err} errors"
        state["classify_stats"] = {
            "matched": int(n_yes), "rejected": int(n_no), "error": int(n_err),
        }