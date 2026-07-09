import json
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


cwd = os.getcwd()

# Ollama connection
OLLAMA_BASE_URL = "http://localhost:11434"   # change if Ollama runs on a remote host
MODEL_NAME      = "gemma4_12b_q5:latest"    # exact name shown by `ollama list`
MAX_WORKERS = 6 # parralel running instances of classification llm
# requires $env:OLLAMA_NUM_PARALLEL = "6" aas gloabal powershell command

# Input / output
CSV_PATH = f"{cwd}\\jobs_scraped.csv"   # path to your scraped jobs CSV
OUT_PATH = None         # None → auto-named  <csv_stem>_classified.csv  and  _final.csv

# Skills profile  (enables 0-100 % skills_matching column)
# Accepted CSV shape  (any columns, one row per skill):

SKILLS = f"{cwd}\\skill_profile.csv" #None

# Set to an integer (e.g. 10) to process only the first N rows — useful for quick tests
LIMIT = 10000000000000000000000000000000 #None

# ── Cell 3 · Derived URLs (do not edit) ──────────────────────────────────────
CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

VALID_CLASSIFICATIONS = {"working_student", "not_working_student", "unsure"}



CLASSIFY_SYSTEM_PROMPT = """You classify German job postings as genuine "Werkstudent" \
(working student) roles or not.

Genuine Werkstudent role — usually ALL of:
- Part-time, ~10-20h/week ("bis zu 20 Stunden", "10-20h/Woche", etc.)
- Requires current enrollment as a student ("eingeschriebener Student", \
"immatrikuliert", "laufendes Studium") at a university/Hochschule, or occasionally \
still in school working toward one
- Field of study reasonably relevant to the role
- Often hourly pay, flexible around lectures/exams

Classify as "not_working_student" if, even when the title says "Werkstudent":
- Full-time (Vollzeit, 35-40+h/week)
- An internship (Praktikum/Praktikant) or fixed-term full-time placement — this is \
a different category from Werkstudent, even if both words appear
- A regular full-time entry-level/junior/graduate/professional role
- No part-time or student framing at all

If there isn't enough detail to be sure, classify as "unsure" rather than guessing.

Respond with ONLY this JSON, nothing else:
{"classification": "working_student" or "not_working_student" or "unsure", \
"reason": "one short sentence, citing the key phrase if possible"}
"""

CLASSIFY_USER_TEMPLATE = """Title: {title}
Company: {company}
Job type (as listed by the site, may be missing or wrong): {job_type}
Description:
{description}
"""

SKILLS_SYSTEM_PROMPT = CLASSIFY_SYSTEM_PROMPT + """
You will also receive the candidate's skills and background. Rate the fit between \
the candidate's skills and the job's requirements as "skills_matching": an integer \
0-100 (0 = no overlap, 100 = excellent match). Use the full range, not just round \
numbers. Do not explain the score.

Respond with ONLY this JSON, nothing else:
{"classification": "working_student" or "not_working_student" or "unsure", \
"reason": "one short sentence on the working-student verdict", \
"skills_matching": integer 0-100}
"""

SKILLS_USER_TEMPLATE = """Candidate skills and background:
{skills}

Job posting:
Title: {title}
Company: {company}
Job type (as listed by the site, may be missing or wrong): {job_type}
Description:
{description}
"""



import urllib

def get_available_models():
    try:
        with urllib.request.urlopen(TAGS_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])], None
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def call_ollama(system_prompt, user_prompt, timeout=90, retries=2):
    """POST to Ollama chat endpoint; return dict with _ok, _error, _raw keys."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "format":  "json",
        "stream":  False,
        "options": {"temperature": 0.0},
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        CHAT_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body.get("message", {}).get("content", "")
            return {"_ok": True, "_error": None, "_raw": json.loads(content)}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                last_error = (
                    f"Model '{MODEL_NAME}' not found (HTTP 404). "
                    f"Run 'ollama list' and copy the exact name. Server: {err_body}"
                )
            else:
                last_error = f"Ollama HTTP {e.code}: {err_body}"
            break
        except urllib.error.URLError as e:
            last_error = f"Cannot reach Ollama at {CHAT_URL} — is 'ollama serve' running? ({e})"
            break
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = f"Unparseable model output (attempt {attempt + 1}): {e}"
            time.sleep(1)
        except Exception as e:
            last_error = f"Unexpected error (attempt {attempt + 1}): {type(e).__name__}: {e}"
            time.sleep(1)
    return {"_ok": False, "_error": last_error, "_raw": None}


def _normalize_classification(value):
    v = str(value or "").strip().lower()
    if v in VALID_CLASSIFICATIONS:
        return v
    aliases = {
        "working student":         "working_student",
        "yes": "working_student",  "true": "working_student",
        "not working student":     "not_working_student",
        "not_a_working_student":   "not_working_student",
        "no": "not_working_student", "false": "not_working_student",
        "uncertain": "unsure",     "unclear": "unsure", "unknown": "unsure",
    }
    return aliases.get(v, "unsure")


def classify_row(title, company, job_type, description):
    user_prompt = CLASSIFY_USER_TEMPLATE.format(
        title=title or "(missing)",
        company=company or "(missing)",
        job_type=job_type or "(missing)",
        description=(description or "(no description available)")[:4000],
    )
    result = call_ollama(CLASSIFY_SYSTEM_PROMPT, user_prompt)
    if not result["_ok"]:
        return {"classification": None,
                "reason": "Classification failed: " + str(result["_error"])}
    p = result["_raw"]
    return {"classification": _normalize_classification(p.get("classification")),
            "reason": str(p.get("reason", "")).strip()}


def classify_row_with_fit(title, company, job_type, description, skills):
    user_prompt = SKILLS_USER_TEMPLATE.format(
        skills=skills,
        title=title or "(missing)",
        company=company or "(missing)",
        job_type=job_type or "(missing)",
        description=(description or "(no description available)")[:4000],
    )
    result = call_ollama(SKILLS_SYSTEM_PROMPT, user_prompt)
    if not result["_ok"]:
        return {"classification": None,
                "reason": "Classification failed: " + str(result["_error"]),
                "skills_matching": None}
    p = result["_raw"]
    sm = p.get("skills_matching")
    try:
        sm = max(0, min(100, int(round(float(sm))))) if sm is not None else None
    except (TypeError, ValueError):
        sm = None
    return {"classification": _normalize_classification(p.get("classification")),
            "reason":         str(p.get("reason", "")).strip(),
            "skills_matching": sm}



def _fmt_json(data):
    lines = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                skill   = item.get("skill") or item.get("name") or json.dumps(item)
                extras  = {k: v for k, v in item.items() if k not in ("skill", "name")}
                extra_s = (" (" + ", ".join(f"{k}: {v}" for k, v in extras.items()) + ")") if extras else ""
                lines.append("- " + str(skill) + extra_s)
            else:
                lines.append("- " + str(item))
    elif isinstance(data, dict):
        for key, value in data.items():
            lines.append(str(key) + ": " + (", ".join(str(v) for v in value) if isinstance(value, list) else str(value)))
    else:
        lines.append(str(data))
    return "\n".join(lines)


def _fmt_csv(df):
    cols  = list(df.columns)
    lines = []
    for _, row in df.iterrows():
        parts = [f"{c}: {row[c]}" for c in cols if pd.notna(row[c])]
        if parts:
            lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def load_skills_profile(skills_arg):
    """
    Accept inline text OR a .json / .csv file path and return a single text
    block ready to drop into the LLM prompt. Falls back to plain text if the
    path doesn't exist on disk.
    """
    if skills_arg is None:
        return None
    candidate_path = skills_arg.strip()
    lower = candidate_path.lower()

    if lower.endswith(".json") and os.path.isfile(candidate_path):
        with open(candidate_path, "r", encoding="utf-8") as f:
            return _fmt_json(json.load(f))

    if lower.endswith(".csv") and os.path.isfile(candidate_path):
        return _fmt_csv(pd.read_csv(candidate_path))

    if (lower.endswith(".json") or lower.endswith(".csv")) and not os.path.isfile(candidate_path):
        print(f"WARNING: '{candidate_path}' looks like a file path but was not found — "
              "treating SKILLS as plain text.")

    return skills_arg


available = get_available_models()
if available is None:
    raise RuntimeError(
        f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
        "Make sure 'ollama serve' is running, and that OLLAMA_BASE_URL is correct."
    )

available, conn_error = get_available_models()

if available is None:
    raise RuntimeError(
        f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
        f"Connection error: {conn_error}\n\n"
        "The Windows error 'Normalerweise darf jede Socketadresse ... nur jeweils einmal verwendet werden'\n"
        "means Ollama IS already running — do NOT run 'ollama serve' again, just re-run this cell.\n"
        "If Ollama truly isn't running, open a new terminal and run: ollama serve\n"
        "If it's on a different host/port, update OLLAMA_BASE_URL in Cell 1."
    )

print(f"✓ Ollama reachable at {OLLAMA_BASE_URL}")
print(f"✓ Model '{MODEL_NAME}' found")


df = pd.read_csv(CSV_PATH)
if LIMIT:
    df = df.head(LIMIT).copy()
    print(f"LIMIT set — using first {LIMIT} rows only.")

title_col   = next((c for c in ["title",       "job_title"]       if c in df.columns), None)
desc_col    = next((c for c in ["description", "job_description"] if c in df.columns), None)
company_col = next((c for c in ["company",     "company_name"]    if c in df.columns), None)
job_type_col= next((c for c in ["job_type"]                       if c in df.columns), None)

if title_col is None:
    raise ValueError(f"No title column found. Columns present: {list(df.columns)}")
if desc_col is None:
    print("WARNING: no description column found — accuracy will be much lower.")

skills_profile = load_skills_profile(SKILLS)
fit_mode       = skills_profile is not None

print(f"Loaded {len(df)} rows from '{CSV_PATH}'")
print(f"Columns detected — title: '{title_col}' | description: '{desc_col}' | "
      f"company: '{company_col}' | job_type: '{job_type_col}'")
if fit_mode:
    print("\nSkills-matching mode ON. Profile:\n" + skills_profile)


import threading
_print_lock = threading.Lock()

def _classify_one(args):
    idx, row = args
    title       = str(row.get(title_col,    "")) if title_col    else ""
    company     = str(row.get(company_col,  "")) if company_col  else ""
    job_type    = str(row.get(job_type_col, "")) if job_type_col else ""
    description = (
        str(row.get(desc_col, ""))[:1500]          # 4000 → 1500: cuts tokens, same accuracy
        if desc_col and pd.notna(row.get(desc_col))
        else ""
    )
    if fit_mode:
        verdict = classify_row_with_fit(title, company, job_type, description, skills_profile)
        sm_str  = f", skills_matching={verdict['skills_matching']}%" if verdict["skills_matching"] is not None else ""
    else:
        verdict = classify_row(title, company, job_type, description)
        sm_str  = ""
    return idx, title, verdict, sm_str

results_map = {}
total = len(df)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(_classify_one, (i, row)): i for i, row in df.iterrows()}
    for n, future in enumerate(as_completed(futures), 1):
        idx, title, verdict, sm_str = future.result()
        results_map[idx] = verdict
        with _print_lock:
            print(f"[{n}/{total}] {title[:60]!r} → {verdict['classification']}{sm_str}")

# Reassemble in original CSV order
results = [results_map[i] for i in df.index]
print("\n✓ Classification complete.")


df["classification"] = [r["classification"] for r in results]
df["match_reason"]   = [r["reason"]         for r in results]

if fit_mode:
    df["skills_matching"] = [r["skills_matching"] for r in results]

stem      = OUT_PATH.rsplit(".", 1)[0] if OUT_PATH else CSV_PATH.rsplit(".", 1)[0]
full_path = stem + "_classified.csv"
df.to_csv(full_path, index=False)

n_ws    = (df["classification"] == "working_student").sum()
n_not   = (df["classification"] == "not_working_student").sum()
n_unsure= (df["classification"] == "unsure").sum()
n_err   = df["classification"].isna().sum()

print(f"Saved full results ({len(df)} rows) → {full_path}")
print(f"  working_student:     {n_ws}")
print(f"  not_working_student: {n_not}  ← will be dropped in final output")
print(f"  unsure:              {n_unsure}")
print(f"  errors:              {n_err}")


final_df = df[df["classification"].isin(["working_student", "unsure"])].copy()

if fit_mode and not final_df.empty:
    final_df = final_df.sort_values("skills_matching", ascending=False)

final_path = stem + "_final.csv"
final_df.to_csv(final_path, index=False)

sort_note = " (sorted by skills_matching, best first)" if fit_mode else ""
print(f"Saved final results ({len(final_df)} rows) → {final_path}{sort_note}")
print(f"  not_working_student rows dropped: {n_not}")