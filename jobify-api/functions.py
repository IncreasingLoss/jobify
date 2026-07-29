"""
Jobify — Core Functions
Scraping, Ollama communication, classification logic, shared state.
"""
import io
from pathlib import Path
import os, json, time, re, logging, threading, subprocess
import urllib, urllib.request, urllib.error
import inspect
import urllib.parse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from jobspy import scrape_jobs
    import jobspy.glassdoor as gd_module
except ImportError:
    print("ERROR: jobspy not installed. Run: pip install jobspy")
    raise SystemExit(1)

# ─── Glassdoor URL-Encoding Bug Patch ────────────────────────────────────────
def _patched_get_location(self, location: str, is_remote: bool):
    """Drop-in replacement for Glassdoor._get_location with URL-encoding fixed.
    Also handles long AI-generated location strings by falling back to the first part."""
    if not location or is_remote:
        return "11047", "STATE"  # remote options

    # Glassdoor's AJAX endpoint often fails with long strings like "Munich, Bavaria, Germany".
    # We try the full string first, but fall back to just the first part (usually the city).
    terms_to_try = [location]
    if "," in location:
        terms_to_try.append(location.split(",")[0].strip())

    for term in terms_to_try:
        encoded_term = urllib.parse.quote(term)
        url = f"{self.base_url}/findPopularLocationAjax.htm?maxLocationsToReturn=10&term={encoded_term}"
        try:
            res = self.session.get(url, headers=getattr(self, 'headers', None) or {})
        except Exception as e:
            continue

        if res.status_code == 429:
            print("[GD-LOC-ERROR] 429 Response - Blocked by Glassdoor for too many requests")
            return None, None
        if res.status_code != 200:
            continue  # Silently try the next term variation

        try:
            items = res.json()
        except ValueError:
            continue

        if not items:
            continue  # Try the next shorter term

        location_type = items[0].get("locationType", "C")
        if location_type == "C":
            location_type = "CITY"
        elif location_type == "S":
            location_type = "STATE"
        elif location_type == "N":
            location_type = "COUNTRY"
        
        return int(items[0]["locationId"]), location_type

    print(f"[GD-LOC-WARN] Location '{location}' not found on Glassdoor")
    return None, None

_patched_get_location._is_jobspy_location_patch = True

def patch_glassdoor_location_bug():
    """Apply the runtime patch only if the installed version still has the bug."""
    current = getattr(gd_module.Glassdoor, '_get_location', None)
    if current and getattr(current, "_is_jobspy_location_patch", False):
        return
    try:
        source = inspect.getsource(current)
        if "quote(" in source:
            return
    except OSError:
        pass
    gd_module.Glassdoor._get_location = _patched_get_location
    print("[Jobify] Patched Glassdoor._get_location (URL-encoding fix applied).")

# Apply patch immediately upon loading the module!
patch_glassdoor_location_bug()

# ─── Glassdoor CSRF-Token Bug Patch ──────────────────────────────────────────
def _patched_get_csrf_token(self):
    """Drop-in replacement for Glassdoor._get_csrf_token using a live bootstrap URL.
    Tries multiple URLs and adds headers to bypass Cloudflare 403s."""
    
    urls_to_try = [
        f"{self.base_url}/Job/index.htm",
        f"{self.base_url}/index.htm",
        f"{self.base_url}/Job/jobs.htm",
    ]
    
    # Ensure we send browser-like headers to avoid Cloudflare blocks
    headers = getattr(self, 'headers', None) or {}
    if 'Accept' not in headers:
        headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    if 'Accept-Language' not in headers:
        headers['Accept-Language'] = 'en-US,en;q=0.5'
    if 'Referer' not in headers:
        headers['Referer'] = self.base_url

    for url in urls_to_try:
        try:
            res = self.session.get(url, headers=headers)
            if res.status_code != 200:
                continue
                
            # Try several known patterns for Glassdoor CSRF tokens
            patterns = [
                r'"token":\s*"([^"]+)"',
                r'"csrfToken":\s*"([^"]+)"',
                r'"gdToken":\s*"([^"]+)"',
                r'name=["\']csrfToken["\']\s+value=["\']([^"\']+)["\']',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, res.text)
                if matches:
                    return matches[0]
                    
            # Fallback: look inside Next.js __NEXT_DATA__ JSON blob
            next_data_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', res.text)
            if next_data_match:
                try:
                    data = json.loads(next_data_match.group(1))
                    def find_token(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if k.lower() in ('token', 'csrftoken', 'gdtoken') and isinstance(v, str):
                                    return v
                                found = find_token(v)
                                if found: return found
                        elif isinstance(obj, list):
                            for item in obj:
                                found = find_token(item)
                                if found: return found
                        return None
                    token = find_token(data)
                    if token:
                        return token
                except Exception:
                    pass
        except Exception:
            continue

    print(f"[GD-CSRF-WARN] Failed to fetch CSRF token (likely Cloudflare block or IP ban).")
    return None

_patched_get_csrf_token._is_jobspy_csrf_patch = True

def patch_glassdoor_csrf_bug():
    """Apply the runtime patch only if the installed version still uses the dead
    bootstrap URL. Safe to call multiple times."""
    current = getattr(gd_module.Glassdoor, '_get_csrf_token', None)
    if current and getattr(current, "_is_jobspy_csrf_patch", False):
        return False
    try:
        source = inspect.getsource(current)
        if "Job/index.htm" in source:
            print("JobSpy already fixed upstream — no CSRF patch needed.")
            return False
    except OSError:
        pass
    gd_module.Glassdoor._get_csrf_token = _patched_get_csrf_token
    print("[Jobify] Patched Glassdoor._get_csrf_token (Next.js bootstrap-URL fix applied).")
    return True

patch_glassdoor_csrf_bug()

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
    "classify_target_types": [],
    "classify_used_skills": False,
    "classified_df": None,
    "match_status": "idle",
    "match_progress": "",
    "match_current": 0,
    "match_total": 0,
    "match_results": {},
}

OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Target-Type Prompt Templates ────────────────────────────────────────────
# ─── Classification Categories & Prompt ──────────────────────────────────────

CATEGORY_COLUMNS = ["fulltime", "parttime", "working_student", "internship", "remote"]

CATEGORY_LABELS = {
    "fulltime": "Full-time",
    "parttime": "Part-time",
    "working_student": "Working Student",
    "internship": "Internship",
    "remote": "Remote",
}

CLASSIFICATION_PROMPT = (
    "You are a job classification expert. Analyze the job posting and classify it into ALL applicable categories.\n\n"
    "For each category, answer \"yes\" or \"no\":\n\n"
    "- fulltime: Is this a FULL-TIME position (Vollzeit)?\n"
    "  • 35-40 hours per week\n"
    "  • No requirement for university enrollment\n"
    "  • NOT a working student (Werkstudent), internship (Praktikum), or part-time (Teilzeit) role\n"
    "  • Regular permanent employment\n\n"
    "- parttime: Is this a PART-TIME position (Teilzeit)?\n"
    "  • Around 20 hours per week\n"
    "  • No requirement for university enrollment\n"
    "  • NOT a working student, internship, or full-time role\n\n"
    "- working_student: Is this a WORKING STUDENT position (Werkstudent)?\n"
    "  • Part-time, around 10-20 hours per week\n"
    "  • Requires current university enrollment (eingeschrieben, immatrikuliert, laufendes Studium)\n"
    "  • Flexible schedule compatible with lectures and exams\n"
    "  • Often hourly pay (Werkstudentenvergütung)\n"
    "  • NOT a full-time role, regular part-time without student requirement, or internship\n\n"
    "- internship: Is this an INTERNSHIP (Praktikum)?\n"
    "  • 20-40 hours per week\n"
    "  • Fixed-term placement for practical experience\n"
    "  • Listed as Praktikum, Praktikant, Intern, or internship\n"
    "  • May be mandatory (Pflichtpraktikum) or voluntary (Freiwilliges Praktikum)\n"
    "  • NOT a full-time permanent role, working student, or regular part-time\n\n"
    "- remote: Is this job FULLY REMOTE?\n"
    "  • No on-site presence required at all\n"
    "  • Listed as \"remote\", \"fully remote\", \"100% remote\", or \"Home-Office\"\n"
    "  • NOT hybrid (partial on-site) or on-site only\n"
    "  • The job can be done entirely from anywhere\n\n"
    "IMPORTANT RULES:\n"
    "- A job can match multiple categories (e.g., a remote full-time job matches both \"fulltime\" and \"remote\")\n"
    "- Typically exactly ONE of fulltime/parttime/working_student/internship should be \"yes\" (the main employment type)\n"
    "- \"remote\" is INDEPENDENT of employment type — it describes the work location, not the contract type\n"
    "- If you are not sure about a category, answer \"no\"\n\n"
    'Respond with ONLY this JSON and nothing else:\n'
    '{"fulltime":"yes" or "no","parttime":"yes" or "no","working_student":"yes" or "no","internship":"yes" or "no","remote":"yes" or "no"}'
)

SKILLS_APPEND = (
    "\nYou also receive the candidate's skills and background. "
    "Rate the fit between the candidate's skills and the job's requirements "
    'as "skills_matching": an integer 0-100 (0 = no overlap, 100 = excellent match). '
    "Use the full range, not just round numbers. Also consider the skills the applicant "
    "is missing, not just the matches. Average out how much is missing compared to "
    "what skills match, and the skill level.\n\n"
    'Respond with ONLY this JSON and nothing else:\n'
    '{"fulltime":"yes" or "no","parttime":"yes" or "no","working_student":"yes" or "no","internship":"yes" or "no","remote":"yes" or "no","skills_matching":integer}'
)

COMPANY_APPEND = (
    "\nIf the 'Company' field is missing, empty, or just says '?', "
    "try to extract the hiring company's name from the 'Title' or 'Description' and include it as "
    '"company_name": "Extracted Name" in the JSON response. If you cannot determine it, omit the key.'
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
    """Parse a list of strings from Ollama output."""
    if not text:
        return []
    text = text.strip()
    items = []

    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            arr = json.loads(text[start:end])
            if isinstance(arr, list):
                for item in arr:
                    s = str(item).strip().strip('"').strip("'").strip()
                    if s and not s.isdigit():
                        items.append(s)
                if items:
                    return items
        except (json.JSONDecodeError, TypeError):
            pass

    for delim in [",", ";", "|"]:
        parsed = []
        for part in text.split(delim):
            part = part.strip().strip('"').strip("'").strip()
            if part and not part.isdigit():
                parsed.append(part)
        if parsed:
            return parsed

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.isdigit():
            continue
        line = re.sub(r"^[\d.)\-\*•\s]+", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if line:
            items.append(line)

    return items

def generate_variants(keywords, location, language, model):
    """Use Ollama to translate keywords into English and the target language."""
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    if not kw_list:
        kw_list = [keywords.strip()]

    loc_prompt = (
        f"Generate exactly 12 location string variations for job search engines.\n"
        f"Original location: \"{location}\"\n"
        f"Generate variations in BOTH {language} AND English.\n\n"
        f"Use ONLY these formats:\n"
        f"- city\n"
        f"- city, federal state\n"
        f"- city, country\n"
        f"- city, federalstate, country\n"
        f"- country, city\n"
        f"Do NOT use markdown code blocks. Return a RAW JSON array only, no formatting.\n"
    )
    loc_text = ollama_chat_text(
        "You are a job search expert. Output ONLY valid JSON arrays.",
        loc_prompt, model, timeout=30
    )
    loc_variants = _parse_variant_list(loc_text) if loc_text else []

    keywords_str = ", ".join(kw_list)
    
    term_prompt = (
        f"You are a job search translator.\n\n"
        f"Input keywords: {keywords_str}\n"
        f"Target language: {language}\n\n"
        f"TASK: For EACH keyword, provide:\n"
        f"1. The exact original keyword\n"
        f"2. English translation (if not already English)\n"
        f"3. {language} translation (if {language} is not English)\n\n"
        f"STRICT RULES:\n"
        f"- Output ONLY a JSON array of plain strings\n"
        f"- NO duplicates whatsoever\n"
        f"- NO combinations with OR, AND, or any operator\n"
        f"- NO prefixes like Werkstudent, Praktikum, working student, intern\n"
        f"- NO quotes, brackets, or extra formatting inside strings\n"
        f"- Each string must be ONE single job title, nothing more\n"
        f"- NO markdown code blocks\n"
    )
    
    term_text = ollama_chat_text(
        "You output ONLY valid JSON arrays. No duplicates. No OR/AND. Single job titles only.",
        term_prompt, model, timeout=30
    )
    
    raw_variants = _parse_variant_list(term_text) if term_text else []
    
    term_variants = []
    seen = set()
    for term in raw_variants:
        term = term.strip()
        if not term:
            continue
        if re.search(r'\bOR\b|\bAND\b', term, re.IGNORECASE):
            continue
        if re.search(r'^(werkstudent|working\s+student|praktikum|intern|hiwi)\s*', term, re.IGNORECASE):
            continue
        term_lower = term.lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)
        term_variants.append(term)

    if not loc_variants:
        loc_variants = _fallback_loc_variants(location)
    if not term_variants:
        term_variants = _fallback_term_variants(keywords)

    return {
        "location_variants": loc_variants[:14],
        "search_variants": term_variants[:14],
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
    """Simple fallback - just return the original keyword, no combinations."""
    kw = kw.strip()
    return [kw]


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
    """LinkedIn: first-winner only (fast). Glassdoor/Indeed: all working combos."""
    working_pairs = []

    def _probe(loc, term):
        """Fast probe: no description, small request. Returns (loc, term, df) or None."""
        try:
            kwargs = {
                "site_name": [site], 
                "search_term": term, 
                "location": loc,
                "results_wanted": 15, 
                "hours_old": hours_old,
                "linkedin_fetch_description": False,
                "proxies": None, 
                "verbose": 0,
            }
            
            if site.lower() == "indeed":
                kwargs["country_indeed"] = country
            elif site.lower() == "glassdoor":
                kwargs["country"] = country.lower()
                
            df = scrape_jobs(**kwargs)
            if df is not None and not df.empty:
                return (loc, term, df)
            print(f"[PROBE-EMPTY] site={site} loc={loc!r} term={term!r}")
        except Exception as e:
            print(f"[PROBE-ERROR] site={site} loc={loc!r} term={term!r} -> {type(e).__name__}: {e}")
        return None

    is_linkedin = site.lower() == "linkedin"
    is_glassdoor = site.lower() == "glassdoor"
    
    if is_glassdoor:
        # Process Glassdoor entirely sequentially with a small delay to avoid Cloudflare 403/400 bans
        print("[Jobify] Scraping Glassdoor sequentially to bypass Cloudflare...")
        for loc in loc_vars:
            for term in term_vars:
                result = _probe(loc, term)
                if result is not None:
                    working_pairs.append(result)
                time.sleep(1)  # Be polite to Glassdoor to avoid WAF blocks
    else:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_probe, loc, term): (loc, term)
                for loc in loc_vars for term in term_vars
            }
            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    working_pairs.append(result)
                    if is_linkedin:
                        break

    if not working_pairs:
        return pd.DataFrame(), "no results"

    results = []
    
    # Also process the full scrape sequentially for Glassdoor to avoid bursts
    for loc, term, probe_df in working_pairs:
        attempt = 0
        full_df = pd.DataFrame()
        while True:
            try:
                kwargs = {
                    "site_name": [site], 
                    "search_term": term, 
                    "location": loc,
                    "results_wanted": 10000, 
                    "hours_old": hours_old,
                    "linkedin_fetch_description": True,
                    "proxies": None, 
                    "verbose": 0,
                }
                
                if site.lower() == "indeed":
                    kwargs["country_indeed"] = country
                elif site.lower() == "glassdoor":
                    kwargs["country"] = country.lower()
                    
                full_df = scrape_jobs(**kwargs)
            except Exception as e:
                if _is_transient(e) and attempt < 1:
                    attempt += 1
                    time.sleep(3)
                    continue
                break
            break

        if full_df is not None and not full_df.empty:
            results.append(full_df)
        elif probe_df is not None and not probe_df.empty:
            results.append(probe_df)
            
        if is_glassdoor:
            time.sleep(1) # Delay between full scrapes too

    if results:
        combined = pd.concat(results, ignore_index=True)
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

    def _scrape_one_site(site):
        with lock:
            state["scrape_site_status"][site] = "running"
            state["scrape_progress"] = f"Scraping {site}..."
        try:
            if site == "google":
                df, status = _resolve_google(google_query)
            else:
                df, status = _resolve_site(site, loc_vars, term_vars, hours_old)
            with lock:
                state["scrape_site_status"][site] = status
            return df
        except Exception as e:
            with lock:
                state["scrape_site_status"][site] = f"error: {e}"
            return pd.DataFrame()

    tasks = list(sites)
    if google_query:
        tasks.append("google")

    site_dfs = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(_scrape_one_site, site): site for site in tasks}
        for f in as_completed(futures):
            df = f.result()
            if df is not None and not df.empty:
                site_dfs.append(df)

    if site_dfs:
        jobs = pd.concat(site_dfs, ignore_index=True)
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


def _classify_one(title, company, job_type, description, skills_text, model):
    desc = (description or "(no description)")[:4000]
    system_prompt = CLASSIFICATION_PROMPT
    if skills_text:
        system_prompt += SKILLS_APPEND

    needs_company = not company or str(company).strip().lower() in ('', '?', '(?)', 'none', 'nan', 'null')
    if needs_company:
        system_prompt += COMPANY_APPEND

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
        result = {f"is_{cat}": None for cat in CATEGORY_COLUMNS}
        result["skills_matching"] = None
        result["extracted_company"] = None
        return result
    d = r["data"]
    sm = d.get("skills_matching")
    try:
        sm = max(0, min(100, int(float(sm)))) if sm is not None else None
    except (TypeError, ValueError):
        sm = None

    extracted_company = None
    if needs_company:
        extracted_company = d.get("company_name")

    result = {}
    for cat in CATEGORY_COLUMNS:
        result[f"is_{cat}"] = _norm_yes_no(d.get(cat))
    result["skills_matching"] = sm
    result["extracted_company"] = extracted_company
    return result


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

    target_types = params.get("target_types", [])
    if not target_types and params.get("target_type"):
        target_types = [params["target_type"]]
    if not target_types:
        target_types = ["working_student"]

    valid_types = [t for t in target_types if t in CATEGORY_COLUMNS]
    if not valid_types:
        with lock:
            state["classify_status"] = "error"
            state["classify_progress"] = f"No valid target types: {target_types}"
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
        state["classify_target_types"] = valid_types
        state["classify_used_skills"] = used_skills

    results_map = {}

    def _do(idx, row):
        t = str(row.get(title_col, "")) if title_col else ""
        c = str(row.get(comp_col, "")) if comp_col else ""
        jt = str(row.get(jt_col, "")) if jt_col else ""
        d = (str(row.get(desc_col, ""))[:1500]
             if desc_col and pd.notna(row.get(desc_col)) else "")
        r = _classify_one(t, c, jt, d, skills_text, model)
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

    for cat in CATEGORY_COLUMNS:
        df[f"is_{cat}"] = [r[f"is_{cat}"] for r in results]

    if used_skills:
        df["skills_matching"] = [r["skills_matching"] for r in results]

    def _build_label(r):
        labels = []
        for cat in CATEGORY_COLUMNS:
            if r[f"is_{cat}"] == "yes":
                labels.append(CATEGORY_LABELS[cat])
        return ", ".join(labels) if labels else "Unclassified"

    df["job_categories"] = [_build_label(r) for r in results]

    def _matches_any(r):
        for tt in valid_types:
            if r[f"is_{tt}"] == "yes":
                return "yes"
        return "no"

    df["is_right_jobtype"] = [_matches_any(r) for r in results]

    for idx, r in zip(df.index, results):
        if r.get("extracted_company"):
            df.at[idx, "company"] = r["extracted_company"]

    n_yes = (df["is_right_jobtype"] == "yes").sum()
    n_no = (df["is_right_jobtype"] == "no").sum()
    n_err = df["is_right_jobtype"].isna().sum()

    df.to_csv(os.path.join(RESULTS_DIR, "jobs_classified.csv"), index=False)
    final_df = df[df["is_right_jobtype"] == "yes"].copy()
    if used_skills and "skills_matching" in final_df.columns and not final_df.empty:
        final_df = final_df.sort_values("skills_matching", ascending=False, na_position="last")
    final_df.to_csv(os.path.join(RESULTS_DIR, "jobs_final.csv"), index=False)

    with lock:
        state["classified_df"] = df
        state["classify_status"] = "done"
        state["classify_progress"] = f"Done — {n_yes} matched, {n_no} rejected, {n_err} errors"
        state["classify_stats"] = {
            "matched": int(n_yes), "rejected": int(n_no), "error": int(n_err),
        }

# ─── File Text Extraction ────────────────────────────────────────────────────

def extract_text_from_file(filename, content_bytes):
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            return content_bytes.decode("utf-8", errors="replace")

        elif ext == ".docx":
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paras.append(cell.text.strip())
            return "\n".join(paras) if paras else None

        elif ext == ".pdf":
            text_parts = []
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
            except ImportError:
                pass
            if not text_parts:
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(io.BytesIO(content_bytes))
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                except ImportError:
                    pass
            return "\n".join(text_parts) if text_parts else None

        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(io.BytesIO(content_bytes))
            text_parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            if para.text.strip():
                                text_parts.append(para.text.strip())
            return "\n".join(text_parts) if text_parts else None

        elif ext == ".odt":
            try:
                from odf.opendocument import load
                from odf.text import P
                doc = load(io.BytesIO(content_bytes))
                text_parts = []
                for para in doc.getElementsByType(P):
                    if para.text.strip():
                        text_parts.append(para.text.strip())
                return "\n".join(text_parts) if text_parts else None
            except ImportError:
                return None

        elif ext == ".rtf":
            raw = content_bytes.decode("utf-8", errors="replace")
            cleaned = re.sub(r"\\[a-z]+\d*[\s]?", "", raw)
            cleaned = re.sub(r"[{}]", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned if len(cleaned) > 50 else None

    except Exception:
        pass
    return None


# ─── Document Rewriting via Ollama ───────────────────────────────────────────

def _rewrite_document(doc_type, original_text, has_original,
                      job_title, job_company, job_description,
                      skills_text, language, model):
    desc_block = f"Description:\n{job_description[:6000]}"
    skills_block = skills_text if skills_text else "(No skills profile provided)"

    if has_original and original_text:
        if doc_type == "cv":
            sys_prompt = (
                "You are a professional CV editor. Your ONLY task is to make minor wording "
                "and emphasis adjustments to better align this CV with a specific job posting.\n\n"
                "ABSOLUTE RULES — VIOLATION IS UNACCEPTABLE:\n"
                "1. NEVER add any skill, technology, tool, or qualification that does NOT appear "
                "in the original CV or the candidate's skills profile below.\n"
                "2. NEVER fabricate work experience, projects, education, or certifications.\n"
                "3. NEVER change dates, durations, or factual information.\n"
                "4. NEVER invent metrics, achievements, or quantifiable results not in the original.\n"
                "5. ONLY rephrase existing content to highlight relevance to the target job.\n"
                "6. If the job requires a skill the candidate doesn't have, simply omit it — do NOT add it.\n"
                "7. Preserve the overall structure and approximate length of the original CV.\n"
                "8. Keep the same language style as the original unless instructed otherwise."
            )
            usr_prompt = (
                f"Target job:\nTitle: {job_title}\nCompany: {job_company}\n{desc_block}\n\n"
                f"Candidate's known skills:\n{skills_block}\n\n"
                f"Original CV:\n{original_text[:8000]}\n\n"
                f"Rewrite the CV in {language}. Make ONLY small, truthful adjustments to improve "
                f"alignment with this job. Output the complete rewritten CV text only — no explanations, "
                f"no markdown, no code fences."
            )
        else:
            sys_prompt = (
                "You are a professional cover letter editor. Tailor this cover letter for a "
                "specific job posting.\n\n"
                "ABSOLUTE RULES:\n"
                "1. NEVER add skills or qualifications not mentioned in the original cover letter "
                "or the candidate's skills profile below.\n"
                "2. NEVER fabricate experiences or achievements.\n"
                "3. Reference the specific company name and job title where appropriate.\n"
                "4. Adjust tone and emphasis to match the job requirements.\n"
                "5. Keep the same general length and structure.\n"
                "6. Keep the same language style as the original unless instructed otherwise."
            )
            usr_prompt = (
                f"Target job:\nTitle: {job_title}\nCompany: {job_company}\n{desc_block}\n\n"
                f"Candidate's known skills:\n{skills_block}\n\n"
                f"Original Cover Letter:\n{original_text[:6000]}\n\n"
                f"Rewrite the cover letter in {language}. Output the complete rewritten cover "
                f"letter text only — no explanations, no markdown, no code fences."
            )
    else:
        if doc_type == "cv":
            sys_prompt = (
                "You are a professional CV writer. Create a CV based ONLY on the candidate's "
                "skills profile below.\n\n"
                "RULES:\n"
                "1. Use ONLY the skills and information provided — nothing else.\n"
                "2. Do NOT invent additional skills, tools, qualifications, or experiences.\n"
                "3. Create a clean, professional CV structure.\n"
                "4. Be honest about skill levels — if a skill level is low, reflect that.\n"
                "5. Do NOT use placeholder text like [Your Name], [Date], [Phone], etc. "
                "Write complete, generic but realistic content."
            )
            usr_prompt = (
                f"Candidate's skills:\n{skills_block}\n\n"
                f"Target job:\nTitle: {job_title}\nCompany: {job_company}\n{desc_block}\n\n"
                f"Write the CV in {language}. Output the complete CV text only — no explanations, "
                f"no markdown, no code fences."
            )
        else:
            sys_prompt = (
                "You are a professional cover letter writer. Create a cover letter based ONLY on "
                "the candidate's skills profile below.\n\n"
                "RULES:\n"
                "1. Use ONLY the skills and information provided — nothing else.\n"
                "2. Do NOT invent additional skills or qualifications.\n"
                "3. Reference the company and job title.\n"
                "4. Keep it professional and concise.\n"
                "5. Do NOT use placeholder text like [Your Name], [Date], etc."
            )
            usr_prompt = (
                f"Candidate's skills:\n{skills_block}\n\n"
                f"Target job:\nTitle: {job_title}\nCompany: {job_company}\n{desc_block}\n\n"
                f"Write the cover letter in {language}. Output the complete cover letter text "
                f"only — no explanations, no markdown, no code fences."
            )

    return ollama_chat_text(sys_prompt, usr_prompt, model, timeout=180)


# ─── Run Match Jobs ──────────────────────────────────────────────────────────

def run_match_jobs(job_indices, cv_text, cl_text, cv_ext, cl_ext,
                   skills_list, language, model):
    with lock:
        classified_df = state["classified_df"]
        scraped_df = state["scraped_df"]

    if classified_df is None or classified_df.empty:
        with lock:
            state["match_status"] = "error"
            state["match_progress"] = "No classified data. Run classification first."
        return

    skills_text = _fmt_skills(skills_list) if skills_list else None
    has_cv = bool(cv_text and cv_text.strip())
    has_cl = bool(cl_text and cl_text.strip())

    if not has_cv and not has_cl:
        with lock:
            state["match_status"] = "error"
            state["match_progress"] = "Provide at least a CV or cover letter."
        return

    total_tasks = len(job_indices) * (1 if has_cv else 0) + \
                  len(job_indices) * (1 if has_cl else 0)
    completed = [0]
    results = {}

    with lock:
        state["match_status"] = "running"
        state["match_current"] = 0
        state["match_total"] = total_tasks
        state["match_progress"] = f"Preparing 0/{total_tasks}..."
        state["match_results"] = {}

    def _safe_fname(base, ext):
        safe = re.sub(r'[^a-zA-Z0-9_\-\s]', '', base).strip().replace(' ', '_')
        return f"{safe[:80]}{ext}"

    for idx in job_indices:
        row = classified_df.iloc[idx]
        title = str(row.get("title", "Unknown"))
        company = str(row.get("company", "Unknown"))

        description = ""
        if scraped_df is not None and not scraped_df.empty and idx < len(scraped_df):
            dv = scraped_df.iloc[idx].get("description")
            if pd.notna(dv):
                description = str(dv)
        if not description:
            dv = row.get("description")
            if pd.notna(dv):
                description = str(dv)

        job_id = str(int(idx))
        result = {
            "title": title,
            "company": company,
            "cv_text": None,
            "cl_text": None,
            "cv_ext": cv_ext or ".txt",
            "cl_ext": cl_ext or ".txt",
            "cv_filename": _safe_fname(f"CV_{company}_{title}", cv_ext or ".txt"),
            "cl_filename": _safe_fname(f"CoverLetter_{company}_{title}", cl_ext or ".txt"),
            "error": None,
        }

        try:
            if has_cv:
                result["cv_text"] = _rewrite_document(
                    "cv", cv_text, True, title, company, description,
                    skills_text, language, model,
                ) or ""
                with lock:
                    completed[0] += 1
                    state["match_current"] = completed[0]
                    state["match_progress"] = (
                        f"Processing {completed[0]}/{total_tasks} — "
                        f"CV for: {title}"
                    )

            if has_cl:
                result["cl_text"] = _rewrite_document(
                    "cl", cl_text, True, title, company, description,
                    skills_text, language, model,
                ) or ""
                with lock:
                    completed[0] += 1
                    state["match_current"] = completed[0]
                    state["match_progress"] = (
                        f"Processing {completed[0]}/{total_tasks} — "
                        f"Cover Letter for: {title}"
                    )

        except Exception as e:
            result["error"] = str(e)
            with lock:
                completed[0] += (1 if has_cv else 0) + (1 if has_cl else 0)
                state["match_current"] = completed[0]

        results[job_id] = result

    with lock:
        state["match_results"] = results
        state["match_status"] = "done"
        n_err = sum(1 for r in results.values() if r.get("error"))
        state["match_progress"] = (
            f"Done — {len(results)} jobs processed, {n_err} error(s)"
        )