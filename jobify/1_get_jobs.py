import time
import logging
import pandas as pd
import numpy as np
from jobspy import scrape_jobs

# --- Silence expected Glassdoor noise ---------------------------------
# While probing multiple location/search-term variants, Glassdoor logs an
# ERROR-level "location not parsed" + "response status code 400" pair for
# every variant that doesn't resolve. That's expected/handled by our own
# retry loop below, not a real failure, so we filter just those two
# messages out. Any other error from that logger (e.g. auth/blocked) still
# gets printed normally.
class _SilenceGlassdoorLocationNoise(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not (
            "location not parsed" in msg
            or "response status code 400" in msg
        )

logging.getLogger("JobSpy:Glassdoor").addFilter(_SilenceGlassdoorLocationNoise())

# Configuration
LOCATION_VARIANTS = [
    "München, Bayern",
    "Munich, Bavaria",
    "München",
    "Munich",
    "Munich, Germany",
    "Bayern",
    "Bavaria, Germany",
]

SEARCH_TERM_VARIANTS = [
    "Werkstudent Data Scientist Analyst",
    "Werkstudent Data Science",
    '"working student" (data scientist OR analyst)',
    "working student data analyst",
    "Werkstudent Data", 
    "werkstudent ai", 
    "werkstudent ml",
    "werkstudent ki",
    "working student ai",
    "working student ki",
    "working student ml",
    "Werkstudent ML",
    "Werkstudent AI",
    "Werkstudent KI",
    "working Student AI",
    "working student KI",
    "working student ML",
]

GOOGLE_SEARCH_TERM_VARIANTS = [
    "working student data scientist or analyst jobs in Munich, Germany since yesterday",
    "Werkstudent data scientist jobs near Munich, Germany",
    "data analyst working student jobs in Munich Germany",
]

SITES_TO_TRY = ["glassdoor", "indeed", "linkedin"]
COUNTRY_INDEED = "Germany"
RESULTS_WANTED = 20000000000000
HOURS_OLD = 240
LINKEDIN_FETCH_DESCRIPTION = True
PROXIES = None

def _is_transient_network_error(exc):
    """Heuristic: does this look like a DNS/connectivity blip rather than a
    real API/parsing error? Covers requests' ConnectionError, tls_client's
    TLSClientExeption (used by the Glassdoor scraper), and the underlying
    urllib3 NameResolutionError — each site raises DNS failures differently."""
    msg = str(exc).lower()
    dns_signals = (
        "nameresolutionerror",
        "getaddrinfo failed",
        "no such host",
        "failed to resolve",
        "max retries exceeded",
        "temporary failure in name resolution",
    )
    return type(exc).__name__ in ("ConnectionError", "TLSClientExeption") or any(
        s in msg for s in dns_signals
    )

def resolve_and_scrape(site, location_variants, search_term_variants,
                        results_wanted=RESULTS_WANTED, hours_old=72, country_indeed=COUNTRY_INDEED,
                        linkedin_fetch_description=False, proxies=None,
                        verbose=1, dns_retries=1, retry_delay=3):
    """Try (location, search_term) combinations for one site until one works.
    Returns (dataframe, location_used, search_term_used) — dataframe is empty
    if every combination failed.

    A transient DNS/connection blip on one variant no longer burns through
    the rest of the list: the SAME (location, search_term) combo is retried
    up to `dns_retries` times, `retry_delay` seconds apart, before moving on.
    Real "no jobs for this variant" results are left alone and still advance
    to the next variant immediately.
    """
    for location in location_variants:
        for search_term in search_term_variants:
            attempt = 0
            while True:
                try:
                    df = scrape_jobs(
                        site_name=[site],
                        search_term=search_term,
                        location=location,
                        results_wanted=results_wanted,
                        hours_old=hours_old,
                        country_indeed=country_indeed,
                        linkedin_fetch_description=linkedin_fetch_description,
                        proxies=proxies,
                        verbose=verbose,
                    )
                except Exception as e:
                    if _is_transient_network_error(e) and attempt < dns_retries:
                        attempt += 1
                        #print(f"  [{site}] location={location!r} search_term={search_term!r} "
                        #      f"-> {type(e).__name__} (looks transient), retry {attempt}/{dns_retries} "
                        #      f"in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    #print(f"  [{site}] location={location!r} search_term={search_term!r} -> raised {type(e).__name__}: {e}")
                    break
                if df is not None and not df.empty:
                    print(f"  [{site}] SUCCESS with location={location!r} search_term={search_term!r} -> {len(df)} jobs")
                    return df, location, search_term
        #        print(f"  [{site}] location={location!r} search_term={search_term!r} -> 0 jobs, trying next variant")
                break
    print(f"  [{site}] All variants exhausted — no results found.")
    return pd.DataFrame(), None, None

def resolve_google(google_search_term_variants, results_wanted=20, proxies=None, verbose=1):
    for term in google_search_term_variants:
        try:
            df = scrape_jobs(
                site_name=["google"],
                google_search_term=term,
                results_wanted=results_wanted,
                proxies=proxies,
                verbose=verbose,
            )
        except Exception as e:
        #    print(f"  [google] google_search_term={term!r} -> raised {type(e).__name__}: {e}")
            continue
        if df is not None and not df.empty:
            print(f"  [google] SUCCESS with google_search_term={term!r} -> {len(df)} jobs")
            return df, term
    print("  [google] All variants exhausted — no results found.")
    return pd.DataFrame(), None

if __name__ == "__main__":
    results = {}
    for site in SITES_TO_TRY:
        #print(f"Resolving site: {site}")
        df, used_location, used_term = resolve_and_scrape(
            site,
            LOCATION_VARIANTS,
            SEARCH_TERM_VARIANTS,
            results_wanted=RESULTS_WANTED,
            hours_old=HOURS_OLD,
            country_indeed=COUNTRY_INDEED,
            linkedin_fetch_description=LINKEDIN_FETCH_DESCRIPTION,
            proxies=PROXIES,
        )
        results[site] = {"df": df, "location": used_location, "search_term": used_term}

    #print("Resolving site: google")
    google_df, google_term_used = resolve_google(GOOGLE_SEARCH_TERM_VARIANTS, results_wanted=RESULTS_WANTED, proxies=PROXIES)
    results["google"] = {"df": google_df, "location": None, "search_term": google_term_used}

    for site, info in results.items():
        n = len(info["df"])
        used_loc = info["location"]
        used_term = info["search_term"]
        #print(f"{site}: {n} jobs (location={used_loc!r}, search_term={used_term!r})")

    all_dfs = [info["df"] for info in results.values() if not info["df"].empty]

    if all_dfs:
        jobs = pd.concat(all_dfs, ignore_index=True)
    else:
        jobs = pd.DataFrame()

    if not jobs.empty:
        if "description" not in jobs.columns:
            jobs["description"] = np.nan
        # Normalize: empty strings / None -> NaN, keep real text as-is (raw, unmodified)
        jobs["description"] = jobs["description"].replace(r"^\s*$", np.nan, regex=True)
        jobs["description"] = jobs["description"].where(jobs["description"].notna(), np.nan)

        n_with_desc = jobs["description"].notna().sum()
        print(f"Total jobs: {len(jobs)} | with description: {n_with_desc} | without: {len(jobs) - n_with_desc}")
    else:
        print("No jobs found across any site/variant combination.")

    if not jobs.empty:
        jobs = jobs.loc[:,['id', 'site', 'job_url', 'title', 'company','location', 'date_posted', 'emails', 'description', 'company_url']]

        # deduplicate by title + company (case/whitespace insensitive) - so that same listings on different platforms get filtered out  
        jobs['_title_key'] = jobs['title'].str.strip().str.lower()
        jobs['_company_key'] = jobs['company'].str.strip().str.lower()
        jobs_dropped = jobs.drop_duplicates(subset=['_title_key', '_company_key'], keep='first')
        jobs_dropped = jobs_dropped.drop(columns=['_title_key', '_company_key'])

        jobs_dropped.to_csv("jobify/jobs_scraped.csv")
        print(jobs_dropped.head(8))