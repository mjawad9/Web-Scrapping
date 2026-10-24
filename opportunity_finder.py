"""
Opportunity Finder v4: jobs and internships in one search, with CV matching
and an application tracker.

Search sources (official public APIs / RSS feeds)
  - Jobicy, Remote OK, We Work Remotely, The Muse
  - Google buttons (site: filters) for LinkedIn, Indeed Pakistan, Rozee.pk,
    Mustakbil and Bayt, plus Google Jobs. Those sites do not allow scraping,
    so the app opens Google for you instead.

New in v4
  - CV matching: upload or paste your CV and every result gets a Match %
    and the list of skills you share with the listing (runs locally, no AI key).
  - Application tracker: save jobs, set status and deadline, add notes,
    download / restore your tracker as CSV. Nothing is stored on the server.

Install:   pip install -r requirements.txt
Run:       streamlit run opportunity_finder.py
"""

import html
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "OpportunityFinderBot/4.0 (student project)"}
TIMEOUT = 15
DELAY_SECONDS = 1.5

REMOTE_WORDS = ("remote", "worldwide", "anywhere", "global", "work from home", "wfh")

# ---------------------------------------------------------------------------
# Location aliases. Typing a city also matches its metro area and country.
# ---------------------------------------------------------------------------
PK_CITIES = ["islamabad", "rawalpindi", "lahore", "karachi", "peshawar",
             "faisalabad", "multan", "quetta", "gilgit", "abbottabad", "sialkot"]
LOCATION_ALIASES = {
    "pakistan": ["pakistan", "pk"] + PK_CITIES,
    "islamabad": ["islamabad", "rawalpindi", "pakistan"],
    "rawalpindi": ["rawalpindi", "islamabad", "pakistan"],
    "lahore": ["lahore", "punjab", "pakistan"],
    "karachi": ["karachi", "sindh", "pakistan"],
    "peshawar": ["peshawar", "pakistan"],
    "usa": ["usa", "us", "united states", "america"],
    "uk": ["uk", "united kingdom", "england", "london"],
    "uae": ["uae", "united arab emirates", "dubai", "abu dhabi"],
    "germany": ["germany", "berlin", "munich", "hamburg", "deutschland"],
    "russia": ["russia", "moscow", "saint petersburg", "st. petersburg"],
}

# Optional custom scraped sources (template: fix selectors, then enable).
SOURCES = [
    {
        "name": "Example Job Site", "category": "Jobs", "enabled": False,
        "url": "https://example.com/jobs?search={query}&where={location}",
        "item": "div.job-card", "title": "h2", "link": "a",
        "company": "span.company", "location": "span.location",
    },
]

# ---------------------------------------------------------------------------
# Skills used for CV matching. Add your own in the sidebar or extend this list.
# ---------------------------------------------------------------------------
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "matlab", "verilog", "vhdl",
    "sql", "mysql", "postgresql", "mongodb", "html", "css", "react", "node.js", "php",
    "django", "flask", "fastapi", "streamlit", "git", "github", "docker", "kubernetes",
    "linux", "aws", "azure", "gcp", "rest api", "api", "ci/cd",
    "machine learning", "deep learning", "data science", "data analysis", "data analytics",
    "data engineering", "nlp", "natural language processing", "computer vision", "ai",
    "artificial intelligence", "llm", "langchain", "crewai", "rag", "generative ai",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "opencv",
    "power bi", "tableau", "excel", "statistics", "big data", "spark", "hadoop",
    "fpga", "vivado", "embedded", "arduino", "raspberry pi", "iot", "robotics", "ros",
    "unity", "unreal engine", "game development", "web development", "frontend",
    "backend", "full stack", "android", "ios", "flutter", "ui/ux", "figma",
    "cybersecurity", "networking", "devops", "agile", "scrum", "testing", "qa",
    "project management", "communication", "research", "technical writing",
]
STOPWORDS = set("""the and for with you your our are will have this that from into out
intern internship junior senior job jobs role team work working remote full time part
engineer developer manager specialist associate lead""".split())

STATUSES = ["Saved", "Applied", "Interview", "Offer", "Rejected"]
TRACK_COLS = ["Title", "Organization", "Status", "Deadline", "Notes",
              "Location", "Source", "Match %", "Link"]


# ---------------------------------------------------------------------------
# Location logic
# ---------------------------------------------------------------------------
def expand_location(location):
    """Turn what the user typed into a list of lowercase search terms."""
    loc = location.strip().lower()
    if not loc:
        return []
    terms = set() if "," in loc else {loc}
    for part in re.split(r"[,/|]", loc):
        part = part.strip()
        if part:
            terms.add(part)
            terms.update(LOCATION_ALIASES.get(part, []))
    return sorted(terms)


def is_remote(*texts):
    blob = " ".join(t for t in texts if t).lower()
    return any(w in blob for w in REMOTE_WORDS)


def term_in(term, text):
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def apply_location_filter(df, terms, include_remote, include_unknown):
    if not terms or df.empty:
        return df

    def keep(row):
        loc = (row["Location"] or "").strip().lower()
        if not loc:
            return include_unknown
        if any(term_in(t, loc) for t in terms):
            return True
        return include_remote and bool(row["Remote"])

    return df[df.apply(keep, axis=1)]


# ---------------------------------------------------------------------------
# Job sources
# ---------------------------------------------------------------------------
def clean_html(text, limit=3000):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def make_row(source, category, title, company, location, link, extra="", remote=False, text="",
             work=None):
    return {
        "Category": category, "Title": title, "Organization": company,
        "Location": location, "Remote": remote,
        "Work type": work or ("Remote" if remote else "On-site"), "Details": extra,
        "Link": link, "Source": source, "Text": clean_html(text),
    }


def as_text(v):
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v or "")


def words_match(blob, keyword):
    """True if every word of keyword appears in blob."""
    blob = blob.lower()
    return all(w in blob for w in keyword.lower().split())


def fetch_jobicy(keyword, internship=False):
    """Jobicy public API: https://jobicy.com/api/v2/remote-jobs"""
    rows = []
    params = {"count": 100}
    if len(keyword.strip()) >= 3:  # API accepts 3-50 characters
        params["tag"] = keyword.strip()[:50]
    try:
        r = requests.get("https://jobicy.com/api/v2/remote-jobs",
                         params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        for j in r.json().get("jobs", []):
            title = j.get("jobTitle", "")
            body = j.get("jobDescription") or j.get("jobExcerpt") or ""
            blob = f"{title} {j.get('jobExcerpt','')} {as_text(j.get('jobLevel'))}"
            if internship and "intern" not in blob.lower():
                continue
            geo = as_text(j.get("jobGeo")) or "Remote"
            rows.append(make_row(
                "Jobicy", "Internships" if internship else "Jobs", title,
                j.get("companyName", ""), geo, j.get("url", ""),
                as_text(j.get("jobType")), remote=is_remote(geo), text=body, work="Remote"))
    except Exception as e:
        st.warning(f"Jobicy failed: {e}")
    return rows


@st.cache_data(ttl=600, show_spinner=False)
def _remoteok_feed():
    r = requests.get("https://remoteok.com/api", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return [j for j in r.json() if isinstance(j, dict) and j.get("position")]


def fetch_remoteok(keyword, internship=False):
    """Remote OK public API (credit: remoteok.com, link back is required)."""
    rows = []
    kw = f"{keyword} intern" if internship else keyword
    try:
        for j in _remoteok_feed():
            blob = f"{j.get('position','')} {j.get('company','')} {as_text(j.get('tags'))} {j.get('description','')}"
            if words_match(blob, kw):
                loc = j.get("location") or "Remote"
                rows.append(make_row(
                    "Remote OK", "Internships" if internship else "Jobs",
                    j.get("position", ""), j.get("company", ""), loc,
                    j.get("url", ""), as_text(j.get("tags")), remote=is_remote(loc),
                    text=f"{as_text(j.get('tags'))} {j.get('description','')}", work="Remote"))
    except Exception as e:
        st.warning(f"Remote OK failed: {e}")
    return rows


WWR_FEEDS = [
    "https://weworkremotely.com/remote-jobs.rss",
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
]


@st.cache_data(ttl=600, show_spinner=False)
def _wwr_feed(url):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        items.append({
            "title": it.findtext("title", ""), "link": it.findtext("link", ""),
            "region": it.findtext("region", ""), "desc": it.findtext("description", ""),
        })
    return items


def fetch_wwr(keyword, internship=False):
    """We Work Remotely public RSS feeds."""
    rows = []
    kw = f"{keyword} intern" if internship else keyword
    for feed in WWR_FEEDS:
        try:
            for it in _wwr_feed(feed):
                if not words_match(f"{it['title']} {it['desc']}", kw):
                    continue
                company, sep, role = it["title"].partition(": ")
                if not sep:
                    company, role = "", it["title"]
                loc = it["region"] or "Remote"
                rows.append(make_row(
                    "We Work Remotely", "Internships" if internship else "Jobs",
                    role, company, loc, it["link"], remote=is_remote(loc), text=it["desc"],
                    work="Remote"))
        except Exception as e:
            st.warning(f"We Work Remotely feed failed: {e}")
    return rows


@st.cache_data(ttl=600, show_spinner=False)
def _muse_page(level, page):
    params = {"page": page}
    if level:
        params["level"] = level
    r = requests.get("https://www.themuse.com/api/public/jobs",
                     params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_muse(keyword, internship=False):
    """The Muse public API (no key). Internship searches use its Internship level."""
    rows = []
    level = "Internship" if internship else None
    try:
        for page in range(5):
            for j in _muse_page(level, page):
                blob = f"{j.get('name','')} {j.get('contents','')}"
                if not words_match(blob, keyword):
                    continue
                loc = ", ".join(l.get("name", "") for l in j.get("locations", []))
                rows.append(make_row(
                    "The Muse", "Internships" if internship else "Jobs",
                    j.get("name", ""), (j.get("company") or {}).get("name", ""),
                    loc, (j.get("refs") or {}).get("landing_page", ""),
                    ", ".join(l.get("name", "") for l in j.get("levels", [])),
                    remote=is_remote(loc), text=j.get("contents", ""),
                    work="Remote" if (is_remote(loc) or "flexible" in loc.lower()) else "On-site"))
    except Exception as e:
        st.warning(f"The Muse failed: {e}")
    return rows


API_SOURCES = {
    "Jobicy": fetch_jobicy,
    "Remote OK": fetch_remoteok,
    "We Work Remotely": fetch_wwr,
    "The Muse": fetch_muse,
}
REMOTE_ONLY_SOURCES = {"Jobicy", "Remote OK", "We Work Remotely"}  # The Muse lists both
WORK_MODES = ["Remote + on-site", "Remote only", "On-site only"]


# ---------------------------------------------------------------------------
# Optional custom scraper (robots.txt aware)
# ---------------------------------------------------------------------------
def allowed_by_robots(url):
    parts = urlparse(url)
    rp = RobotFileParser()
    try:
        rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True


def text_of(parent, selector):
    if not selector:
        return ""
    el = parent.select_one(selector)
    return el.get_text(" ", strip=True) if el else ""


def scrape_source(src, keyword, location):
    rows = []
    url = src["url"].format(query=quote_plus(keyword), location=quote_plus(location))
    if not allowed_by_robots(url):
        st.warning(f"{src['name']}: blocked by robots.txt, skipped.")
        return rows
    try:
        time.sleep(DELAY_SECONDS)
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select(src["item"]):
            title = text_of(card, src["title"])
            a = card.select_one(src["link"]) if src.get("link") else None
            href = urljoin(url, a["href"]) if a and a.has_attr("href") else url
            loc = text_of(card, src.get("location"))
            if title:
                rows.append(make_row(src["name"], src["category"], title,
                                     text_of(card, src.get("company")), loc, href,
                                     remote=is_remote(loc, title),
                                     text=card.get_text(" ", strip=True)))
    except Exception as e:
        st.warning(f"{src['name']} failed: {e}")
    return rows


# ---------------------------------------------------------------------------
# Google search buttons (for sites that do not allow scraping)
# ---------------------------------------------------------------------------
GOOGLE_SITES = {
    "LinkedIn": "linkedin.com/jobs",
    "Indeed Pakistan": "pk.indeed.com",
    "Rozee.pk": "rozee.pk",
    "Mustakbil": "mustakbil.com",
    "Bayt": "bayt.com",
}
FRESHNESS = {"Any time": "", "Past 24 hours": "d", "Past week": "w", "Past month": "m"}


def google_location_clause(location, work_mode):
    """Build ("Rawalpindi" OR "Islamabad" OR "Pakistan" OR remote) for a Google query."""
    parts = [p.strip() for p in re.split(r"[,/|]", location) if p.strip()]
    names, seen = [], set()
    for i, p in enumerate(parts):
        aliases = LOCATION_ALIASES.get(p.lower(), []) if i == 0 else []  # expand only the first place
        for n in [p] + aliases:
            if len(n) > 2 and n.lower() not in seen:
                seen.add(n.lower())
                names.append(n)
    names = [f'"{n}"' for n in names[:4]]
    if work_mode == "Remote + on-site" and names:
        names.append("remote")
    elif work_mode == "Remote only":
        names += ["anywhere", "worldwide"]
    return f"({' OR '.join(names)})" if names else ""


def google_links(keyword, location, internship, work_mode, freshness):
    """Google searches limited to one job site each, plus Google's own Jobs panel."""
    role = f'"{keyword}"' + (" (intern OR internship)" if internship else "")
    if work_mode == "Remote only":
        role += " remote"
    elif work_mode == "On-site only":
        role += " -remote"
    loc = google_location_clause(location, work_mode)
    tbs = f"&tbs=qdr:{FRESHNESS[freshness]}" if FRESHNESS[freshness] else ""
    base = "https://www.google.com/search?q="
    word = "internship" if internship else "jobs"
    mode_word = {"Remote only": " remote", "On-site only": " -remote"}.get(work_mode, "")
    jobs_query = f"{keyword} {word} {location}{mode_word}".strip()
    links = {"Google Jobs": f"{base}{quote_plus(jobs_query)}&ibp=htl;jobs"}
    for name, site in GOOGLE_SITES.items():
        query = " ".join(x for x in (f"site:{site}", role, loc) if x)
        links[name] = f"{base}{quote_plus(query)}{tbs}"
    return links


# ---------------------------------------------------------------------------
# CV matching (local, no AI key needed)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_cv_file(name, data):
    if name.lower().endswith(".pdf"):
        try:
            from io import BytesIO
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(data)).pages)
        except Exception as e:
            return f"__ERROR__ {e}"
    return data.decode("utf-8", "ignore")


def build_skill_regex(extra_skills):
    extra = [s.strip().lower() for s in extra_skills.split(",") if s.strip()]
    skills = sorted(set(SKILLS) | set(extra), key=len, reverse=True)
    pattern = r"(?<![A-Za-z0-9])(" + "|".join(re.escape(s) for s in skills) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, re.I)


def find_skills(rx, text):
    return {m.lower() for m in rx.findall(text or "")}


def add_cv_match(df, cv_text, extra_skills):
    """Add 'Match %' and 'Matched skills' columns and sort best first."""
    rx = build_skill_regex(extra_skills)
    cv_skills = find_skills(rx, cv_text)
    cv_lower = cv_text.lower()
    pcts, shared = [], []
    for _, r in df.iterrows():
        job_text = f"{r['Title']} {r['Details']} {r['Text']}"
        job_skills = find_skills(rx, job_text)
        matched = job_skills & cv_skills
        words = [w for w in re.findall(r"[a-z]{3,}", str(r["Title"]).lower()) if w not in STOPWORDS]
        title_score = (sum(1 for w in words if w in cv_lower) / len(words)) if words else 0.0
        if job_skills and cv_skills:
            score = 0.7 * (len(matched) / len(job_skills)) + 0.3 * title_score
        else:
            score = title_score
        pcts.append(round(100 * score))
        shared.append(", ".join(sorted(matched)))
    df = df.copy()
    df["Match %"] = pcts
    df["Matched skills"] = shared
    return df.sort_values("Match %", ascending=False).reset_index(drop=True), cv_skills


# ---------------------------------------------------------------------------
# Messages shown after a button press (they survive st.rerun)
# ---------------------------------------------------------------------------
def flash(kind, msg):
    st.session_state.setdefault("flash", []).append((kind, msg))


def show_flash():
    for kind, msg in st.session_state.pop("flash", []):
        getattr(st, kind)(msg)


# ---------------------------------------------------------------------------
# Tracker (kept in your browser session, never on the server)
# ---------------------------------------------------------------------------
def empty_tracker():
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in TRACK_COLS})
    df["Deadline"] = pd.to_datetime(df["Deadline"])
    return df


def add_to_tracker(rows):
    t = st.session_state["tracker"]
    existing = set(t["Link"].dropna())
    new = []
    for _, r in rows.iterrows():
        if r["Link"] in existing:
            continue
        new.append({
            "Title": r["Title"], "Organization": r["Organization"], "Status": "Saved",
            "Deadline": pd.NaT, "Notes": "", "Location": r["Location"], "Source": r["Source"],
            "Match %": r.get("Match %"), "Link": r["Link"],
        })
    start = len(t)
    if new:
        t = pd.concat([t, pd.DataFrame(new)], ignore_index=True)
        t["Deadline"] = pd.to_datetime(t["Deadline"], errors="coerce")
        st.session_state["tracker"] = t
        st.session_state["tracker_ver"] += 1
    return list(range(start, start + len(new)))


def apply_tracker_edits(key):
    """Callback: merge the editor's changes into the stored tracker."""
    state = st.session_state.get(key, {})
    df = st.session_state["tracker"].copy().reset_index(drop=True)
    for idx, changes in state.get("edited_rows", {}).items():
        for col, val in changes.items():
            if col == "Deadline":
                val = pd.to_datetime(val, errors="coerce")
            df.at[int(idx), col] = val
    if state.get("deleted_rows"):
        df = df.drop(index=state["deleted_rows"])
    added = []
    for row in state.get("added_rows", []):
        new = {c: None for c in TRACK_COLS}
        new["Status"] = "Saved"
        new.update(row)
        added.append(new)
    if added:
        df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
    df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
    st.session_state["tracker"] = df.reset_index(drop=True)
    st.session_state["tracker_ver"] += 1


def render_tracker():
    t = st.session_state["tracker"].copy()
    t["Deadline"] = pd.to_datetime(t["Deadline"], errors="coerce")
    st.subheader("📋 My applications")
    st.caption("Saved only in this browser session. Download the CSV to keep it, "
               "and restore it below next time.")

    # Deadline reminders
    open_rows = t[t["Deadline"].notna() & ~t["Status"].isin(["Offer", "Rejected"])]
    if not open_rows.empty:
        days = (open_rows["Deadline"] - pd.Timestamp.today().normalize()).dt.days
        for _, r in open_rows[days < 0].iterrows():
            st.error(f"⏰ Deadline passed: {r['Title']} ({r['Organization']})")
        for (_, r), d in zip(open_rows[(days >= 0) & (days <= 7)].iterrows(), days[(days >= 0) & (days <= 7)]):
            st.warning(f"⏰ {d} day(s) left: {r['Title']} ({r['Organization']})")

    if t.empty:
        st.info("Nothing saved yet. Tick **Save** next to a result and press "
                "**Add selected to tracker**.")
    else:
        key = f"tracker_editor_{st.session_state['tracker_ver']}"
        st.data_editor(
            t[TRACK_COLS], key=key, num_rows="dynamic", hide_index=True,
            use_container_width=True, on_change=apply_tracker_edits, args=(key,),
            disabled=["Source", "Match %", "Link"],
            column_config={
                "Status": st.column_config.SelectboxColumn("Status", options=STATUSES, required=True),
                "Deadline": st.column_config.DateColumn("Deadline", format="DD MMM YYYY"),
                "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                "Match %": st.column_config.NumberColumn("Match %", format="%d%%"),
            },
        )
        counts = t["Status"].value_counts()
        st.caption(" · ".join(f"{s}: {int(counts.get(s, 0))}" for s in STATUSES))

        st.download_button("⬇️ Download tracker (CSV)", t[TRACK_COLS].to_csv(index=False).encode("utf-8"),
                           file_name="my_applications.csv", mime="text/csv")

    up = st.file_uploader("Restore a saved tracker (CSV)", type="csv", key="tracker_upload")
    if up is not None:
        sig = f"{up.name}-{up.size}"
        if st.session_state.get("tracker_loaded") != sig:
            try:
                df = pd.read_csv(up)
                for c in TRACK_COLS:
                    if c not in df.columns:
                        df[c] = None
                df = df[TRACK_COLS]
                df["Deadline"] = pd.to_datetime(df["Deadline"], errors="coerce")
                df["Notes"] = df["Notes"].fillna("")
                st.session_state["tracker"] = df
                st.session_state["tracker_ver"] += 1
                st.session_state["tracker_loaded"] = sig
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")


# ---------------------------------------------------------------------------
# Search tab
# ---------------------------------------------------------------------------
def run_search(p):
    keyword, internship = p["keyword"], p["category"] == "Internships"
    loc_terms = expand_location(p["location"])
    rows = []
    with st.spinner("Searching sources..."):
        for name in p["chosen"]:
            if p["work_mode"] == "On-site only" and name in REMOTE_ONLY_SOURCES:
                continue  # these boards only list remote jobs
            rows += API_SOURCES[name](keyword, internship)
        for src in p["custom"]:
            rows += scrape_source(src, f"{keyword} intern" if internship else keyword, p["location"])

    st.session_state["res_ver"] += 1
    if not rows:
        st.session_state["results"] = {"query": p, "df": pd.DataFrame(), "total": 0, "terms": loc_terms}
        return
    df = pd.DataFrame(rows).drop_duplicates(subset=["Title", "Organization", "Link"])
    total = len(df)
    df = apply_location_filter(df, loc_terms, p["include_remote"], p["include_unknown"])
    if p["work_mode"] == "Remote only":
        df = df[df["Work type"] == "Remote"]
    elif p["work_mode"] == "On-site only":
        df = df[df["Work type"] == "On-site"]
    df = df.reset_index(drop=True)
    st.session_state["results"] = {"query": p, "df": df, "total": total, "terms": loc_terms}


def render_results(cv_text, extra_skills):
    res = st.session_state.get("results")
    if res is None:
        st.info("Enter a keyword in the sidebar and press **Search**.")
        return
    p, df = res["query"], res["df"]
    internship = p["category"] == "Internships"

    with st.expander("🔍 Find exact jobs with Google (sites that don't allow scraping)", expanded=True):
        st.caption("Each button opens a Google search limited to one job site, using your "
                   "keyword, location and date filter.")
        cols = st.columns(3)
        links = google_links(p["keyword"], p["location"], internship, p["work_mode"], p["freshness"])
        for i, (name, url) in enumerate(links.items()):
            cols[i % 3].link_button(name, url, use_container_width=True)

    if df.empty:
        if res["total"]:
            st.warning("Nothing matched this location and work type. Try a country name, "
                       "tick 'include listings with no location', or switch the work type.")
        else:
            st.warning("No results from the job sources. Try another keyword, or use the Google buttons above."
                       + (" Only The Muse lists on-site jobs here, so the Google buttons are best for local roles."
                          if p["work_mode"] == "On-site only" else ""))
        return

    if res["terms"]:
        st.caption(f"Location terms used: {', '.join(res['terms'])} · "
                   f"{len(df)} of {res['total']} results match")

    cols_show = ["Save", "Title", "Organization", "Location", "Work type", "Source", "Link"]
    if cv_text.strip():
        df, cv_skills = add_cv_match(df, cv_text, extra_skills)
        cols_show = ["Save", "Match %", "Matched skills"] + cols_show[1:]
        st.caption("CV skills detected: " + (", ".join(sorted(cv_skills)) or
                   "none (matching on title words only; add skills in the sidebar)"))

    view = df.copy()
    view.insert(0, "Save", False)
    st.success(f"Found {len(df)} results")
    edited = st.data_editor(
        view[cols_show], key=f"res_editor_{st.session_state['res_ver']}", hide_index=True,
        use_container_width=True, disabled=[c for c in cols_show if c != "Save"],
        column_config={
            "Save": st.column_config.CheckboxColumn("Save"),
            "Match %": st.column_config.ProgressColumn("Match %", min_value=0, max_value=100, format="%d%%"),
            "Link": st.column_config.LinkColumn("Link", display_text="Open"),
        },
    )
    c1, c2 = st.columns([1, 1])
    if c1.button("➕ Add selected to tracker"):
        chosen = df.loc[edited.index[edited["Save"]]]
        if chosen.empty:
            flash("info", "Tick Save next to at least one result first.")
        else:
            new_idx = add_to_tracker(chosen)
            st.session_state["res_ver"] += 1
            if not new_idx:
                flash("info", "Those jobs are already in your tracker.")
            else:
                flash("success", f"Added {len(new_idx)} job(s) to your tracker.")
        st.rerun()
    c2.download_button("⬇️ Download results (CSV)",
                       df.drop(columns=["Text"]).to_csv(index=False).encode("utf-8"),
                       file_name=f"{p['category'].lower()}_{p['keyword'].replace(' ', '_')}.csv",
                       mime="text/csv")
    st.caption("Listings provided by [Jobicy](https://jobicy.com), "
               "[Remote OK](https://remoteok.com), "
               "[We Work Remotely](https://weworkremotely.com) and "
               "[The Muse](https://www.themuse.com).")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Opportunity Finder", page_icon="🔎", layout="wide")
    st.session_state.setdefault("tracker", empty_tracker())
    st.session_state.setdefault("tracker_ver", 0)
    st.session_state.setdefault("res_ver", 0)

    st.title("🔎 Opportunity Finder")
    st.caption("Jobs and internships from multiple sources, matched to your CV and tracked in one place.")

    with st.sidebar:
        st.header("Search")
        keyword = st.text_input("Keyword", placeholder="e.g. machine learning")
        category = st.selectbox("Type", ["Jobs", "Internships"])
        chosen = list(API_SOURCES)  # all sources are always searched

        st.subheader("Location")
        location = st.text_input("Where?", placeholder="e.g. Rawalpindi, Pakistan")
        work_mode = st.radio("Work type", WORK_MODES, horizontal=False,
                             help="Jobicy, Remote OK and We Work Remotely list remote jobs only. "
                                  "The Muse lists both remote and on-site jobs.")
        include_unknown = st.checkbox("Also include listings with no location", value=False)
        freshness = st.selectbox("Google links: posted within", list(FRESHNESS), index=2)

        with st.expander("📄 CV matching (optional)"):
            cv_file = st.file_uploader("Upload CV (PDF or TXT)", type=["pdf", "txt"])
            cv_paste = st.text_area("...or paste your CV text", height=120)
            extra_skills = st.text_input("Extra skills to look for", placeholder="comma separated")
            st.caption("Your CV is read in memory for this session only and is not saved.")

        enabled_custom = [s for s in SOURCES if s["enabled"]]
        go = st.button("Search", type="primary", use_container_width=True)

    cv_text = cv_paste or ""
    if cv_file is not None:
        text = read_cv_file(cv_file.name, cv_file.getvalue())
        if text.startswith("__ERROR__"):
            st.sidebar.warning("Could not read that PDF (is pypdf installed?). Paste the text instead.")
        else:
            cv_text = f"{cv_text}\n{text}"

    show_flash()
    tab_search, tab_tracker = st.tabs(["🔎 Search", "📋 My tracker"])

    with tab_search:
        if go:
            if not keyword.strip():
                st.error("Please enter a keyword.")
            else:
                run_search({
                    "keyword": keyword.strip(), "category": category, "chosen": chosen,
                    "location": location, "work_mode": work_mode,
                    "include_remote": work_mode != "On-site only",
                    "include_unknown": include_unknown, "freshness": freshness,
                    "custom": enabled_custom,
                })
        render_results(cv_text, extra_skills)

    with tab_tracker:
        render_tracker()


if __name__ == "__main__":
    main()
