"""
Opportunity Finder v2: jobs and internships in one search.

New in v2
  * Groq AI (optional): expands your keyword, expands your location
    (nearby cities / country names) and ranks results against your profile.
  * Fixed location search: aliases (Rawalpindi -> Islamabad, Pakistan ...),
    whole-word matching, a "Remote" column, and separate switches for
    "include remote jobs" and "include unknown locations".

Install:   pip install streamlit requests beautifulsoup4 pandas
Run:       streamlit run opportunity_finder.py

Groq key (never hard-code it in the file or upload it to GitHub). Use ONE of:
  - environment variable:   set GROQ_API_KEY=gsk_...        (Windows)
                            export GROQ_API_KEY=gsk_...     (Linux/Mac)
  - .streamlit/secrets.toml:  GROQ_API_KEY = "gsk_..."
  - paste it into the sidebar box while the app is running
"""

import json
import os
import re
import time
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "OpportunityFinderBot/2.0 (student project)"}
TIMEOUT = 15
DELAY_SECONDS = 1.5

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # change in the sidebar if Groq retires it

REMOTE_WORDS = ("remote", "worldwide", "anywhere", "global", "work from home", "wfh")

# ---------------------------------------------------------------------------
# Location aliases. Typing a city also matches its metro area and country.
# Add your own lines here.
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

# ---------------------------------------------------------------------------
# Optional custom scraped sources (templates: fix selectors, then enable).
# {query} and {location} are replaced in the url.
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "Example Job Site", "category": "Jobs", "enabled": False,
        "url": "https://example.com/jobs?search={query}&where={location}",
        "item": "div.job-card", "title": "h2", "link": "a",
        "company": "span.company", "location": "span.location",
    },
]


# ---------------------------------------------------------------------------
# Groq helpers
# ---------------------------------------------------------------------------
def get_groq_key(sidebar_value: str) -> str:
    if sidebar_value.strip():
        return sidebar_value.strip()
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def groq_json(api_key, model, system, user):
    """Call Groq and return the parsed JSON object (raises on failure)."""
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=40,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def ai_expand_keywords(api_key, model, keyword, category):
    try:
        data = groq_json(
            api_key, model,
            'You help students search for opportunities. Reply ONLY with JSON: '
            '{"keywords": ["..."]} containing up to 3 short alternative search '
            'phrases (synonyms or closely related roles/programs).',
            f"Category: {category}\nKeyword: {keyword}",
        )
        return [k.strip() for k in data.get("keywords", []) if isinstance(k, str)][:3]
    except Exception as e:
        st.warning(f"Groq keyword expansion failed: {e}")
        return []


def ai_expand_location(api_key, model, location):
    try:
        data = groq_json(
            api_key, model,
            'Reply ONLY with JSON: {"terms": ["..."]}. List up to 8 lowercase place '
            "names that a job or internship listing might use for the given location: "
            "the place itself, its metro area/nearby cities, its province/state and its country.",
            f"Location: {location}",
        )
        return [t.strip().lower() for t in data.get("terms", []) if isinstance(t, str)]
    except Exception as e:
        st.warning(f"Groq location expansion failed: {e}")
        return []


def ai_rank(api_key, model, df, profile, category, keyword):
    """Score the first rows of df from 0-10 for fit with the user's profile."""
    items = [
        {"i": int(i), "title": r.Title, "org": r.Organization, "location": r.Location}
        for i, r in df.iterrows()
    ]
    try:
        data = groq_json(
            api_key, model,
            'Reply ONLY with JSON: {"results": [{"i": 0, "score": 0-10, "reason": "max 12 words"}]}. '
            "Score how well each listing fits the person's profile and search.",
            f"Category: {category}\nSearch: {keyword}\nProfile: {profile or 'not given'}\n"
            f"Listings: {json.dumps(items)}",
        )
        scores = {int(x["i"]): x for x in data.get("results", []) if "i" in x}
        df["AI Score"] = [scores.get(int(i), {}).get("score") for i in df.index]
        df["Why"] = [scores.get(int(i), {}).get("reason", "") for i in df.index]
    except Exception as e:
        st.warning(f"Groq ranking failed: {e}")
    return df


# ---------------------------------------------------------------------------
# Location logic (the fix)
# ---------------------------------------------------------------------------
def expand_location(location, ai_terms=None):
    """Turn what the user typed into a list of lowercase search terms."""
    loc = location.strip().lower()
    if not loc:
        return []
    terms = {loc}
    for part in re.split(r"[,/|]", loc):
        part = part.strip()
        if part:
            terms.add(part)
            terms.update(LOCATION_ALIASES.get(part, []))
    terms.update(ai_terms or [])
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
# Sources
# ---------------------------------------------------------------------------
def make_row(source, category, title, company, location, link, extra="", remote=False):
    return {
        "Category": category, "Title": title, "Organization": company,
        "Location": location, "Remote": remote, "Details": extra,
        "Link": link, "Source": source,
    }


def fetch_remotive(keyword):
    rows = []
    try:
        r = requests.get("https://remotive.com/api/remote-jobs",
                         params={"search": keyword}, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        for j in r.json().get("jobs", []):
            rows.append(make_row(
                "Remotive", "Jobs", j.get("title", ""), j.get("company_name", ""),
                j.get("candidate_required_location", ""), j.get("url", ""),
                j.get("job_type", ""),
                remote=is_remote(j.get("candidate_required_location", ""))))
    except Exception as e:
        st.warning(f"Remotive failed: {e}")
    return rows


@st.cache_data(ttl=600, show_spinner=False)
def _arbeitnow_board():
    r = requests.get("https://www.arbeitnow.com/api/job-board-api",
                     headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("data", [])


def fetch_arbeitnow(keyword):
    rows, kw = [], keyword.lower()
    try:
        for j in _arbeitnow_board():
            blob = f"{j.get('title','')} {j.get('description','')} {' '.join(j.get('tags', []))}".lower()
            if kw in blob:
                rows.append(make_row(
                    "Arbeitnow", "Jobs", j.get("title", ""), j.get("company_name", ""),
                    j.get("location", ""), j.get("url", ""),
                    ", ".join(j.get("job_types", [])),
                    remote=is_remote(j.get("location", ""))))
    except Exception as e:
        st.warning(f"Arbeitnow failed: {e}")
    return rows


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
                                     remote=is_remote(loc, title)))
    except Exception as e:
        st.warning(f"{src['name']} failed: {e}")
    return rows


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Opportunity Finder", page_icon="🔎", layout="wide")
    st.title("🔎 Opportunity Finder")
    st.caption("Jobs and internships from multiple sources, with optional Groq AI.")

    with st.sidebar:
        st.header("Search")
        keyword = st.text_input("Keyword", placeholder="e.g. machine learning")
        category = st.selectbox("Type", ["Jobs", "Internships"])

        st.subheader("Location")
        location = st.text_input("Where?", placeholder="e.g. Rawalpindi, Pakistan")
        include_remote = st.checkbox("Also include remote / worldwide", value=True)
        include_unknown = st.checkbox("Also include listings with no location", value=False)

        st.subheader("Groq AI (optional)")
        key_input = st.text_input("Groq API key", type="password",
                                  help="Or set GROQ_API_KEY as env variable / Streamlit secret.")
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_key = get_groq_key(key_input)
        use_ai = st.checkbox("Use AI", value=bool(api_key), disabled=not api_key)
        ai_keywords = st.checkbox("Expand keyword with AI", value=True, disabled=not use_ai)
        ai_location = st.checkbox("Expand location with AI", value=True, disabled=not use_ai)
        ai_sort = st.checkbox("Rank results by fit", value=True, disabled=not use_ai)
        profile = st.text_area("Your profile / interests (for ranking)", height=100,
                               placeholder="e.g. Computer engineering graduate, Python, ML, "
                                           "looking for AI internships",
                               disabled=not use_ai)

        use_apis = st.checkbox("Use API sources (Remotive, Arbeitnow)", value=True)
        enabled_custom = [s for s in SOURCES if s["enabled"]]
        st.write(f"Custom scraped sources enabled: **{len(enabled_custom)}**")
        go = st.button("Search", type="primary", use_container_width=True)

    if not go:
        st.info("Enter a keyword and press **Search**.")
        return
    if not keyword.strip():
        st.error("Please enter a keyword.")
        return

    # keywords to search
    base = f"{keyword} intern" if category == "Internships" else keyword
    queries = [base]
    if use_ai and ai_keywords:
        queries += ai_expand_keywords(api_key, model, keyword, category)
    queries = list(dict.fromkeys(queries))

    # location terms
    extra_terms = ai_expand_location(api_key, model, location) if (use_ai and ai_location and location.strip()) else []
    loc_terms = expand_location(location, extra_terms)

    rows = []
    with st.spinner("Searching sources..."):
        for q in queries:
            if use_apis and category in ("Jobs", "Internships"):
                rows += fetch_remotive(q)
                rows += fetch_arbeitnow(q)
            for src in enabled_custom:
                if src["category"] == category or (category == "Internships" and src["category"] == "Jobs"):
                    rows += scrape_source(src, q, location)

    if not rows:
        st.warning("No results. Try another keyword or enable more sources.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["Title", "Organization", "Link"])
    total = len(df)
    df = apply_location_filter(df, loc_terms, include_remote, include_unknown).reset_index(drop=True)

    if loc_terms:
        st.caption(f"Location terms used: {', '.join(loc_terms)} · "
                   f"{len(df)} of {total} results match")
    if queries[1:]:
        st.caption(f"Keywords searched: {', '.join(queries)}")

    if df.empty:
        st.warning("Nothing matched this location. Try a country name, tick "
                   "'include remote', or tick 'include listings with no location'.")
        return

    if use_ai and ai_sort:
        with st.spinner("Ranking with Groq..."):
            top = ai_rank(api_key, model, df.head(30).copy(), profile, category, keyword)
            df = pd.concat([top, df.iloc[30:]], ignore_index=True)
            if "AI Score" in df.columns:
                df = df.sort_values("AI Score", ascending=False, na_position="last")

    st.success(f"Found {len(df)} results")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")})
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{category.lower()}_{keyword.replace(' ', '_')}.csv",
                       mime="text/csv")


if __name__ == "__main__":
    main()
