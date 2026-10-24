"""
Opportunity Finder v3: jobs and internships in one search.

Sources (official public APIs / RSS feeds, no scraping of blocked sites)
  - Jobicy            https://jobicy.com            (public API)
  - Remote OK         https://remoteok.com          (public API, link back required)
  - We Work Remotely  https://weworkremotely.com    (public RSS feed)
  - The Muse          https://www.themuse.com       (public API, has an Internship level)
  Plus one-click search links for LinkedIn, Indeed Pakistan and Rozee.pk
  (these sites do not allow scraping, so the app opens them for you instead).

Optional Groq AI: expands your keyword and location, and ranks results by fit.

Install:   pip install streamlit requests beautifulsoup4 pandas
Run:       streamlit run opportunity_finder.py

Groq key (never hard-code it or upload it to GitHub). Use ONE of:
  - Streamlit secrets / .streamlit/secrets.toml:  GROQ_API_KEY = "gsk_..."
  - environment variable GROQ_API_KEY
  - the sidebar box (shown only when no key is found above)
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, quote_plus, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "OpportunityFinderBot/3.0 (student project)"}
TIMEOUT = 15
DELAY_SECONDS = 1.5

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # current Groq production models

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
    """Call Groq and return the parsed JSON object (raises with Groq's message on failure)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_completion_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=40,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])


def ai_expand_keywords(api_key, model, keyword, category):
    try:
        data = groq_json(
            api_key, model,
            'You help students search for opportunities. Reply ONLY with JSON: '
            '{"keywords": ["..."]} containing up to 3 short alternative search '
            'phrases (synonyms or closely related roles).',
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
    """Score rows of df from 0-10 for fit with the user's profile."""
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
# Location logic
# ---------------------------------------------------------------------------
def expand_location(location, ai_terms=None):
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
# Job sources
# ---------------------------------------------------------------------------
def make_row(source, category, title, company, location, link, extra="", remote=False):
    return {
        "Category": category, "Title": title, "Organization": company,
        "Location": location, "Remote": remote, "Details": extra,
        "Link": link, "Source": source,
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
            blob = f"{title} {j.get('jobExcerpt','')} {as_text(j.get('jobLevel'))}"
            if internship and "intern" not in blob.lower():
                continue
            geo = as_text(j.get("jobGeo")) or "Remote"
            rows.append(make_row(
                "Jobicy", "Internships" if internship else "Jobs", title,
                j.get("companyName", ""), geo, j.get("url", ""),
                as_text(j.get("jobType")), remote=is_remote(geo)))
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
                    j.get("url", ""), as_text(j.get("tags")), remote=is_remote(loc)))
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
                    role, company, loc, it["link"], remote=is_remote(loc)))
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
                    remote=is_remote(loc)))
    except Exception as e:
        st.warning(f"The Muse failed: {e}")
    return rows


API_SOURCES = {
    "Jobicy": fetch_jobicy,
    "Remote OK": fetch_remoteok,
    "We Work Remotely": fetch_wwr,
    "The Muse": fetch_muse,
}


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
                                     remote=is_remote(loc, title)))
    except Exception as e:
        st.warning(f"{src['name']} failed: {e}")
    return rows


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def direct_links(keyword, location, internship):
    q = f"{keyword} intern" if internship else keyword
    loc = location.strip() or "Pakistan"
    return {
        "LinkedIn": f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(q)}&location={quote_plus(loc)}",
        "Indeed Pakistan": f"https://pk.indeed.com/jobs?q={quote_plus(q)}&l={quote_plus(loc)}",
        "Rozee.pk": f"https://www.rozee.pk/job/jsearch/q/{quote(q)}",
    }


def main():
    st.set_page_config(page_title="Opportunity Finder", page_icon="🔎", layout="wide")
    st.title("🔎 Opportunity Finder")
    st.caption("Jobs and internships from multiple sources, with optional Groq AI.")

    with st.sidebar:
        st.header("Search")
        keyword = st.text_input("Keyword", placeholder="e.g. machine learning")
        category = st.selectbox("Type", ["Jobs", "Internships"])
        chosen = st.multiselect("Job sources", list(API_SOURCES), default=list(API_SOURCES))

        st.subheader("Location")
        location = st.text_input("Where?", placeholder="e.g. Rawalpindi, Pakistan")
        include_remote = st.checkbox("Also include remote / worldwide", value=True)
        include_unknown = st.checkbox("Also include listings with no location", value=False)

        st.subheader("Groq AI (optional)")
        api_key = get_groq_key("")
        if api_key:
            st.caption("✅ Groq key loaded from secrets")
        else:
            api_key = get_groq_key(st.text_input("Groq API key", type="password"))
        with st.expander("Advanced"):
            model = st.selectbox("Groq model", MODELS)
        use_ai = st.checkbox("Use AI", value=bool(api_key), disabled=not api_key)
        ai_keywords = st.checkbox("Expand keyword with AI", value=True, disabled=not use_ai)
        ai_location = st.checkbox("Expand location with AI", value=True, disabled=not use_ai)
        ai_sort = st.checkbox("Rank results by fit", value=True, disabled=not use_ai)
        profile = st.text_area("Your profile / interests (for ranking)", height=100,
                               placeholder="e.g. Computer engineering graduate, Python, ML, "
                                           "looking for AI internships",
                               disabled=not use_ai)

        enabled_custom = [s for s in SOURCES if s["enabled"]]
        go = st.button("Search", type="primary", use_container_width=True)

    if not go:
        st.info("Enter a keyword and press **Search**.")
        return
    if not keyword.strip():
        st.error("Please enter a keyword.")
        return

    internship = category == "Internships"

    with st.expander("🔗 Also search directly on these sites", expanded=False):
        st.caption("These sites do not allow scraping, so they open in a new tab.")
        cols = st.columns(3)
        for col, (name, url) in zip(cols, direct_links(keyword, location, internship).items()):
            col.link_button(name, url, use_container_width=True)

    queries = [keyword]
    if use_ai and ai_keywords:
        queries += ai_expand_keywords(api_key, model, keyword, category)
    queries = list(dict.fromkeys(queries))

    extra_terms = ai_expand_location(api_key, model, location) if (use_ai and ai_location and location.strip()) else []
    loc_terms = expand_location(location, extra_terms)

    rows = []
    with st.spinner("Searching sources..."):
        for q in queries:
            for name in chosen:
                rows += API_SOURCES[name](q, internship)
            for src in enabled_custom:
                rows += scrape_source(src, f"{q} intern" if internship else q, location)

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
    st.caption("Listings provided by [Jobicy](https://jobicy.com), "
               "[Remote OK](https://remoteok.com), "
               "[We Work Remotely](https://weworkremotely.com) and "
               "[The Muse](https://www.themuse.com).")


if __name__ == "__main__":
    main()
