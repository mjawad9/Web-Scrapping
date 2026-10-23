import time
from urllib.parse import urljoin, urlparse, quote_plus
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "OpportunityFinderBot/1.0 (student project)"}
TIMEOUT = 15
DELAY_SECONDS = 1.5  # pause between requests to avoid hammering servers

SOURCES = [
    {
        "name": "Example Job Site",
        "category": "Jobs",
        "enabled": False,  # set True after you fix the selectors
        "url": "https://example.com/jobs?search={query}",
        "item": "div.job-card",          # one result block
        "title": "h2",                    # title inside the block
        "link": "a",                      # link inside the block
        "company": "span.company",        # optional
        "location": "span.location",      # optional
    },
    {
        "name": "Example Admissions Page",
        "category": "Admissions",
        "enabled": False,
        "url": "https://example.edu/admissions/programs?q={query}",
        "item": "li.program",
        "title": "a",
        "link": "a",
        "company": None,
        "location": None,
    },
]


def allowed_by_robots(url: str) -> bool:
    """Return True if robots.txt allows fetching this URL."""
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True  # robots.txt unreachable: proceed carefully


def text_of(parent, selector):
    if not selector:
        return ""
    el = parent.select_one(selector)
    return el.get_text(" ", strip=True) if el else ""


def make_row(source, category, title, company, location, link, extra=""):
    return {
        "Category": category,
        "Title": title,
        "Organization": company,
        "Location": location,
        "Details": extra,
        "Link": link,
        "Source": source,
    }



def fetch_remotive(keyword: str):
    rows = []
    try:
        r = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": keyword},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for j in r.json().get("jobs", []):
            rows.append(
                make_row(
                    "Remotive", "Jobs", j.get("title", ""), j.get("company_name", ""),
                    j.get("candidate_required_location", "Remote"), j.get("url", ""),
                    j.get("job_type", ""),
                )
            )
    except Exception as e:
        st.warning(f"Remotive failed: {e}")
    return rows


def fetch_arbeitnow(keyword: str):
    rows = []
    kw = keyword.lower()
    try:
        r = requests.get(
            "https://www.arbeitnow.com/api/job-board-api", headers=HEADERS, timeout=TIMEOUT
        )
        r.raise_for_status()
        for j in r.json().get("data", []):
            blob = f"{j.get('title','')} {j.get('description','')} {' '.join(j.get('tags', []))}".lower()
            if kw in blob:
                rows.append(
                    make_row(
                        "Arbeitnow", "Jobs", j.get("title", ""), j.get("company_name", ""),
                        j.get("location", ""), j.get("url", ""),
                        ", ".join(j.get("job_types", [])),
                    )
                )
    except Exception as e:
        st.warning(f"Arbeitnow failed: {e}")
    return rows



def scrape_source(src: dict, keyword: str):
    rows = []
    url = src["url"].format(query=quote_plus(keyword))

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
            if title:
                rows.append(
                    make_row(
                        src["name"], src["category"], title,
                        text_of(card, src.get("company")),
                        text_of(card, src.get("location")), href,
                    )
                )
    except Exception as e:
        st.warning(f"{src['name']} failed: {e}")
    return rows



def main():
    st.set_page_config(page_title="Opportunity Finder", page_icon="🔎", layout="wide")
    st.title("🔎 Opportunity Finder")
    st.caption("Jobs, internships and admissions from multiple sources in one search.")

    with st.sidebar:
        st.header("Search")
        keyword = st.text_input("Keyword", placeholder="e.g. machine learning")
        category = st.selectbox("Type", ["Jobs", "Internships", "Admissions"])
        location_filter = st.text_input("Location contains (optional)", placeholder="e.g. Remote, Pakistan")
        use_apis = st.checkbox("Use API sources (Remotive, Arbeitnow)", value=True)
        enabled_custom = [s for s in SOURCES if s["enabled"]]
        st.write(f"Custom scraped sources enabled: **{len(enabled_custom)}**")
        go = st.button("Search", type="primary", use_container_width=True)

    if not go:
        st.info("Enter a keyword in the sidebar and press **Search**.")
        return
    if not keyword.strip():
        st.error("Please enter a keyword.")
        return

    # For internships, search "intern" alongside the keyword
    query = f"{keyword} intern" if category == "Internships" else keyword

    rows = []
    with st.spinner("Searching sources..."):
        if use_apis and category in ("Jobs", "Internships"):
            rows += fetch_remotive(query)
            rows += fetch_arbeitnow(query)
        for src in enabled_custom:
            if src["category"] == category or (category == "Internships" and src["category"] == "Jobs"):
                rows += scrape_source(src, query)

    if not rows:
        st.warning("No results. Try another keyword or enable more sources in SOURCES.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["Title", "Organization", "Link"])
    if location_filter.strip():
        df = df[df["Location"].str.contains(location_filter, case=False, na=False)]

    st.success(f"Found {len(df)} results")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")},
    )
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{category.lower()}_{keyword.replace(' ', '_')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
