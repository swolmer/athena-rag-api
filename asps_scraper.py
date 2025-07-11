import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Constants ---
BASE_URL = "https://www.plasticsurgery.org/cosmetic-procedures"
SAVE_DIR = os.path.join("org_data", "asps", "html_pages")
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

# --- Step 1: Scrape links from main page ---
def get_asps_procedure_links():
    try:
        print("🔍 Scraping ASPS main page for subpage links...")
        response = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a", href=True)
        cosmetic_links = [
            a["href"].strip() for a in links
            if "cosmetic-procedures" in a["href"]
            and a["href"] != "/cosmetic-procedures"
        ]

        sub_urls = set()
        for href in cosmetic_links:
            if href.startswith("/cosmetic-procedures/"):
                sub_urls.add(urljoin(BASE_URL, href))
            elif href.startswith("https://www.plasticsurgery.org/cosmetic-procedures/"):
                sub_urls.add(href)

        sorted_links = sorted(sub_urls)
        print(f"✅ Found {len(sorted_links)} subpage links.")
        return sorted_links

    except Exception as e:
        print(f"❌ Failed to retrieve ASPS links: {e}")
        return []

# --- Step 2: Download each page to HTML ---
def download_all_subpages(subpage_urls):
    for url in subpage_urls:
        slug = url.strip("/").split("/")[-1]
        filename = os.path.join(SAVE_DIR, f"{slug}.html")

        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"✅ Saved {slug}.html")
        except Exception as e:
            print(f"⚠️ Failed to download {url}: {e}")

# --- Entrypoint ---
if __name__ == "__main__":
    print("📥 Starting ASPS scraper...")
    links = get_asps_procedure_links()
    if links:
        download_all_subpages(links)
    else:
        print("❌ No subpages to download.")
