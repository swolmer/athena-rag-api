import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.plasticsurgery.org/cosmetic-procedures"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

def get_asps_procedure_links():
    try:
        print("🔍 Scraping ASPS main page for subpage links...")
        response = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a", href=True)
        
        # Look for links that contain cosmetic-procedures
        cosmetic_links = []
        for a in links:
            href = a['href'].strip()
            if 'cosmetic-procedures' in href and href != "/cosmetic-procedures":
                cosmetic_links.append(href)
        
        # Convert relative URLs to absolute URLs and filter valid procedure links
        sub_urls = set()
        for href in cosmetic_links:
            if href.startswith('/cosmetic-procedures/'):
                # Relative URL
                full_url = urljoin(BASE_URL, href)
                sub_urls.add(full_url)
            elif href.startswith('https://www.plasticsurgery.org/cosmetic-procedures/'):
                # Already absolute URL
                sub_urls.add(href)
        
        sorted_links = sorted(sub_urls)
        print(f"✅ Found {len(sorted_links)} subpage links.")
        return sorted_links

    except Exception as e:
        print(f"❌ Failed to retrieve ASPS links: {e}")
        return []

if __name__ == "__main__":
    urls = get_asps_procedure_links()
    for url in urls:
        print(url)
