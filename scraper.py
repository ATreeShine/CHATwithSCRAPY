import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_data(url):
    """
    Scrape visible text content from the provided URL.
    Extracts paragraphs, headers, and list items.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')

        tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text = "\n".join(tag.get_text(strip=True) for tag in tags)
        logger.info(f"Scraped {len(text)} characters from {url}")
        return text
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return ""

def scrape_multiple_sources(urls, max_workers=5):
    """
    Scrape multiple URLs concurrently using a thread pool.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_data, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"Error in future for {url}: {e}")
    return "\n".join(results)

if __name__ == "__main__":
    urls = [
        "https://opensource.com/article/21/6/what-open-source",
        "https://opensource.com/resources",
        "https://www.linuxfoundation.org/",
        "https://www.apache.org/",
        "https://www.eclipse.org/"
    ]
    combined_data = scrape_multiple_sources(urls)
    with open("data.txt", "w", encoding="utf-8") as f:
        f.write(combined_data)
    logger.info("Scraping completed and data saved to data.txt")
