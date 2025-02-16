from scraper import scrape_multiple_sources
from train_model import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # A comprehensive list of URLs from open source, technology, and full Wikipedia articles.
    urls = [
        "https://opensource.com/article/21/6/what-open-source",
        "https://opensource.com/resources",
        "https://www.linuxfoundation.org/",
        "https://www.apache.org/",
        "https://www.eclipse.org/",
        "https://opensource.guide/",
        "https://www.fsf.org/",
        "https://en.wikipedia.org/wiki/Open-source",
        "https://en.wikipedia.org/wiki/Free_software",
        "https://en.wikipedia.org/wiki/Software_license",
        "https://en.wikipedia.org/wiki/Computer_programming",
        "https://en.wikipedia.org/wiki/Wikipedia"  # Full Wikipedia page overview
    ]
    
    data_file = "data.txt"
    logger.info("Starting enhanced scraping process...")
    combined_text = scrape_multiple_sources(urls, max_workers=10)
    
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    logger.info(f"Scraping completed. Data saved to {data_file}")

    logger.info("Starting enhanced training process...")
    # Increase epochs and use more data for improved training and interactive responses.
    train_model(data_file, epochs=10, batch_size=4)
    logger.info("Training completed. You can now interact with your chatbot using chatbot.py.")

if __name__ == "__main__":
    main()
