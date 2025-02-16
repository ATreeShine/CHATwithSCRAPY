# Enhanced Open Source Chatbot with Full Wikipedia Sources

This project demonstrates a pipeline that:
- Scrapes data from multiple sources including open source sites and full Wikipedia articles.
- Fine-tunes the GPT-2 model using the scraped data.
- Provides an interactive chatbot that handles greetings and common interaction patterns.

## Files

- **requirements.txt**  
  List of dependencies to install.
  
- **scraper.py**  
  A multi-threaded scraper that collects text content from a list of URLs.

- **train_model.py**  
  A training script that fine-tunes GPT-2 with enhanced training parameters.

- **chatbot.py**  
  An interactive chatbot that includes custom responses for greetings and feeling inquiries, in addition to GPT-2 generated responses.

- **main.py**  
  Orchestrates the scraping and training process, including expanded Wikipedia sources.

## Setup and Run

1. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
2. Run the program

3. Learn, Train, Use.
   
