import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import logging

# Initialize the summarization pipeline and sentiment analyzer
summarizer = pipeline("summarization")
analyzer = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def scrape_website(url):
    logging.debug(f"Scraping website: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting content
        title = soup.title.string if soup.title else "No title found"
        headings = [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])]
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
        links = [a['href'] for a in soup.find_all('a', href=True)]

        # Concatenate all text for analysis
        all_text = ' '.join(headings + paragraphs)
        if len(all_text) > 1024:
            all_text = all_text[:1024]  # Summarization works better on smaller inputs
        summary = summarizer(all_text, max_length=min(len(all_text)//2, 100), min_length=30, do_sample=False)[0]['summary_text']

        # Perform sentiment analysis using TextBlob and VADER
        blob = TextBlob(all_text)
        vader_sentiment = analyzer.polarity_scores(all_text)

        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "positivity": vader_sentiment['pos'],
            "negativity": vader_sentiment['neg'],
            "neutrality": vader_sentiment['neu'],
            "compound": vader_sentiment['compound']
        }

        # Generate word cloud
        wordcloud = generate_wordcloud(all_text)

        scraped_data = {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "summary": summary,
            "word_count": len(all_text.split()),
            "sentiment": sentiment,
            "wordcloud": wordcloud
        }
        logging.info(f"Scraping completed for {url}")
        return scraped_data
    except requests.RequestException as e:
        logging.error(f"Request error while scraping {url}: {e}")
        return {"error": f"Request error: {e}"}
    except Exception as e:
        logging.error(f"Error while scraping {url}: {e}")
        return {"error": f"An error occurred: {e}"}

def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        logging.info("Word cloud generated successfully")
        return img_base64
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        return {"error": f"Error generating word cloud: {e}"}

