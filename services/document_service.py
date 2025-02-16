import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from PyPDF2 import PdfReader
import pandas as pd
import docx
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import io
import base64
import json
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract
import pydicom
import logging

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize the summarization pipeline and sentiment analyzer
summarizer = pipeline("summarization")
analyzer = SentimentIntensityAnalyzer()

# Maximum input length for the summarization model
MAX_INPUT_LENGTH = 1024

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def read_document(document, is_prescription=False):
    filename = document.filename.lower()
    logging.debug(f"Reading document: {filename}")
    try:
        if filename.endswith('.pdf'):
            reader = PdfReader(document)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return {"type": "text", "content": text}
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(document)
            return {"type": "dataframe", "content": df.to_dict(orient='records')}
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(document, on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(document, encoding='ISO-8859-1', on_bad_lines='skip')
            return {"type": "dataframe", "content": df.to_dict(orient='records')}
        elif filename.endswith('.json'):
            data = json.load(document)
            df = pd.json_normalize(data)
            return {"type": "dataframe", "content": df.to_dict(orient='records')}
        elif filename.endswith('.xml'):
            tree = ET.parse(document)
            root = tree.getroot()
            data = [{child.tag: child.text for child in elem} for elem in root]
            df = pd.DataFrame(data)
            return {"type": "dataframe", "content": df.to_dict(orient='records')}
        elif filename.endswith('.docx'):
            doc = docx.Document(document)
            text = "\n".join([para.text for para in doc.paragraphs])
            return {"type": "text", "content": text}
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(document)
            text = pytesseract.image_to_string(image)
            return {"type": "image", "content": text}
        elif filename.endswith('.dcm'):
            dicom_data = pydicom.dcmread(document)
            image = dicom_data.pixel_array
            metadata = {elem.description(): str(elem.value) for elem in dicom_data if elem.tag.group != 0x7FE0}  # Exclude pixel data and convert to string
            return {"type": "dicom", "content": image, "metadata": metadata}
        elif filename.endswith('.txt'):
            text = document.read().decode('utf-8')
            return {"type": "text", "content": text}
        else:
            logging.error("Unsupported document type.")
            return {"error": "Unsupported document type."}
    except Exception as e:
        logging.error(f"Error reading document: {e}")
        return {"error": str(e)}

def analyze_document(content):
    logging.debug(f"Analyzing document content of type: {content.get('type')}")
    try:
        if content.get("type") == "text":
            return analyze_text(content["content"])
        elif content.get("type") == "dataframe":
            df = pd.DataFrame(content["content"])
            logging.debug(f"DataFrame created from content: \n{df.head()}")
            text = " ".join(df.apply(lambda x: " ".join(x.dropna().astype(str)), axis=1))
            return {**analyze_text(text), **analyze_dataframe(df)}
        elif content.get("type") == "image":
            return analyze_text(content["content"])
        elif content.get("type") == "dicom":
            return analyze_dicom(content["content"], content["metadata"])
        else:
            logging.error("Unable to analyze content.")
            return {"error": "Unable to analyze content."}
    except Exception as e:
        logging.error(f"Error analyzing document: {e}")
        return {"error": str(e)}

def analyze_text(text):
    try:
        word_count = len(text.split())
        truncated_text = text[:MAX_INPUT_LENGTH]
        blob = TextBlob(truncated_text)
        vader_sentiment = analyzer.polarity_scores(truncated_text)
        
        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "positivity": vader_sentiment['pos'],
            "negativity": vader_sentiment['neg'],
            "neutrality": vader_sentiment['neu'],
            "compound": vader_sentiment['compound']
        }

        summary = summarizer(truncated_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        wordcloud = generate_wordcloud(truncated_text)

        analysis = {
            "word_count": word_count,
            "summary": summary,
            "sentiment": sentiment,
            "wordcloud": wordcloud
        }
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing text: {e}")
        return {"error": str(e)}

def analyze_dataframe(df):
    try:
        if df.empty:
            logging.error("DataFrame is empty.")
            return {"error": "DataFrame is empty."}

        logging.debug(f"DataFrame columns: {df.columns.tolist()}")
        logging.debug(f"DataFrame head:\n{df.head()}")

        description = df.describe().to_dict()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        df_summary = df.describe(include='all').to_string()
        truncated_summary = df_summary[:MAX_INPUT_LENGTH]
        summary = summarizer(truncated_summary, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

        graphs = {}
        if numerical_columns:
            logging.debug(f"Numerical columns: {numerical_columns}")
            numerical_df = df[numerical_columns]
            graphs['histogram'] = generate_histogram(numerical_df)
        if categorical_columns:
            logging.debug(f"Categorical columns: {categorical_columns}")
            categorical_df = df[categorical_columns]
            graphs['bar_chart'] = generate_bar_chart(categorical_df)
            graphs['pie_chart'] = generate_pie_chart(categorical_df)

        analysis = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "description": description,
            "summary": summary,
            "column_summary": {col: {"data_type": str(df[col].dtype), "example_values": df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()} for col in df.columns},
            "graphs": graphs
        }
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Error analyzing dataframe: {e}")
        return {"error": str(e)}

def analyze_dicom(image, metadata):
    try:
        metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
        summary = summarizer(metadata_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('DICOM Image')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        logging.info("DICOM image analyzed successfully")
        return {"analysis": {"metadata": metadata, "summary": summary, "dicom_image": img_base64}}
    except Exception as e:
        logging.error(f"Error analyzing DICOM image: {e}")
        return {"error": str(e)}

def generate_histogram(df):
    try:
        num_columns = len(df.columns)
        if num_columns == 0:
            logging.error("No columns to plot.")
            return None

        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

        if num_columns == 1:
            axes = [axes]  # Ensure axes is always a list
            df.hist(ax=axes[0])
            axes[0].set_title(f'Histogram of {df.columns[0]}')
            axes[0].set_xlabel(df.columns[0])
            axes[0].set_ylabel('Frequency')
        else:
            for i, col in enumerate(df.columns):
                ax = axes[i]
                df[col].hist(ax=ax)
                ax.set_title(f'Histogram of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating histogram: {e}")
        return None

def generate_bar_chart(df):
    try:
        num_columns = len(df.columns)
        if num_columns == 0:
            logging.error("No columns to plot.")
            return None

        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

        for i, col in enumerate(df.columns):
            ax = axes[i]
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Bar Chart of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')

        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating bar chart: {e}")
        return None

def generate_pie_chart(df):
    try:
        num_columns = len(df.columns)
        if num_columns == 0:
            logging.error("No columns to plot.")
            return None

        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

        for i, col in enumerate(df.columns):
            ax = axes[i]
            df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title(f'Pie Chart of {col}')
            ax.set_ylabel('')

        buf = io.BytesIO()
        plt.tight_layout(pad=3.0)
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Error generating pie chart: {e}")
        return None

def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        logging.info("Word cloud generated successfully")
        return img_base64
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        return None
