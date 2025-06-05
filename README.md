# Empowering Healthcare with Generative AI

## Overview

This project presents a cutting-edge healthcare chatbot application that integrates fine-tuned LLaMA and enhanced BioBERT models. Designed to tackle complex medical Q&A, document analysis, and prescription interpretation, it offers scalable, ethical, and high-performing AI solutions for real-world healthcare settings.

##  Key Features

- **Fine-tuned LLaMA for Medical QA**: Trained on 150k+ curated medical Q&A pairs.
- **Enhanced BioBERT with CRF for NER**: Annotated dataset with detailed prescription entities.
- **Advanced OCR & Preprocessing**: Improves extraction from handwritten prescriptions and documents.
- **Voice & Text Interfaces**: Supports multilingual speech interaction and text-based querying.
- **Scalable Modular Architecture**: Built using Flask, Docker, Kubernetes, and asynchronous services.
- **Ethical AI Compliance**: GDPR/HIPAA-aligned anonymization, audit mechanisms, and user consent flows.

##  Directory Structure

Chatbot/
│
├── app.py # Main Flask app entry point
├── services/
│ ├── document_service.py # Document analysis (OCR, summaries, etc.)
│ ├── scraping_service.py # Web scraping and content parsing
│ ├── prescription_service.py # Medical prescription NER via OCR
│ ├── image_service.py # Image-based OCR handling
│ ├── text_service.py # LLaMA-based text generation/Q&A
│ ├── voice_service.py # Speech-to-text and voice output
│ └── search_service.py # API/web search integration
├── templates/
│ └── index.html # Frontend UI
├── requirements.txt # Dependencies



##  Tech Stack

- **Language Models**: LLaMA (fine-tuned), BioBERT (enhanced with CRF)
- **Frameworks**: Flask, Hugging Face Transformers, PyTorch
- **Libraries**: OpenCV, SpeechRecognition, Pytesseract, BeautifulSoup4, WordCloud
- **Deployment**: Docker, Kubernetes, GitHub Actions CI/CD
- **Monitoring**: Prometheus, Grafana

##  Evaluation

- **QA Accuracy**: 88% EM, 91% F1 (better than ChatGPT, Med-PaLM)
- **Response Time**: 1.2s average (vs 2.5s baseline)
- **User Satisfaction**: 4.6/5 (across 280+ users incl. patients & professionals)

##  Compliance

- AES-256 encryption, TLS 1.3
- Anonymization & PII redaction
- Role-based access (RBAC), MFA
- Full HIPAA & GDPR compliance

##  Future Work

- GPT-4 integration via distillation
- Multimodal input (text, image, sensor)
- Real-time health data analytics
- Enhanced personalization & multilingual UX
