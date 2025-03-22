# 📰 News Sentiment Analysis with Hindi Audio Summary

This project fetches news articles for a given company, analyzes sentiment using FinBERT, extracts key topics, and provides audio summaries in Hindi. It includes a Streamlit UI and FastAPI backend.

---

## 🚀 Features

- 🔎 News scraping from NewsAPI
- 🤖 Sentiment Analysis using FinBERT
- 🧠 Topic extraction using spaCy NER + KeyBERT
- 🗣️ Hindi translation and TTS audio summary using MarianMT + gTTS
- 📊 Visual comparisons for sentiment distribution
- 🧩 REST API via FastAPI
- 🌐 Streamlit frontend and API endpoints deployable to Hugging Face Spaces

---

## ⚙️ Setup Instructions

### 📦 Install dependencies
```bash
pip install -r requirements.txt
```

### 🔌 Run Streamlit App
```bash
streamlit run app.py
```

### 🔌 Run FastAPI Server
```bash
uvicorn api:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to test APIs with Swagger UI.

---

## 🧠 Model Details

| Task                | Model |
|---------------------|-----------------------------|
| Sentiment Analysis  | `yiyanghkust/finbert-tone` (HuggingFace Transformers) |
| Topic Extraction    | `spaCy` (`en_core_web_trf`) + `KeyBERT` |
| Hindi Translation   | `Helsinki-NLP/opus-mt-en-hi` |
| Hindi TTS           | `gTTS` (Google Text-to-Speech) |

---

## 🔌 API Development

`api.py` exposes 3 endpoints:

| Endpoint               | Method | Description                                |
|------------------------|--------|--------------------------------------------|
| `/scrape`              | GET    | Fetch and analyze articles (via utils.py)  |
| `/translate-summary`   | GET    | Translate English text to Hindi            |
| `/tts-hindi`           | GET    | Generate Hindi speech for given text       |

---


## 🔗 Third-Party APIs Used

- 📰 **NewsAPI.org** — to fetch real-time articles
- 🤖 **Hugging Face Models** — via `transformers` library
- 🔊 **gTTS** — converts translated text to Hindi speech

---

## ✅ Assumptions & Limitations

- Summaries are extracted from article descriptions (`description` field from NewsAPI).
- Hindi translation assumes neutral sentence structure (no grammar adaptation).
- TTS output is generated as one combined MP3 per company.
- Currently supports up to 50 articles per company for performance reasons.

---

## 📂 Project Structure

```
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend
├── utils.py                # Data scraping and processing
├── requirements.txt
├── README.md

```

---

## ☁️ Deployment on Hugging Face

1. Push this repo to GitHub.
2. Create a new [Hugging Face Space](https://huggingface.co/spaces)
3. Select:
   - SDK: **Streamlit** or **FastAPI**
   - Repo URL: link to your GitHub
4. Hugging Face will auto-deploy from `requirements.txt`.

---


Built with ❤️ for modern multilingual analysis.
