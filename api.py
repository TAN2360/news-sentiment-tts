
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from utils import scrape_news
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os

app = FastAPI()

# Load translation model once
translation_model_name = "Helsinki-NLP/opus-mt-en-hi"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_en_to_hi(text: str) -> str:
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

def generate_tts_audio(text: str, filename="tts_output_hi.mp3") -> str:
    tts = gTTS(text, lang='hi')
    tts.save(filename)
    return filename

@app.get("/")
def root():
    return {"message": "News Summarization and Sentiment Analysis API"}

@app.get("/scrape")
def scrape(company: str = Query(...), num_articles: int = Query(10, ge=10, le=50)):
    result = scrape_news(company, num_articles)
    return JSONResponse(content=result)

@app.get("/translate-summary")
def translate_summary(text: str):
    hindi = translate_en_to_hi(text)
    return {"hindi_translation": hindi}

@app.get("/tts-hindi")
def tts_hindi(text: str):
    file_path = generate_tts_audio(text)
    return FileResponse(path=file_path, filename="tts_output_hi.mp3", media_type="audio/mpeg")
