import gradio as gr
import pytesseract
from PIL import Image
import tensorflow as tf
import pickle
import numpy as np
import re
from transformers import pipeline
import requests
import cohere

GEMINI_API_KEY = ''
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
COHERE_API_KEY = ""
co = cohere.Client(COHERE_API_KEY)

MODEL_PATH = "/model.h5"
sentiment_model = tf.keras.models.load_model(MODEL_PATH)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

LANGUAGES = {
    "English": "eng",
    "Hindi": "hin",
    "Tamil": "tam",
    "Urdu": "urd",
    "Bengali": "ben",
    "Telugu": "tel"
}

CATEGORIES = ["Medical", "Event", "Politics", "Sports", "Finance", "Entertainment", "Technology", "Education"]

def extract_text(image, language):
    if not language:
        return "⚠️ Please select a language!"
    try:
        lang_code = LANGUAGES.get(language, "eng")
        text = pytesseract.image_to_string(image, lang=lang_code)
        return text if text.strip() else "⚠️ No text detected. Try another language."
    except Exception as e:
        return f"❌ Error extracting text: {e}\nEnsure Tesseract OCR is installed and configured."

def analyze_sentiment(text):
    if not text.strip():
        return "⚠️ No text provided for sentiment analysis."
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    prediction = sentiment_model.predict(padded_sequence)[0][0]
    return "😊 Positive" if prediction >= 0.5 else "😞 Negative"

def gemini_generate(prompt, system_instruction=None):
    headers = {'Content-Type': 'application/json'}
    params = {'key': GEMINI_API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    if system_instruction:
        data["system_instruction"] = system_instruction
    try:
        res = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Gemini API error: {e}"

def translate_to_english(text, src_lang):
    if not text.strip():
        return "⚠️ No text provided for translation."
    prompt = f"Translate the following text from {src_lang} to English.\nText: {text}"
    return gemini_generate(prompt)

def summarize_with_gemini(english_text):
    if not english_text.strip():
        return "⚠️ No text provided for summarization."
    prompt = f"Summarize the following English text in a concise paragraph.\nText: {english_text}"
    return gemini_generate(prompt)

def translate_from_english(text, target_lang):
    if not text.strip():
        return "⚠️ No text provided for translation."
    prompt = f"Translate the following English text to {target_lang}.\nText: {text}"
    return gemini_generate(prompt)

def full_nlp_pipeline(image, input_lang, output_lang):
    lang_code = LANGUAGES.get(input_lang, "eng")
    if not input_lang in LANGUAGES:
        return f"❌ Unsupported input language: {input_lang}", "", ""
    try:
        extracted = pytesseract.image_to_string(image, lang=lang_code)
    except Exception as e:
        return f"❌ OCR error: {e}", "", ""
    if not extracted.strip():
        return "⚠️ No text detected.", "", ""
    
    if input_lang != "English":
        english = translate_to_english(extracted, input_lang)
    else:
        english = extracted
    
    summary = summarize_with_gemini(english)
    
    if output_lang != "English":
        if not output_lang in LANGUAGES:
            return extracted, summary, f"❌ Unsupported output language: {output_lang}"
        summary_out = translate_from_english(summary, output_lang)
    else:
        summary_out = summary
    
    return extracted, summary, summary_out

def classify_text(text):
    if not text.strip(): 
        return "⚠️ No text provided for classification."
    result = classifier(text, candidate_labels=CATEGORIES)
    best_category = result["labels"][0]
    confidence = result["scores"][0]
    return f"**Category:** {best_category} (Confidence: {confidence:.2f})"

def extract_dates(text):
    if not text.strip():
        return "⚠️ No text provided for date extraction."
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b"
    ]
    extracted_dates = []
    for pattern in date_patterns:
        extracted_dates.extend(re.findall(pattern, text))
    
    return ", ".join(extracted_dates) if extracted_dates else "No dates found."

with gr.Blocks(theme="soft") as app:
    gr.Markdown("# 📖 Multilingual NLP Translation & Summarization Suite")
    gr.Markdown("**Extract Text, Translate, Summarize, and Re-translate Effortlessly.**")

    with gr.Group():
        gr.Markdown("### 🖼️ Upload Image for Text Extraction and Translation")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image", height=300)
            input_lang_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label="Input Language")
            output_lang_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label="Output Language")
        process_btn = gr.Button("🚀 Process Pipeline", variant="primary")
        extracted_text_output = gr.Textbox(label="Extracted Text (OCR)", interactive=False)
        summary_en_output = gr.Textbox(label="Summary (English)", interactive=False)
        summary_out_output = gr.Textbox(label="Summary (Output Language)", interactive=False)

        process_btn.click(full_nlp_pipeline, inputs=[image_input, input_lang_dropdown, output_lang_dropdown], outputs=[extracted_text_output, summary_en_output, summary_out_output])

    with gr.Group():
        gr.Markdown("### 💬 Sentiment Analysis")
        sentiment_output = gr.Textbox(label="Sentiment Result", interactive=False)
        sentiment_btn = gr.Button("📊 Analyze Sentiment", variant="secondary")
        sentiment_btn.click(analyze_sentiment, inputs=extracted_text_output, outputs=sentiment_output)

    with gr.Group():
        gr.Markdown("### 📂 Document Classification")
        classification_output = gr.Textbox(label="Category", interactive=False)
        classify_btn = gr.Button("🗂 Classify Text", variant="secondary")
        classify_btn.click(classify_text, inputs=extracted_text_output, outputs=classification_output)

    with gr.Group():
        gr.Markdown("### 📅 Important Date Extraction")
        date_output = gr.Textbox(label="Extracted Dates", interactive=False)
        date_btn = gr.Button("📆 Extract Dates", variant="primary")
        date_btn.click(extract_dates, inputs=extracted_text_output, outputs=date_output)

app.launch()