import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from utils import scrape_news
from collections import Counter
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os

# Load translation model once
translation_model_name = "Helsinki-NLP/opus-mt-en-hi"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_en_to_hi(text):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

def generate_hindi_tts(articles, filename="tts_output_hi.mp3"):
    combined_text = " ".join([translate_en_to_hi(article['summary']) for article in articles if article['summary']])
    tts = gTTS(combined_text, lang='hi')
    tts.save(filename)
    return filename

# Streamlit UI
st.title("News Sentiment and Comparative Analysis")

# Clear cache on page refresh
st.cache_resource.clear()
st.cache_data.clear()

# User Input
col1, col2 = st.columns(2)
with col1:
    company1 = st.text_input("Enter First Company Name:")
with col2:
    company2 = st.text_input("Enter Second Company Name (Optional):")

num_articles = st.slider("Number of Articles:", min_value=10, max_value=50, value=10)

# Define Colors
COLORS = ["#4CAF50", "#FFC107", "#F44336"]

# Convert Data to CSV
def convert_to_csv(data):
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")

# Convert Data to JSON
def convert_to_json(data):
    return json.dumps(data, indent=4).encode("utf-8")

# Maintain Session State to Prevent Reset
if "results" not in st.session_state:
    st.session_state.results = None


# Analyze Sentiment Button
if st.button("Analyze Sentiment", key="analyze_sentiment"):
    if not company1.strip():
        st.error("Please enter at least one company name.")
    else:
        with st.spinner(f"Fetching news for {company1}..."):
            result1 = scrape_news(company1, num_articles)

        if "error" in result1:
            st.error(result1["error"])
        else:
            st.session_state.results = {"company1": result1, "company2": None}

            if company2.strip():
                with st.spinner(f"Fetching news for {company2}..."):
                    result2 = scrape_news(company2, num_articles)

                if "error" not in result2:
                    st.session_state.results["company2"] = result2

# If Analysis Results Exist, Show Data
if st.session_state.results:
    result1 = st.session_state.results["company1"]
    result2 = st.session_state.results["company2"]

    # Ensure safe defaults for company2-related variables
    sentiment_summary2 = None
    articles2 = []
    coverage_differences2 = []
    common_topics2 = []
    unique_topics2 = []

    articles1 = result1["articles"]
    sentiment_summary1 = result1["sentiment_summary"]
    coverage_differences1 = sentiment_summary1.get("Coverage Differences", [])

    if company2.strip() and result2 and "sentiment_summary" in result2:
        sentiment_summary2 = result2["sentiment_summary"]
        coverage_differences2 = sentiment_summary2.get("Coverage Differences", [])
        articles2 = result2["articles"]

        all_topics2 = [topic for article in articles2 for topic in article.get("focus_topics", [])]
        common_topics2 = [topic for topic, count in Counter(all_topics2).items() if count > 1]
        unique_topics2 = list(set(all_topics2) - set(common_topics2))
    else:
        sentiment_summary2 = None
        coverage_differences2 = []
        articles2 = []
        common_topics2 = []
        unique_topics2 = []

    all_topics1 = [topic for article in articles1 for topic in article.get("focus_topics", [])]
    common_topics1 = [topic for topic, count in Counter(all_topics1).items() if count > 1]
    unique_topics1 = list(set(all_topics1) - set(common_topics1))

    # Display Sentiment Comparison Graphs
    st.subheader("Sentiment Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{company1}")
        labels1 = ["Positive", "Neutral", "Negative"]
        sizes1 = [sentiment_summary1["Positive"], sentiment_summary1["Neutral"], sentiment_summary1["Negative"]]
        fig, ax = plt.subplots()
        ax.pie(sizes1, labels=labels1, autopct="%1.1f%%", colors=COLORS, startangle=140)
        ax.axis("equal")
        st.pyplot(fig)

    if company2.strip() and result2 and "sentiment_summary" in result2:
        with col2:
            st.subheader(f"{company2}")
            labels2 = ["Positive", "Neutral", "Negative"]
            sizes2 = [sentiment_summary2["Positive"], sentiment_summary2["Neutral"], sentiment_summary2["Negative"]]
            fig, ax = plt.subplots()
            ax.pie(sizes2, labels=labels2, autopct="%1.1f%%", colors=COLORS, startangle=140)
            ax.axis("equal")
            st.pyplot(fig)

    # Display Sentiment Bar Chart
    st.subheader("Comparative Sentiment Score")
    sizes1 = [sentiment_summary1["Positive"], sentiment_summary1["Neutral"], sentiment_summary1["Negative"]]
    sentiment_comparison = {
        "Sentiment": ["Positive", "Neutral", "Negative"],
        f"{company1}": sizes1,
    }

    if company2.strip() and sentiment_summary2:
        sizes2 = [sentiment_summary2["Positive"], sentiment_summary2["Neutral"], sentiment_summary2["Negative"]]
        sentiment_comparison[f"{company2}"] = sizes2

    import altair as alt
    df_comparison = pd.DataFrame(sentiment_comparison)
    chart = alt.Chart(df_comparison).transform_fold(
        fold=[col for col in df_comparison.columns if col != "Sentiment"],
        as_=["Company", "Value"]
    ).mark_bar().encode(
        x=alt.X('Sentiment:N', title='Sentiment'),
        y=alt.Y('Value:Q', title='Number of Articles'),
        color='Company:N',
        tooltip=['Company:N', 'Sentiment:N', 'Value:Q']
    ).properties(width=600, height=400)

    st.altair_chart(chart, use_container_width=True)

    # Display Extracted Articles
    st.subheader("Extracted Articles")
    if company2.strip() and result2 and "sentiment_summary" in result2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{company1} Articles")
            for i, article in enumerate(articles1, start=1):
                st.markdown(f"""
<div style='padding:10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); border-radius:10px; margin-bottom:10px'>
<b>Title:</b> {article['title']}<br>
<b>Summary:</b> {article['summary']}<br>
<b>Sentiment:</b> {article['sentiment']}<br>
<b>Topics:</b> {', '.join(article['focus_topics'])}
</div>
""", unsafe_allow_html=True)
        with col2:
            st.subheader(f"{company2} Articles")
            for i, article in enumerate(articles2, start=1):
                st.markdown(f"""
<div style='padding:10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); border-radius:10px; margin-bottom:10px'>
<b>Title:</b> {article['title']}<br>
<b>Summary:</b> {article['summary']}<br>
<b>Sentiment:</b> {article['sentiment']}<br>
<b>Topics:</b> {', '.join(article['focus_topics'])}
</div>
""", unsafe_allow_html=True)
    else:
        for i, article in enumerate(articles1, start=1):
            with st.expander(f"**Article {i}: {article['title']}**"):
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Sentiment:** {article['sentiment']}")
                st.write(f"**Topics:** {', '.join(article['focus_topics'])}")

    # Display Coverage Differences
    st.subheader("Coverage Differences")
    for diff in coverage_differences1:
        st.write(f"**Comparison:** {diff['Comparison']}")
        st.write(f"**Impact:** {diff['Impact']}")
        st.markdown("---")

    if company2.strip() and result2 and "sentiment_summary" in result2 and coverage_differences2:
        for diff in coverage_differences2:
            st.write(f"**Comparison:** {diff['Comparison']}")
            st.write(f"**Impact:** {diff['Impact']}")
            st.markdown("---")

    # Display Topic Overlap
    st.subheader("Final Topic Overlap")
    st.write(f"**Common Topics (All Articles):** {common_topics1}")
    st.write(f"**Unique Topics (All Articles):** {unique_topics1}")

    if company2.strip() and result2 and "sentiment_summary" in result2 and common_topics2:
        st.write(f"**Common Topics ({company2}):** {common_topics2}")
        st.write(f"**Unique Topics ({company2}):** {unique_topics2}")

    # Display Final Sentiment Analysis
    st.subheader("Final Sentiment Analysis")
    st.write(f"{sentiment_summary1.get('Final Sentiment Analysis', 'No final analysis available.')}")

    if company2.strip() and result2 and "sentiment_summary" in result2:
        st.write(f"{sentiment_summary2.get('Final Sentiment Analysis', 'No final analysis available.')}")
    else:
        st.write(f"No final analysis available for {company2}.")


# Sidebar TTS Playback
with st.sidebar:
    st.subheader("🗣️ Summarized Audio (Hindi)")

    if st.button("🔊 Play Hindi Audio Summary (Company 1)", key="tts_c1") and articles1:
        tts_file = generate_hindi_tts(articles1, filename="tts_c1.mp3")
        with open(tts_file, "rb") as f:
            st.audio(f.read(), format="audio/mp3")

    if company2.strip() and articles2:
        if st.button("🔊 Play Hindi Audio Summary (Company 2)", key="tts_c2"):
            tts_file = generate_hindi_tts(articles2, filename="tts_c2.mp3")
            with open(tts_file, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

# Sidebar for Download Reports
with st.sidebar:
    st.subheader("Download Reports")

    if "articles1" in locals():
        st.download_button(f"Download {company1} CSV", convert_to_csv(articles1), f"{company1}_news_sentiment.csv", "text/csv")
        st.download_button(f"Download {company1} JSON", convert_to_json(articles1), f"{company1}_news_sentiment.json", "application/json")

    if company2.strip() and result2 and "articles" in result2 and articles2:
        st.download_button(f"Download {company2} CSV", convert_to_csv(articles2), f"{company2}_news_sentiment.csv", "text/csv")
        st.download_button(f"Download {company2} JSON", convert_to_json(articles2), f"{company2}_news_sentiment.json", "application/json")
