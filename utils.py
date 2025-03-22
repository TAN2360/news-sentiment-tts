import requests
import torch
import spacy
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Load Enhanced NLP Models
nlp = spacy.load("en_core_web_trf")  # Transformer-based model for better accuracy
kw_model = KeyBERT()

# Load FinBERT tokenizer & model
MODEL_NAME = "yiyanghkust/finbert-tone"  # More accurate for news sentiment
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Load VADER for fallback (optional)
sia = SentimentIntensityAnalyzer()

NEWS_API_KEY = "3cd146b5de2f4a1e82f72e040871958c"

def get_sentiment_finbert(text):
    """Perform sentiment analysis using a fine-tuned FinBERT model."""
    
    if not text.strip():
        return "Neutral"
    
    # Preprocess: Limit input length for better accuracy
    text = text[:512]  # Truncate to 512 characters to prevent model overload
    
    # Tokenize and classify sentiment
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs).logits

    # Apply softmax to get probabilities
    sentiment_probs = torch.nn.functional.softmax(outputs, dim=-1).tolist()[0]

    # Map sentiment categories: [Positive, Neutral, Negative]
    sentiments = ["Positive", "Neutral", "Negative"]
    
    # Get highest probability sentiment
    sentiment = sentiments[sentiment_probs.index(max(sentiment_probs))]

    # Confidence threshold adjustment
    confidence_threshold = 0.7  # Avoid misclassification due to low confidence
    max_conf = max(sentiment_probs)
    
    if max_conf < confidence_threshold:
        return "Neutral"  # Default to Neutral if confidence is too low

    return sentiment

def extract_focus_topics(text):
    """Extract key topics using refined NER and KeyBERT."""
    if not text.strip():
        return ["No Key Topics Found"]
    
    doc = nlp(text)
    
    # Extract named entities (focus on important categories)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT", "LAW"]]

    # Extract KeyBERT keywords (boosts relevance)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=7)
    keyword_list = [kw[0] for kw in keywords]

    # Merge entities and keywords, ensuring diversity
    topics = list(set(entities + keyword_list))

    # Remove irrelevant words
    blacklist_words = {"news", "article", "company", "business", "industry", "report", "source"}
    filtered_topics = [topic for topic in topics if topic.lower() not in blacklist_words][:3]  # Keep top 3

    return filtered_topics if filtered_topics else ["No Key Topics Found"]

def analyze_topic_overlap(articles):
    """Analyze common and unique topics across articles."""
    all_topics = [set(article["focus_topics"]) for article in articles]

    # Find common topics (topics appearing in at least 2 articles)
    common_topics = set.intersection(*all_topics) if len(all_topics) > 1 else set()

    # Identify unique topics per article
    unique_topics_per_article = []
    for i, article_topics in enumerate(all_topics):
        unique_topics = article_topics - common_topics
        unique_topics_per_article.append({
            f"Unique Topics in Article {i + 1}": list(unique_topics)
        })

    return {
        "Common Topics": list(common_topics),
        "Unique Topics Per Article": unique_topics_per_article
    }

def compare_articles(articles):
    """Generate coverage differences between consecutive articles."""
    coverage_differences = []

    for i in range(len(articles) - 1):
        article1 = articles[i]
        article2 = articles[i + 1]

        comparison = f"Article {i+1} highlights {', '.join(article1['focus_topics'])}, while Article {i+2} discusses {', '.join(article2['focus_topics'])}."
        impact = f"The first article focuses on {article1['sentiment'].lower()} aspects, whereas the second highlights {article2['sentiment'].lower()} factors."

        coverage_differences.append({
            "Comparison": comparison,
            "Impact": impact
        })

    return coverage_differences

def generate_final_sentiment_analysis(sentiment_summary, company_name=None):
    """Generate a final sentiment conclusion with confidence and icons."""
    total = sentiment_summary["Total Articles"]
    pos = sentiment_summary["Positive"]
    neu = sentiment_summary["Neutral"]
    neg = sentiment_summary["Negative"]

    if total == 0:
        return "No sentiment data available."

    majority = max(pos, neu, neg)
    sentiment = (
        "positive" if pos == majority else
        "neutral" if neu == majority else
        "negative"
    )

    confidence = round((majority / total) * 100)
    icon = "📈" if sentiment == "positive" else "📉" if sentiment == "negative" else "🔄"

    subject = f"{company_name}’s latest news coverage" if company_name else f"{total} articles analyzed"
    base = f"{icon} {subject} is mostly {sentiment} ({confidence}% of articles)."

    if sentiment == "positive":
        extra = " Potential stock growth expected."
    elif sentiment == "negative":
        extra = " Possible market concern or declining sentiment."
    else:
        extra = " The market outlook appears mixed or stable."

    return base + extra


def scrape_news(company_name, num_articles=10):
    """Fetch news articles, analyze sentiment, and extract topics."""
    num_articles = max(10, min(num_articles, 50))  # Limit between 10 and 50

    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize={num_articles}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": f"Failed to fetch news. Status Code: {response.status_code}, Response: {response.text}"}
    
    articles = response.json().get("articles", [])
    
    if not articles:
        return {"error": "No articles found for the given company."}

    news_list = []
    sentiment_counts = Counter()
    all_focus_topics = []

    for article in articles:
        title = article.get("title", "No Title")
        summary = (article.get("description") or "").strip() or "No Summary"

        # Use FinBERT for sentiment analysis
        sentiment = get_sentiment_finbert(summary) if summary != "No Summary" else "Neutral"
        sentiment_counts[sentiment] += 1  # Count sentiment occurrences

        # Extract Focus Topics
        focus_topics = extract_focus_topics(summary)
        all_focus_topics.extend(focus_topics)

        news_list.append({
            "title": title,
            "summary": summary,
            "sentiment": sentiment,
            "focus_topics": focus_topics
        })

    # Analyze topic overlap
    topic_analysis = analyze_topic_overlap(news_list)

    # Generate coverage differences
    coverage_differences = compare_articles(news_list)

    # Generate final sentiment conclusion
    final_sentiment_analysis = generate_final_sentiment_analysis({
        "Positive": sentiment_counts["Positive"],
        "Negative": sentiment_counts["Negative"],
        "Neutral": sentiment_counts["Neutral"],
        "Total Articles": num_articles
    }, company_name=company_name)

    # Generate sentiment summary
    sentiment_summary = {
        "Positive": sentiment_counts["Positive"],
        "Negative": sentiment_counts["Negative"],
        "Neutral": sentiment_counts["Neutral"],
        "Total Articles": num_articles,
        "Topic Overlap": topic_analysis,
        "Coverage Differences": coverage_differences,
        "Final Sentiment Analysis": final_sentiment_analysis
    }

    return {"articles": news_list, "sentiment_summary": sentiment_summary}
