"""Sentiment detection utilities using HuggingFace or fallback logic."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Attempt to load HuggingFace pipeline lazily
_sentiment_pipeline = None

def get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            _sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace pipeline: {e}. Falling back to simple detection.")
            _sentiment_pipeline = "fallback"
    return _sentiment_pipeline

def fallback_detect_sentiment(text: str) -> Dict[str, str]:
    lower = text.lower()
    if any(x in lower for x in ["not happy", "angry", "frustrated", "bad", "disappointed", "issue", "problem"]):
        sentiment = "negative"
    elif any(x in lower for x in ["thank", "great", "happy", "good", "awesome", "perfect"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment, "confidence": 0.70}

def detect_sentiment(text: str) -> Dict[str, str]:
    pipe = get_pipeline()
    if pipe == "fallback":
        return fallback_detect_sentiment(text)
    
    try:
        results = pipe(text)
        result = results[0] # [{'label': 'POS/NEG/NEU', 'score': 0.99}]
        label = result['label'].lower()
        if label == 'pos':
            sentiment = "positive"
        elif label == 'neg':
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {"sentiment": sentiment, "confidence": round(result['score'], 2)}
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return fallback_detect_sentiment(text)
