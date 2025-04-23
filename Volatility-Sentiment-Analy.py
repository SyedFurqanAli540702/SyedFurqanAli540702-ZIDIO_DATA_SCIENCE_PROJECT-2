import json
import pandas as pd
import numpy as np
from textblob import TextBlob

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Sample sentiment data (simulated news/social media)
sentiment_data = [
    {"date": "2025-04-20", "text": "Bitcoin is surging again! Investors are optimistic."},
    {"date": "2025-04-21", "text": "Crypto market crash expected due to new regulations."},
    {"date": "2025-04-22", "text": "Ethereum shows strong resilience amid market uncertainty."},
    {"date": "2025-04-23", "text": "Mixed reactions in the crypto community about altcoin performance."}
]

# Add sentiment scores
for item in sentiment_data:
    item["sentiment_score"] = analyze_sentiment(item["text"])

# Save to JSON file
with open("volatility_sent_
