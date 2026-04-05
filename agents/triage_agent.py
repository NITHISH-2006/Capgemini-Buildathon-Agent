"""Triage agent: classify incoming requests, detect language, sentiment, and routing."""

from crewai import Agent, LLM
from crewai.tools import tool
import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parents[1]))
from tools.sentiment import detect_sentiment

@tool("Sentiment Detection")
def sentiment_detection_tool(text: str) -> str:
    """Analyze sentiment of text. Returns positive, neutral, or negative and confidence."""
    res = detect_sentiment(text)
    return f"Sentiment: {res['sentiment']}, Confidence: {res['confidence']}"

def create_triage_agent(llm: LLM) -> Agent:
    return Agent(
        role="Triage Intent & Sentiment Specialist",
        goal="Analyze incoming support queries to detect intent, sentiment, and confidently decide whether to route to Resolver, Action or Escalation Agent. Always output structured info including intent, sentiment, confidence score, and recommended_next_agent. If confidence is < 70% or sentiment is strongly negative, recommend Escalation Agent.",
        backstory="You are an expert customer support dispatcher at Capgemini. Your keen eye instantly categorizes user needs and emotional states.",
        verbose=True,
        allow_delegation=False,
        tools=[sentiment_detection_tool],
        llm=llm
    )
