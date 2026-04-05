"""Resolver agent: use RAG and LLM intelligence to produce customer-facing answers."""

from crewai import Agent, LLM
from crewai.tools import tool
import sys
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parents[1]))
from tools.rag import retrieve_context

@tool("Retrieve FAQ Context")
def rag_tool(query: str) -> str:
    """Useful to search Capgemini FAQ knowledge base for answers to user questions."""
    return retrieve_context(query)

def create_resolver_agent(llm: LLM) -> Agent:
    return Agent(
        role="Customer Support Resolution Expert",
        goal="Provide friendly, accurate, and context-aware answers for simple and general queries using the Retrieve FAQ Context tool. Always base your answers on retrieved context and include your confidence score.",
        backstory="You are a brilliant problem solver at Capgemini. You rely entirely on your documented FAQ knowledge to help users quickly and accurately. You speak professionally and politely.",
        verbose=True,
        allow_delegation=False,
        tools=[rag_tool],
        llm=llm
    )
