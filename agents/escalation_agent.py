"""Escalation agent: detect when cases should be handed off to human support."""

from crewai import Agent, LLM
from crewai.tools import tool
import sys
from pathlib import Path
import json

# Fix imports
sys.path.append(str(Path(__file__).parents[1]))
from tools.actions import create_ticket

@tool("Escalate Ticket Action")
def escalate_ticket_tool(input_str: str) -> str:
    """Create an escalated support ticket for humans. Input must be formatted as 'customer_name | issue'"""
    parts = input_str.split("|")
    if len(parts) == 2:
        res = create_ticket(parts[0].strip(), parts[1].strip())
        return json.dumps(res)
    return "Error: format must be 'customer_name | issue'"

def create_escalation_agent(llm: LLM) -> Agent:
    return Agent(
        role="Human Handoff & Escalation Manager",
        goal="Receive complex or negatively scored queries that triage failed to solve confidently. Create an escalated support ticket for the user using the tool and give them a polite, reassuring handoff message including their Ticket ID.",
        backstory="You are a senior customer success manager at Capgemini. You excel at de-escalating tense situations and warmly assuring customers that human experts are taking over their problem immediately.",
        verbose=True,
        allow_delegation=False,
        tools=[escalate_ticket_tool],
        llm=llm
    )
