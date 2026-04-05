"""Action agent: perform autonomous actions like password reset and ticket creation."""

from crewai import Agent, LLM
from crewai.tools import tool
import sys
from pathlib import Path
import json

# Fix imports
sys.path.append(str(Path(__file__).parents[1]))
from tools.actions import password_reset, create_ticket

@tool("Password Reset Action")
def password_reset_tool(username: str) -> str:
    """Reset the password for a user. Input should be the exact username."""
    res = password_reset(username)
    return json.dumps(res)

@tool("Create Ticket Action")
def create_ticket_tool(input_str: str) -> str:
    """Create a support ticket. Input should be a JSON-like string with 'customer' and 'issue' keys, e.g., customer|issue"""
    parts = input_str.split("|")
    if len(parts) == 2:
        res = create_ticket(parts[0].strip(), parts[1].strip())
        return json.dumps(res)
    return "Error: format must be 'customer | issue'"

def create_action_agent(llm: LLM) -> Agent:
    return Agent(
        role="Autonomous Action Executor",
        goal="Execute required autonomous actions based on user needs. If they need a password reset, use the tool and confirm the new temporary password to them clearly. If they need a ticket without escalation, use the ticket tool.",
        backstory="You are the action engine of Capgemini's IT team. Your precision in executing password resets and ticket creation ensures flawless automated IT ops.",
        verbose=True,
        allow_delegation=False,
        tools=[password_reset_tool, create_ticket_tool],
        llm=llm
    )
